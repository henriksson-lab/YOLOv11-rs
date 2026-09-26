use anyhow::{Context, Result};
use image::{imageops, Rgb, RgbImage};
use std::path::PathBuf;
use std::sync::Arc;

use super::dataset::{Dataset, Label};

/// Host-side geometric augmentation used by the source-neutral trainer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GeometryPolicy {
    /// Letterbox only. Useful as a controlled augmentation baseline.
    #[default]
    None,
    /// Cycle through all eight exact square symmetries over eight epochs.
    D4,
}

/// One detection target in source-image pixel coordinates.
#[derive(Clone, Copy, Debug)]
pub struct BoxTarget {
    pub class: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Source-neutral image and targets. Storage adapters return this type before
/// YOLO resizing and augmentation.
#[derive(Clone, Debug)]
pub struct RawSample {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub targets: Vec<BoxTarget>,
}

/// Storage boundary used by the multithreaded trainer.
///
/// An OME-Zarr implementation can override `epoch_keys` to retain spatial
/// locality and `prefetch` to warm its decoded chunk cache. Augmentation stays
/// in this crate and therefore has identical target semantics for every store.
pub trait SampleStore: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn load(&self, key: usize) -> Result<RawSample>;

    fn epoch_keys(&self, _epoch: usize, _seed: u64) -> Vec<usize> {
        (0..self.len()).collect()
    }

    fn prefetch(&self, _keys: &[usize]) -> Result<()> {
        Ok(())
    }
}

/// File-backed implementation used by the command-line trainer.
pub struct FileSampleStore {
    filenames: Vec<PathBuf>,
    labels: Vec<Vec<Label>>,
}

impl FileSampleStore {
    pub fn new(filenames: Vec<PathBuf>) -> Result<Self> {
        let records = Dataset::load_label(&filenames)?;
        let (filenames, labels) = records.into_iter().unzip();
        Ok(Self { filenames, labels })
    }

    pub fn from_dataset(dataset: &Dataset) -> Self {
        Self {
            filenames: dataset.filenames.clone(),
            labels: dataset.labels.clone(),
        }
    }
}

impl SampleStore for FileSampleStore {
    fn len(&self) -> usize {
        self.filenames.len()
    }

    fn load(&self, key: usize) -> Result<RawSample> {
        let path = self
            .filenames
            .get(key)
            .with_context(|| format!("sample key {key} is out of range"))?;
        let image = image::open(path)
            .with_context(|| format!("failed to load image {}", path.display()))?
            .to_rgb8();
        let (width, height) = image.dimensions();
        let targets = self.labels[key]
            .iter()
            .map(|label| BoxTarget {
                class: label.class,
                x1: (label.cx - label.w * 0.5) * width as f32,
                y1: (label.cy - label.h * 0.5) * height as f32,
                x2: (label.cx + label.w * 0.5) * width as f32,
                y2: (label.cy + label.h * 0.5) * height as f32,
            })
            .collect();
        Ok(RawSample {
            rgb: image.into_raw(),
            width: width as usize,
            height: height as usize,
            targets,
        })
    }

    fn epoch_keys(&self, epoch: usize, seed: u64) -> Vec<usize> {
        let mut keys: Vec<_> = (0..self.len()).collect();
        let mut state = seed ^ (epoch as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for index in (1..keys.len()).rev() {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            let selected = ((value ^ (value >> 31)) as usize) % (index + 1);
            keys.swap(index, selected);
        }
        keys
    }
}

#[derive(Clone, Debug)]
pub struct PreparedSample {
    /// RGB, channel first, normalized to 0..1.
    pub image: Vec<f32>,
    pub cls: Vec<f32>,
    pub bbox: Vec<[f32; 4]>,
}

/// Prepare one sample with a letterbox and one of the eight exact square
/// symmetries. These transforms introduce no interpolation beyond the initial
/// resize and require no additional source reads.
pub fn prepare_sample(
    raw: RawSample,
    input_size: usize,
    orientation: u8,
) -> Result<PreparedSample> {
    let source = RgbImage::from_raw(raw.width as u32, raw.height as u32, raw.rgb)
        .context("raw RGB buffer has the wrong size")?;
    let scale = (input_size as f32 / raw.width as f32).min(input_size as f32 / raw.height as f32);
    let resized_width = ((raw.width as f32 * scale).round() as usize).max(1);
    let resized_height = ((raw.height as f32 * scale).round() as usize).max(1);
    let resized = imageops::resize(
        &source,
        resized_width as u32,
        resized_height as u32,
        imageops::FilterType::Triangle,
    );
    let pad_x = (input_size - resized_width) / 2;
    let pad_y = (input_size - resized_height) / 2;
    let mut image = RgbImage::from_pixel(input_size as u32, input_size as u32, Rgb([114; 3]));
    imageops::replace(&mut image, &resized, pad_x as i64, pad_y as i64);

    let mut labels = Vec::with_capacity(raw.targets.len());
    for target in raw.targets {
        let x1 = (target.x1 * scale + pad_x as f32).clamp(0.0, input_size as f32);
        let y1 = (target.y1 * scale + pad_y as f32).clamp(0.0, input_size as f32);
        let x2 = (target.x2 * scale + pad_x as f32).clamp(0.0, input_size as f32);
        let y2 = (target.y2 * scale + pad_y as f32).clamp(0.0, input_size as f32);
        if x2 > x1 && y2 > y1 {
            labels.push((
                target.class,
                [
                    (x1 + x2) * 0.5 / input_size as f32,
                    (y1 + y2) * 0.5 / input_size as f32,
                    (x2 - x1) / input_size as f32,
                    (y2 - y1) / input_size as f32,
                ],
            ));
        }
    }

    let reflected = orientation >= 4;
    let rotations = orientation % 4;
    if reflected {
        image = imageops::flip_horizontal(&image);
        for (_, bbox) in &mut labels {
            bbox[0] = 1.0 - bbox[0];
        }
    }
    for _ in 0..rotations {
        image = imageops::rotate90(&image);
        for (_, bbox) in &mut labels {
            let [cx, cy, width, height] = *bbox;
            *bbox = [1.0 - cy, cx, height, width];
        }
    }

    let pixels = image.as_raw();
    let area = input_size * input_size;
    let mut chw = vec![0.0f32; 3 * area];
    for pixel in 0..area {
        chw[pixel] = pixels[pixel * 3] as f32 / 255.0;
        chw[area + pixel] = pixels[pixel * 3 + 1] as f32 / 255.0;
        chw[2 * area + pixel] = pixels[pixel * 3 + 2] as f32 / 255.0;
    }
    let (cls, bbox) = labels.into_iter().unzip();
    Ok(PreparedSample {
        image: chw,
        cls,
        bbox,
    })
}

pub type SharedSampleStore = Arc<dyn SampleStore>;
