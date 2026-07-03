use anyhow::{Context, Result};
use burn::prelude::*;
use image::{GenericImageView, ImageFormat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use crate::data::augment;
use crate::data::resize;

/// A single label entry: (class, cx, cy, w, h) all in [0,1] normalized coords.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    pub class: f32,
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

/// Dataset for YOLO training/evaluation.
pub struct Dataset {
    pub filenames: Vec<PathBuf>,
    pub labels: Vec<Vec<Label>>,
    pub input_size: u32,
    pub augment: bool,
    pub mosaic: bool,
    pub params: AugmentParams,
    pub albumentations: Albumentations,
}

/// A single sample returned by the dataset.
pub struct Sample<B: Backend> {
    /// Image tensor [3, H, W] in 0..1 range.
    pub image: Tensor<B, 3>,
    /// Class labels [N].
    pub cls: Vec<f32>,
    /// Bounding boxes [N, 4] in normalized (cx, cy, w, h).
    pub bbox: Vec<[f32; 4]>,
}

pub struct Albumentations;

impl Albumentations {
    pub fn new() -> Self {
        Self
    }

    pub fn call(
        &self,
        image: image::RgbImage,
        box_: Vec<[f32; 4]>,
        cls: Vec<f32>,
    ) -> (image::RgbImage, Vec<[f32; 4]>, Vec<f32>) {
        (image, box_, cls)
    }
}

impl Dataset {
    /// Create a new dataset.
    pub fn new(
        filenames: Vec<PathBuf>,
        input_size: u32,
        augment: bool,
        params: &AugmentParams,
    ) -> Result<Self> {
        let records = Self::load_label(&filenames)?;
        let (filenames, labels): (Vec<_>, Vec<_>) = records.into_iter().unzip();
        Ok(Self {
            filenames,
            labels,
            input_size,
            augment,
            mosaic: augment,
            params: params.clone(),
            albumentations: Albumentations::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.filenames.len()
    }

    /// Get a single sample (with augmentation if enabled).
    pub fn get_item<B: Backend>(&self, index: usize, device: &B::Device) -> Result<Sample<B>> {
        let to_sample = |img: &image::RgbImage| -> Tensor<B, 3> {
            let (w, h) = img.dimensions();
            let raw = img.as_raw();
            let hw = (h * w) as usize;
            let mut sample = vec![0.0f32; 3 * hw];
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let idx = (y * w as usize + x) * 3;
                    sample[y * w as usize + x] = raw[idx + 2] as f32;
                    sample[hw + y * w as usize + x] = raw[idx + 1] as f32;
                    sample[2 * hw + y * w as usize + x] = raw[idx] as f32;
                }
            }
            Tensor::<B, 1>::from_floats(sample.as_slice(), device)
                .reshape([3, h as usize, w as usize])
        };
        let to_targets = |labels: &[Label]| -> (Vec<f32>, Vec<[f32; 4]>) {
            let mut cls = Vec::with_capacity(labels.len());
            let mut bbox = Vec::with_capacity(labels.len());
            for label in labels {
                cls.push(label.class);
                bbox.push([label.cx, label.cy, label.w, label.h]);
            }
            (cls, bbox)
        };

        if self.mosaic && crate::rng::gen_f32() < self.params.mosaic {
            let (mut img, mut labels) = self.load_mosaic(index)?;

            if crate::rng::gen_f32() < self.params.mix_up {
                let mix_index = crate::rng::gen_range_usize(0..self.len());
                let (mix_img, mix_labels) = self.load_mosaic(mix_index)?;
                (img, labels) = augment::mix_up(&img, &labels, &mix_img, &mix_labels);
            }

            let labels = xy2wh(&labels, img.width() as f32, img.height() as f32);
            let cls: Vec<f32> = labels.iter().map(|label| label.class).collect();
            let bbox: Vec<[f32; 4]> = labels
                .iter()
                .map(|label| [label.cx, label.cy, label.w, label.h])
                .collect();
            let (mut img, bbox, cls) = self.albumentations.call(img, bbox, cls);
            let mut labels: Vec<Label> = cls
                .into_iter()
                .zip(bbox)
                .map(|(class, bbox)| Label {
                    class,
                    cx: bbox[0],
                    cy: bbox[1],
                    w: bbox[2],
                    h: bbox[3],
                })
                .collect();

            augment::augment_hsv(&mut img, &self.params);
            if crate::rng::gen_f32() < self.params.flip_ud {
                image::imageops::flip_vertical_in_place(&mut img);
                for l in &mut labels {
                    l.cy = 1.0 - l.cy;
                }
            }
            if crate::rng::gen_f32() < self.params.flip_lr {
                image::imageops::flip_horizontal_in_place(&mut img);
                for l in &mut labels {
                    l.cx = 1.0 - l.cx;
                }
            }

            let tensor = to_sample(&img);
            let (cls, bbox) = to_targets(&labels);
            return Ok(Sample {
                image: tensor,
                cls,
                bbox,
            });
        }

        let (img, _) = self.load_image(index)?;
        let (loaded_w, loaded_h) = img.dimensions();
        let (img, ratio, pad) = resize::resize(&img, self.input_size, self.augment)?;

        let (loaded_w, loaded_h) = (loaded_w as f32, loaded_h as f32);
        let labels = wh2xy(
            &self.labels[index],
            ratio.0 * loaded_w,
            ratio.1 * loaded_h,
            pad.0,
            pad.1,
        );

        if self.augment {
            let (img, labels) = augment::random_perspective(&img, &labels, &self.params, &[0, 0]);
            let labels = xy2wh(&labels, img.width() as f32, img.height() as f32);
            let cls: Vec<f32> = labels.iter().map(|label| label.class).collect();
            let bbox: Vec<[f32; 4]> = labels
                .iter()
                .map(|label| [label.cx, label.cy, label.w, label.h])
                .collect();
            let (mut img, bbox, cls) = self.albumentations.call(img, bbox, cls);
            let mut labels: Vec<Label> = cls
                .into_iter()
                .zip(bbox)
                .map(|(class, bbox)| Label {
                    class,
                    cx: bbox[0],
                    cy: bbox[1],
                    w: bbox[2],
                    h: bbox[3],
                })
                .collect();

            augment::augment_hsv(&mut img, &self.params);
            if crate::rng::gen_f32() < self.params.flip_ud {
                image::imageops::flip_vertical_in_place(&mut img);
                for l in &mut labels {
                    l.cy = 1.0 - l.cy;
                }
            }
            if crate::rng::gen_f32() < self.params.flip_lr {
                image::imageops::flip_horizontal_in_place(&mut img);
                for l in &mut labels {
                    l.cx = 1.0 - l.cx;
                }
            }

            let tensor = to_sample(&img);
            let (cls, bbox) = to_targets(&labels);
            Ok(Sample {
                image: tensor,
                cls,
                bbox,
            })
        } else {
            let labels = xy2wh(&labels, img.width() as f32, img.height() as f32);
            let tensor = to_sample(&img);
            let (cls, bbox) = to_targets(&labels);
            Ok(Sample {
                image: tensor,
                cls,
                bbox,
            })
        }
    }

    pub fn load_image(&self, index: usize) -> Result<(image::DynamicImage, (u32, u32))> {
        let path = &self.filenames[index];
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let image = if ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg") {
            let data = fs::read(path)
                .with_context(|| format!("Failed to read image: {}", path.display()))?;
            let mut decoder = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&data));
            decoder.decode_headers().map_err(|e| {
                anyhow::anyhow!("JPEG header error for {}: {:?}", path.display(), e)
            })?;
            let info = decoder
                .info()
                .ok_or_else(|| anyhow::anyhow!("No JPEG info for {}", path.display()))?;
            let (w, h) = (info.width as u32, info.height as u32);
            let pixels = decoder.decode().map_err(|e| {
                anyhow::anyhow!("JPEG decode error for {}: {:?}", path.display(), e)
            })?;
            let mut rgb = image::RgbImage::from_raw(w, h, pixels)
                .ok_or_else(|| anyhow::anyhow!("Bad JPEG buffer size for {}", path.display()))?;
            for pixel in rgb.pixels_mut() {
                pixel.0.swap(0, 2);
            }
            image::DynamicImage::ImageRgb8(rgb)
        } else {
            let mut image = image::open(path)
                .with_context(|| format!("Failed to load image: {}", path.display()))?
                .to_rgb8();
            for pixel in image.pixels_mut() {
                pixel.0.swap(0, 2);
            }
            image::DynamicImage::ImageRgb8(image)
        };
        let (w, h) = image.dimensions();
        let r = self.input_size as f64 / h.max(w) as f64;
        let image = if (r - 1.0).abs() > f64::EPSILON {
            let new_w = (w as f64 * r) as u32;
            let new_h = (h as f64 * r) as u32;
            let filter = if self.augment {
                resize::resample()
            } else {
                image::imageops::FilterType::Triangle
            };
            image.resize_exact(new_w, new_h, filter)
        } else {
            image
        };
        Ok((image, (h, w)))
    }

    pub fn load_mosaic(&self, index: usize) -> Result<(image::RgbImage, Vec<Label>)> {
        let s = self.input_size as i32;
        let border = [-(s + 1) / 2, -(s + 1) / 2];
        let xc = ((-border[0]) as f32
            + crate::rng::gen_f32() * ((2 * s + border[1]) - (-border[0])) as f32)
            as u32;
        let yc = ((-border[0]) as f32
            + crate::rng::gen_f32() * ((2 * s + border[1]) - (-border[0])) as f32)
            as u32;

        let canvas_size = (self.input_size * 2) as u32;
        let mut canvas = image::RgbImage::new(canvas_size, canvas_size);
        let mut all_labels = Vec::new();
        let mut indices = vec![index];
        for _ in 0..3 {
            indices.push(crate::rng::gen_range_usize(0..self.len()));
        }
        crate::rng::shuffle(&mut indices);

        for (i, &idx) in indices.iter().enumerate() {
            let (img, _) = self.load_image(idx)?;
            let (nw, nh) = img.dimensions();
            let img = img.to_rgb8();

            let (x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b): (u32, u32, u32, u32, u32, u32, u32, u32) =
                match i {
                    0 => {
                        let x1a = xc.saturating_sub(nw);
                        let y1a = yc.saturating_sub(nh);
                        (x1a, y1a, xc, yc, nw - (xc - x1a), nh - (yc - y1a), nw, nh)
                    }
                    1 => {
                        let x2a = (xc + nw).min(canvas_size);
                        let y1a = yc.saturating_sub(nh);
                        (xc, y1a, x2a, yc, 0, nh - (yc - y1a), (x2a - xc).min(nw), nh)
                    }
                    2 => {
                        let x1a = xc.saturating_sub(nw);
                        let y2a = (yc + nh).min(canvas_size);
                        (x1a, yc, xc, y2a, nw - (xc - x1a), 0, nw, (y2a - yc).min(nh))
                    }
                    _ => {
                        let x2a = (xc + nw).min(canvas_size);
                        let y2a = (yc + nh).min(canvas_size);
                        (
                            xc,
                            yc,
                            x2a,
                            y2a,
                            0,
                            0,
                            (x2a - xc).min(nw),
                            (y2a - yc).min(nh),
                        )
                    }
                };

            let pad_w = x1a as f32 - x1b as f32;
            let pad_h = y1a as f32 - y1b as f32;

            let copy_h: u32 = (y2a - y1a).min(y2b - y1b);
            let copy_w: u32 = (x2a - x1a).min(x2b - x1b);
            for dy in 0..copy_h {
                for dx in 0..copy_w {
                    let src_x = x1b + dx;
                    let src_y = y1b + dy;
                    if src_x < nw && src_y < nh {
                        let px = img.get_pixel(src_x, src_y).0;
                        let dst_x = x1a + dx;
                        let dst_y = y1a + dy;
                        if dst_x < canvas_size && dst_y < canvas_size {
                            canvas.put_pixel(dst_x, dst_y, image::Rgb([px[0], px[1], px[2]]));
                        }
                    }
                }
            }

            for l in wh2xy(&self.labels[idx], nw as f32, nh as f32, pad_w, pad_h) {
                let lx1 = l.cx.max(0.0).min(canvas_size as f32);
                let ly1 = l.cy.max(0.0).min(canvas_size as f32);
                let lx2 = l.w.max(0.0).min(canvas_size as f32);
                let ly2 = l.h.max(0.0).min(canvas_size as f32);
                all_labels.push(Label {
                    class: l.class,
                    cx: lx1,
                    cy: ly1,
                    w: lx2,
                    h: ly2,
                });
            }
        }

        Ok(augment::random_perspective(
            &canvas,
            &all_labels,
            &self.params,
            &border,
        ))
    }

    pub fn load_label(filenames: &[PathBuf]) -> Result<Vec<(PathBuf, Vec<Label>)>> {
        const CACHE_MAGIC: &[u8] = b"YOLOV11_RS_CACHE_V1\n";
        let first = filenames.first().expect("list index out of range");
        let mut cache_path = first.parent().map_or_else(
            || std::ffi::OsString::from(""),
            |path| path.as_os_str().to_os_string(),
        );
        cache_path.push(".cache");
        let cache_path = PathBuf::from(cache_path);
        let mut foreign_cache = false;
        if let Ok(bytes) = fs::read(&cache_path) {
            if bytes.starts_with(CACHE_MAGIC) {
                if let Ok(records) = bincode::deserialize(&bytes[CACHE_MAGIC.len()..]) {
                    return Ok(records);
                }
            } else {
                foreign_cache = true;
            }
        }

        let mut all_labels = Vec::with_capacity(filenames.len());
        for path in filenames {
            if !path.exists() {
                all_labels.push((path.clone(), Vec::new()));
                continue;
            }
            let reader = match image::ImageReader::open(path) {
                Ok(reader) => reader,
                Err(_) => continue,
            };
            let reader = match reader.with_guessed_format() {
                Ok(reader) => reader,
                Err(_) => continue,
            };
            let format_ok = matches!(
                reader.format(),
                Some(
                    ImageFormat::Bmp
                        | ImageFormat::Jpeg
                        | ImageFormat::Png
                        | ImageFormat::Tiff
                        | ImageFormat::WebP
                )
            );
            if !format_ok {
                continue;
            }
            let image = match reader.decode() {
                Ok(image) => image,
                Err(_) => continue,
            };
            let shape = image.dimensions();
            if shape.0 <= 9 || shape.1 <= 9 {
                continue;
            }

            let marker = format!(
                "{}images{}",
                std::path::MAIN_SEPARATOR,
                std::path::MAIN_SEPARATOR
            );
            let replacement = format!(
                "{}labels{}",
                std::path::MAIN_SEPARATOR,
                std::path::MAIN_SEPARATOR
            );
            let mut s = path.to_string_lossy().into_owned();
            if let Some(pos) = s.rfind(&marker) {
                s.replace_range(pos..pos + marker.len(), &replacement);
            }
            let label_path = PathBuf::from(s).with_extension("txt");
            let labels = if label_path.exists() {
                let content = fs::read_to_string(&label_path)?;
                let mut rows: Vec<([u32; 5], Label)> = Vec::new();
                let mut invalid_label = false;
                let mut expected_len = None;
                for line in content.lines().filter(|line| !line.trim().is_empty()) {
                    let parts: Vec<f32> = line
                        .split_whitespace()
                        .map(str::parse)
                        .collect::<Result<_, _>>()
                        .with_context(|| {
                            format!("Failed to parse label file: {}", label_path.display())
                        })?;
                    if let Some(expected_len) = expected_len {
                        if parts.len() != expected_len {
                            anyhow::bail!("ragged label rows in {}", label_path.display());
                        }
                    } else {
                        expected_len = Some(parts.len());
                    }
                    if parts.len() != 5 {
                        invalid_label = true;
                        break;
                    }
                    if parts.iter().any(|value| !value.is_finite() || *value < 0.0)
                        || parts[1..].iter().any(|value| *value > 1.0)
                    {
                        invalid_label = true;
                        break;
                    }
                    let key = [
                        (if parts[0] == 0.0 { 0.0 } else { parts[0] }).to_bits(),
                        (if parts[1] == 0.0 { 0.0 } else { parts[1] }).to_bits(),
                        (if parts[2] == 0.0 { 0.0 } else { parts[2] }).to_bits(),
                        (if parts[3] == 0.0 { 0.0 } else { parts[3] }).to_bits(),
                        (if parts[4] == 0.0 { 0.0 } else { parts[4] }).to_bits(),
                    ];
                    rows.push((
                        key,
                        Label {
                            class: parts[0],
                            cx: parts[1],
                            cy: parts[2],
                            w: parts[3],
                            h: parts[4],
                        },
                    ));
                }
                if invalid_label {
                    continue;
                }
                let mut first_indices: HashMap<[u32; 5], usize> = HashMap::new();
                for (index, (key, _)) in rows.iter().enumerate() {
                    first_indices.entry(*key).or_insert(index);
                }
                let mut unique: Vec<([u32; 5], usize)> = first_indices.into_iter().collect();
                unique.sort_by_key(|(key, _)| *key);
                unique
                    .into_iter()
                    .map(|(_, index)| rows[index].1.clone())
                    .collect()
            } else {
                Vec::new()
            };
            all_labels.push((path.clone(), labels));
        }
        if !foreign_cache {
            let mut bytes = CACHE_MAGIC.to_vec();
            bytes.extend(bincode::serialize(&all_labels)?);
            fs::write(cache_path, bytes)?;
        }
        Ok(all_labels)
    }
}

/// Augmentation parameters (parsed from args.yaml).
#[derive(Debug, Clone, Deserialize)]
pub struct AugmentParams {
    pub hsv_h: f32,
    pub hsv_s: f32,
    pub hsv_v: f32,
    pub degrees: f32,
    pub translate: f32,
    pub scale: f32,
    pub shear: f32,
    pub flip_ud: f32,
    pub flip_lr: f32,
    pub mosaic: f32,
    pub mix_up: f32,
}

/// Simple batch collation.
pub struct Batch<B: Backend> {
    /// [B, 3, H, W]
    pub images: Tensor<B, 4>,
    /// Flat class labels [total_targets, 1].
    pub cls: Tensor<B, 2>,
    /// Flat bounding boxes [total_targets, 4].
    pub bbox: Tensor<B, 2>,
    /// Batch index per target [total_targets].
    pub idx: Tensor<B, 1>,
}

impl Dataset {
    pub fn collate_fn<B: Backend>(samples: &[Sample<B>], device: &B::Device) -> Batch<B> {
        let imgs: Vec<Tensor<B, 4>> = samples
            .iter()
            .map(|s| s.image.clone().unsqueeze_dim(0))
            .collect();
        let images = Tensor::cat(imgs, 0);

        let mut all_cls = Vec::new();
        let mut all_bbox = Vec::new();
        let mut all_idx = Vec::new();

        for (i, s) in samples.iter().enumerate() {
            for j in 0..s.cls.len() {
                all_cls.push(s.cls[j]);
                all_bbox.extend_from_slice(&s.bbox[j]);
                all_idx.push(i as f32);
            }
        }

        let n = all_cls.len();
        // Burn does not support 0-element tensors; use a dummy target when batch has no labels
        let (cls, bbox, idx) = if n == 0 {
            let cls = Tensor::<B, 2>::zeros([1, 1], device);
            let bbox = Tensor::<B, 2>::zeros([1, 4], device);
            let idx = Tensor::<B, 1>::from_floats([-1.0f32].as_slice(), device);
            (cls, bbox, idx)
        } else {
            let cls = Tensor::<B, 1>::from_floats(all_cls.as_slice(), device).reshape([n, 1]);
            let bbox = Tensor::<B, 1>::from_floats(all_bbox.as_slice(), device).reshape([n, 4]);
            let idx = Tensor::<B, 1>::from_floats(all_idx.as_slice(), device);
            (cls, bbox, idx)
        };

        Batch {
            images,
            cls,
            bbox,
            idx,
        }
    }
}

pub fn wh2xy(labels: &[Label], w: f32, h: f32, pad_w: f32, pad_h: f32) -> Vec<Label> {
    labels
        .iter()
        .map(|label| Label {
            class: label.class,
            cx: w * (label.cx - label.w / 2.0) + pad_w,
            cy: h * (label.cy - label.h / 2.0) + pad_h,
            w: w * (label.cx + label.w / 2.0) + pad_w,
            h: h * (label.cy + label.h / 2.0) + pad_h,
        })
        .collect()
}

pub fn xy2wh(labels: &[Label], w: f32, h: f32) -> Vec<Label> {
    labels
        .iter()
        .map(|label| {
            let x1 = label.cx.clamp(0.0, w - 1e-3);
            let y1 = label.cy.clamp(0.0, h - 1e-3);
            let x2 = label.w.clamp(0.0, w - 1e-3);
            let y2 = label.h.clamp(0.0, h - 1e-3);
            Label {
                class: label.class,
                cx: ((x1 + x2) / 2.0) / w,
                cy: ((y1 + y2) / 2.0) / h,
                w: (x2 - x1) / w,
                h: (y2 - y1) / h,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{wh2xy, xy2wh, Albumentations, AugmentParams, Dataset, Label};
    use burn::backend::ndarray::NdArray;
    use image::{GenericImageView, ImageFormat, Rgb, RgbImage};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn augment_params() -> AugmentParams {
        AugmentParams {
            hsv_h: 0.0,
            hsv_s: 0.0,
            hsv_v: 0.0,
            degrees: 0.0,
            translate: 0.0,
            scale: 0.0,
            shear: 0.0,
            flip_ud: 0.0,
            flip_lr: 0.0,
            mosaic: 0.0,
            mix_up: 0.0,
        }
    }

    fn temp_root() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "yolov11_rs_dataset_test_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn write_image(path: &Path, width: u32, height: u32) {
        let mut img = RgbImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([10, 20, 30]);
        }
        img.save(path).unwrap();
    }

    #[test]
    fn dataset_new_keeps_missing_images_and_deduplicates_labels_like_python() {
        let root = temp_root();
        let images = root.join("images/train");
        let labels = root.join("labels/train");
        fs::create_dir_all(&images).unwrap();
        fs::create_dir_all(&labels).unwrap();

        let valid = images.join("valid.png");
        let png_content_unknown_ext = images.join("png_content.data");
        let small = images.join("small.png");
        let invalid_label = images.join("invalid_label.png");
        let missing = images.join("missing.png");
        write_image(&valid, 12, 12);
        let mut png_content = RgbImage::new(12, 12);
        for pixel in png_content.pixels_mut() {
            *pixel = Rgb([3, 4, 5]);
        }
        png_content
            .save_with_format(&png_content_unknown_ext, ImageFormat::Png)
            .unwrap();
        write_image(&small, 9, 12);
        write_image(&invalid_label, 12, 12);
        fs::write(
            labels.join("valid.txt"),
            "2 0.25 0.25 0.10 0.10\n1 0.5 0.5 0.25 0.25\n1 0.5 0.5 0.25 0.25\n1 0.5 -0.0 0.25 0.25\n1 0.5 0.0 0.25 0.25\n",
        )
        .unwrap();
        fs::write(labels.join("png_content.txt"), "3 0.5 0.5 0.20 0.20\n").unwrap();
        fs::write(labels.join("small.txt"), "0 0.5 0.5 0.2 0.2\n").unwrap();
        fs::write(labels.join("invalid_label.txt"), "0 1.2 0.5 0.2 0.2\n").unwrap();

        let dataset = Dataset::new(
            vec![
                valid.clone(),
                png_content_unknown_ext.clone(),
                small,
                invalid_label,
                missing.clone(),
            ],
            16,
            false,
            &augment_params(),
        )
        .unwrap();

        assert_eq!(
            dataset.filenames,
            vec![valid, png_content_unknown_ext, missing]
        );
        assert_eq!(dataset.labels.len(), 3);
        assert_eq!(dataset.labels[0].len(), 3);
        assert_eq!(dataset.labels[0][0].class, 1.0);
        assert_eq!(
            [
                dataset.labels[0][0].cx,
                dataset.labels[0][0].cy,
                dataset.labels[0][0].w,
                dataset.labels[0][0].h
            ],
            [0.5, -0.0, 0.25, 0.25]
        );
        assert_eq!(
            [
                dataset.labels[0][1].cx,
                dataset.labels[0][1].cy,
                dataset.labels[0][1].w,
                dataset.labels[0][1].h
            ],
            [0.5, 0.5, 0.25, 0.25]
        );
        assert_eq!(dataset.labels[0][2].class, 2.0);
        assert_eq!(dataset.labels[1][0].class, 3.0);
        assert!(dataset.labels[2].is_empty());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn load_label_propagates_malformed_numeric_labels_like_python() {
        let root = temp_root();
        let images = root.join("images/train");
        let labels = root.join("labels/train");
        fs::create_dir_all(&images).unwrap();
        fs::create_dir_all(&labels).unwrap();

        let image = images.join("bad.png");
        write_image(&image, 12, 12);
        fs::write(labels.join("bad.txt"), "0 nope 0.5 0.2 0.2\n").unwrap();

        let err = match Dataset::new(vec![image], 16, false, &augment_params()) {
            Ok(_) => panic!("malformed numeric labels should fail dataset construction"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Failed to parse label file"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn load_label_propagates_ragged_label_rows_like_numpy_array() {
        let root = temp_root();
        let images = root.join("images/train");
        let labels = root.join("labels/train");
        fs::create_dir_all(&images).unwrap();
        fs::create_dir_all(&labels).unwrap();

        let image = images.join("ragged.png");
        write_image(&image, 12, 12);
        fs::write(labels.join("ragged.txt"), "0 0.5 0.5 0.2 0.2\n1 0.4 0.4\n").unwrap();

        let err = match Dataset::new(vec![image], 16, false, &augment_params()) {
            Ok(_) => panic!("ragged label rows should fail dataset construction"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("ragged label rows"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn load_label_replaces_last_images_segment_like_python_rsplit() {
        let root = temp_root();
        let images = root.join("images/nested/images/train");
        let labels = root.join("images/nested/labels/train");
        fs::create_dir_all(&images).unwrap();
        fs::create_dir_all(&labels).unwrap();

        let image = images.join("nested.png");
        write_image(&image, 12, 12);
        fs::write(labels.join("nested.txt"), "4.5 0.5 0.5 0.2 0.2\n").unwrap();

        let dataset = Dataset::new(vec![image], 16, false, &augment_params()).unwrap();

        assert_eq!(dataset.labels[0].len(), 1);
        assert_eq!(dataset.labels[0][0].class, 4.5);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[should_panic(expected = "list index out of range")]
    fn dataset_new_empty_filenames_matches_python_index_error() {
        let _ = Dataset::new(Vec::new(), 16, false, &augment_params());
    }

    #[test]
    fn load_label_reuses_guarded_rust_cache() {
        let root = temp_root();
        let images = root.join("images/train");
        let labels = root.join("labels/train");
        fs::create_dir_all(&images).unwrap();
        fs::create_dir_all(&labels).unwrap();

        let image = images.join("cached.png");
        write_image(&image, 12, 12);
        fs::write(labels.join("cached.txt"), "3 0.5 0.5 0.25 0.25\n").unwrap();

        let first = Dataset::new(vec![image.clone()], 16, false, &augment_params()).unwrap();
        assert_eq!(first.labels[0][0].class, 3.0);
        assert!(images.with_extension("cache").exists());

        fs::write(labels.join("cached.txt"), "4 0.25 0.25 0.10 0.10\n").unwrap();
        let second = Dataset::new(vec![image], 16, false, &augment_params()).unwrap();

        assert_eq!(second.labels[0][0].class, 3.0);
        assert_eq!(
            [
                second.labels[0][0].cx,
                second.labels[0][0].cy,
                second.labels[0][0].w,
                second.labels[0][0].h
            ],
            [0.5, 0.5, 0.25, 0.25]
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn load_label_appends_cache_suffix_like_python_dirname_string() {
        let root = temp_root();
        let images = root.join("images/train.v1");
        let labels = root.join("labels/train.v1");
        fs::create_dir_all(&images).unwrap();
        fs::create_dir_all(&labels).unwrap();

        let image = images.join("cached.png");
        write_image(&image, 12, 12);
        fs::write(labels.join("cached.txt"), "5 0.5 0.5 0.25 0.25\n").unwrap();

        let dataset = Dataset::new(vec![image], 16, false, &augment_params()).unwrap();

        assert_eq!(dataset.labels[0][0].class, 5.0);
        assert!(root.join("images/train.v1.cache").exists());
        assert!(!root.join("images/train.cache").exists());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn load_label_preserves_foreign_cache_file() {
        let root = temp_root();
        let images = root.join("images/train");
        let labels = root.join("labels/train");
        fs::create_dir_all(&images).unwrap();
        fs::create_dir_all(&labels).unwrap();

        let image = images.join("foreign.png");
        let cache = images.with_extension("cache");
        let foreign_bytes = b"PYTORCH CACHE PLACEHOLDER".to_vec();
        write_image(&image, 12, 12);
        fs::write(labels.join("foreign.txt"), "2 0.5 0.5 0.25 0.25\n").unwrap();
        fs::write(&cache, &foreign_bytes).unwrap();

        let dataset = Dataset::new(vec![image], 16, false, &augment_params()).unwrap();

        assert_eq!(dataset.labels[0][0].class, 2.0);
        assert_eq!(fs::read(cache).unwrap(), foreign_bytes);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn get_item_decodes_bgr_internally_and_returns_rgb_tensor_like_python() {
        let root = temp_root();
        let images = root.join("images/train");
        fs::create_dir_all(&images).unwrap();

        let image = images.join("rgb.png");
        let mut img = RgbImage::new(12, 12);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([10, 20, 30]);
        }
        img.save(&image).unwrap();

        let dataset = Dataset::new(vec![image], 12, false, &augment_params()).unwrap();
        let device = Default::default();
        let sample = dataset.get_item::<NdArray>(0, &device).unwrap();
        let data: Vec<f32> = sample.image.to_data().to_vec().unwrap();
        let hw = 12 * 12;

        assert_eq!(data[0], 10.0);
        assert_eq!(data[hw], 20.0);
        assert_eq!(data[2 * hw], 30.0);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn load_image_scales_small_validation_images_like_python() {
        let root = temp_root();
        let images = root.join("images/train");
        fs::create_dir_all(&images).unwrap();

        let image = images.join("small_valid.png");
        let mut img = RgbImage::new(12, 12);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([40, 50, 60]);
        }
        img.save(&image).unwrap();

        let dataset = Dataset::new(vec![image], 24, false, &augment_params()).unwrap();
        let device = Default::default();
        let sample = dataset.get_item::<NdArray>(0, &device).unwrap();
        let dims = sample.image.dims();
        let data: Vec<f32> = sample.image.to_data().to_vec().unwrap();
        let hw = 24 * 24;

        assert_eq!(dims, [3, 24, 24]);
        assert_eq!(data[0], 40.0);
        assert_eq!(data[hw], 50.0);
        assert_eq!(data[2 * hw], 60.0);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn load_image_uses_python_float_truncation_for_resize_dimensions() {
        let root = temp_root();
        let images = root.join("images/train");
        fs::create_dir_all(&images).unwrap();

        let image = images.join("precision.png");
        write_image(&image, 7, 7);

        let dataset = Dataset {
            filenames: vec![image],
            labels: vec![Vec::new()],
            input_size: 31,
            augment: false,
            mosaic: false,
            params: augment_params(),
            albumentations: Albumentations::new(),
        };

        let (loaded, _) = dataset.load_image(0).unwrap();
        assert_eq!(loaded.dimensions(), (31, 31));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn get_item_clips_final_boxes_like_python_xy2wh() {
        let root = temp_root();
        let images = root.join("images/train");
        let labels = root.join("labels/train");
        fs::create_dir_all(&images).unwrap();
        fs::create_dir_all(&labels).unwrap();

        let image = images.join("full_box.png");
        write_image(&image, 12, 12);
        fs::write(labels.join("full_box.txt"), "0 0.5 0.5 1.0 1.0\n").unwrap();

        let dataset = Dataset::new(vec![image], 12, false, &augment_params()).unwrap();
        let device = Default::default();
        let sample = dataset.get_item::<NdArray>(0, &device).unwrap();
        let expected_edge = (12.0_f32 - 1e-3) / 12.0;

        assert_eq!(sample.cls, vec![0.0]);
        assert!((sample.bbox[0][0] - expected_edge / 2.0).abs() < 1e-6);
        assert!((sample.bbox[0][1] - expected_edge / 2.0).abs() < 1e-6);
        assert!((sample.bbox[0][2] - expected_edge).abs() < 1e-6);
        assert!((sample.bbox[0][3] - expected_edge).abs() < 1e-6);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn dataset_wh2xy_and_xy2wh_match_python_edge_fixture() {
        let labels = vec![
            Label {
                class: 2.0,
                cx: 0.5,
                cy: 0.25,
                w: 0.5,
                h: 0.25,
            },
            Label {
                class: 3.0,
                cx: 1.1,
                cy: -0.1,
                w: 1.4,
                h: 8.7,
            },
        ];
        let xy = wh2xy(&labels, 100.0, 80.0, 3.0, 5.0);

        assert_eq!(xy[0].class, 2.0);
        assert_eq!(
            [xy[0].cx, xy[0].cy, xy[0].w, xy[0].h],
            [28.0, 15.0, 78.0, 35.0]
        );
        assert_eq!(xy[1].class, 3.0);
        assert!((xy[1].cx - 43.000004).abs() < 1e-5);
        assert!((xy[1].cy + 351.0).abs() < 1e-6);
        assert!((xy[1].w - 183.0).abs() < 1e-6);
        assert!((xy[1].h - 345.0).abs() < 1e-6);

        let labels = vec![
            Label {
                class: 0.0,
                cx: -10.0,
                cy: 5.0,
                w: 120.0,
                h: 90.0,
            },
            Label {
                class: 1.0,
                cx: 20.0,
                cy: -5.0,
                w: 30.0,
                h: 40.0,
            },
        ];
        let wh = xy2wh(&labels, 100.0, 80.0);

        assert_eq!(wh[0].class, 0.0);
        assert!((wh[0].cx - 0.499995).abs() < 1e-6);
        assert!((wh[0].cy - 0.53124374).abs() < 1e-6);
        assert!((wh[0].w - 0.99999).abs() < 1e-6);
        assert!((wh[0].h - 0.9374875).abs() < 1e-6);
        assert_eq!(wh[1].class, 1.0);
        assert_eq!(
            [wh[1].cx, wh[1].cy, wh[1].w, wh[1].h],
            [0.25, 0.25, 0.1, 0.5]
        );
    }
}
