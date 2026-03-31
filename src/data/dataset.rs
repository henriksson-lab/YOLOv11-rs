use anyhow::{Context, Result};
use image::GenericImageView;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::data::augment;
use crate::data::resize;

/// A single label entry: (class, cx, cy, w, h) all in [0,1] normalized coords.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    pub class: u32,
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
    // Augmentation params
    pub hsv_h: f32,
    pub hsv_s: f32,
    pub hsv_v: f32,
    pub degrees: f32,
    pub translate: f32,
    pub scale: f32,
    pub shear: f32,
    pub flip_ud: f32,
    pub flip_lr: f32,
    pub mosaic_prob: f32,
    pub mixup_prob: f32,
}

/// A single sample returned by the dataset.
pub struct Sample {
    /// Image tensor [3, H, W] in 0..1 range.
    pub image: candle_core::Tensor,
    /// Class labels [N, 1].
    pub cls: Vec<f32>,
    /// Bounding boxes [N, 4] in normalized (cx, cy, w, h).
    pub bbox: Vec<[f32; 4]>,
}

impl Dataset {
    /// Create a new dataset.
    /// `filenames_txt` is a file listing image paths, one per line.
    /// `data_dir` is the root COCO-format directory.
    pub fn new(
        filenames: Vec<PathBuf>,
        input_size: u32,
        augment: bool,
        params: &AugmentParams,
    ) -> Result<Self> {
        let labels = Self::load_labels(&filenames)?;
        Ok(Self {
            filenames,
            labels,
            input_size,
            augment,
            mosaic: augment,
            hsv_h: params.hsv_h,
            hsv_s: params.hsv_s,
            hsv_v: params.hsv_v,
            degrees: params.degrees,
            translate: params.translate,
            scale: params.scale,
            shear: params.shear,
            flip_ud: params.flip_ud,
            flip_lr: params.flip_lr,
            mosaic_prob: params.mosaic,
            mixup_prob: params.mix_up,
        })
    }

    pub fn len(&self) -> usize {
        self.filenames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.filenames.is_empty()
    }

    /// Get a single sample (with augmentation if enabled).
    pub fn get(&self, index: usize, device: &candle_core::Device) -> Result<Sample> {
        let mut rng = rand::thread_rng();

        if self.mosaic && rng.gen::<f32>() < self.mosaic_prob {
            // Mosaic augmentation
            let mut indices = vec![index];
            for _ in 0..3 {
                indices.push(rng.gen_range(0..self.len()));
            }
            let (img, labels) = self.load_mosaic(&indices)?;

            // Apply perspective augmentation
            let (img, labels) = augment::random_perspective(
                &img,
                &labels,
                self.input_size,
                self.degrees,
                self.translate,
                self.scale,
                self.shear,
                &[-(self.input_size as i32) / 2, -(self.input_size as i32) / 2],
            );

            let (img, labels) = self.apply_pixel_augments(img, labels, &mut rng);
            let tensor = resize::image_to_tensor(&img, device)?;
            let (cls, bbox) = split_labels(&labels);
            return Ok(Sample { image: tensor, cls, bbox });
        }

        // Regular loading
        let img = self.load_image(index)?;
        let (img, ratio, pad) = resize::letterbox(&img, self.input_size, self.augment)?;

        let mut labels = self.labels[index].clone();
        // Transform labels to pixel coords, then back to normalized
        let (orig_w, orig_h) = (
            img.width() as f32 / ratio.0,
            img.height() as f32 / ratio.1,
        );
        // Actually, apply ratio and pad to labels
        for l in &mut labels {
            // Convert from normalized xywh to pixel xyxy
            let x1 = (l.cx - l.w / 2.0) * orig_w * ratio.0 + pad.0;
            let y1 = (l.cy - l.h / 2.0) * orig_h * ratio.1 + pad.1;
            let x2 = (l.cx + l.w / 2.0) * orig_w * ratio.0 + pad.0;
            let y2 = (l.cy + l.h / 2.0) * orig_h * ratio.1 + pad.1;
            // Back to normalized xywh
            l.cx = ((x1 + x2) / 2.0) / self.input_size as f32;
            l.cy = ((y1 + y2) / 2.0) / self.input_size as f32;
            l.w = (x2 - x1) / self.input_size as f32;
            l.h = (y2 - y1) / self.input_size as f32;
        }

        if self.augment {
            let (img, labels) = self.apply_pixel_augments(img, labels, &mut rng);
            let tensor = resize::image_to_tensor(&img, device)?;
            let (cls, bbox) = split_labels(&labels);
            Ok(Sample { image: tensor, cls, bbox })
        } else {
            let tensor = resize::image_to_tensor(&img, device)?;
            let (cls, bbox) = split_labels(&labels);
            Ok(Sample { image: tensor, cls, bbox })
        }
    }

    fn load_image(&self, index: usize) -> Result<image::DynamicImage> {
        let path = &self.filenames[index];
        image::open(path).with_context(|| format!("Failed to load image: {}", path.display()))
    }

    fn load_mosaic(&self, indices: &[usize]) -> Result<(image::RgbImage, Vec<Label>)> {
        let s = self.input_size as i32;
        let mut rng = rand::thread_rng();
        let xc = rng.gen_range(s / 2..3 * s / 2) as u32;
        let yc = rng.gen_range(s / 2..3 * s / 2) as u32;

        let canvas_size = (self.input_size * 2) as u32;
        let mut canvas = image::RgbImage::new(canvas_size, canvas_size);
        let mut all_labels = Vec::new();

        for (i, &idx) in indices.iter().enumerate() {
            let img = self.load_image(idx)?;
            let (iw, ih) = img.dimensions();
            let r = (self.input_size as f32 / ih.max(iw) as f32).min(1.0);
            let nw = (iw as f32 * r) as u32;
            let nh = (ih as f32 * r) as u32;
            let img = img.resize_exact(nw, nh, image::imageops::FilterType::Triangle).to_rgb8();

            let (x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b): (u32, u32, u32, u32, u32, u32, u32, u32) = match i {
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
                    (xc, yc, x2a, y2a, 0, 0, (x2a - xc).min(nw), (y2a - yc).min(nh))
                }
            };

            let pad_w = x1a as f32 - x1b as f32;
            let pad_h = y1a as f32 - y1b as f32;

            // Copy pixels
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

            // Transform labels
            for l in &self.labels[idx] {
                let lx1 = (l.cx - l.w / 2.0) * nw as f32 + pad_w;
                let ly1 = (l.cy - l.h / 2.0) * nh as f32 + pad_h;
                let lx2 = (l.cx + l.w / 2.0) * nw as f32 + pad_w;
                let ly2 = (l.cy + l.h / 2.0) * nh as f32 + pad_h;
                // Clip to canvas
                let lx1 = lx1.max(0.0).min(canvas_size as f32);
                let ly1 = ly1.max(0.0).min(canvas_size as f32);
                let lx2 = lx2.max(0.0).min(canvas_size as f32);
                let ly2 = ly2.max(0.0).min(canvas_size as f32);
                if lx2 > lx1 + 2.0 && ly2 > ly1 + 2.0 {
                    all_labels.push(Label {
                        class: l.class,
                        cx: (lx1 + lx2) / 2.0 / canvas_size as f32,
                        cy: (ly1 + ly2) / 2.0 / canvas_size as f32,
                        w: (lx2 - lx1) / canvas_size as f32,
                        h: (ly2 - ly1) / canvas_size as f32,
                    });
                }
            }
        }

        Ok((canvas, all_labels))
    }

    fn apply_pixel_augments(
        &self,
        mut img: image::RgbImage,
        mut labels: Vec<Label>,
        rng: &mut impl Rng,
    ) -> (image::RgbImage, Vec<Label>) {
        // HSV augmentation
        augment::augment_hsv(&mut img, self.hsv_h, self.hsv_s, self.hsv_v);

        // Flip up-down
        if rng.gen::<f32>() < self.flip_ud {
            image::imageops::flip_vertical_in_place(&mut img);
            for l in &mut labels {
                l.cy = 1.0 - l.cy;
            }
        }

        // Flip left-right
        if rng.gen::<f32>() < self.flip_lr {
            image::imageops::flip_horizontal_in_place(&mut img);
            for l in &mut labels {
                l.cx = 1.0 - l.cx;
            }
        }

        (img, labels)
    }

    /// Load labels from corresponding .txt files.
    fn load_labels(filenames: &[PathBuf]) -> Result<Vec<Vec<Label>>> {
        let mut all_labels = Vec::with_capacity(filenames.len());
        for path in filenames {
            let label_path = image_to_label_path(path);
            let labels = if label_path.exists() {
                let content = fs::read_to_string(&label_path)?;
                content
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .filter_map(|line| {
                        let parts: Vec<f32> = line
                            .split_whitespace()
                            .filter_map(|s| s.parse().ok())
                            .collect();
                        if parts.len() == 5 {
                            Some(Label {
                                class: parts[0] as u32,
                                cx: parts[1],
                                cy: parts[2],
                                w: parts[3],
                                h: parts[4],
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };
            all_labels.push(labels);
        }
        Ok(all_labels)
    }

    /// Load filenames from a text file listing image paths.
    pub fn load_filenames(txt_path: &Path, data_dir: &Path, split: &str) -> Result<Vec<PathBuf>> {
        let content = fs::read_to_string(txt_path)?;
        Ok(content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| {
                let basename = Path::new(line.trim()).file_name().unwrap().to_owned();
                data_dir.join("images").join(split).join(basename)
            })
            .collect())
    }
}

/// Convert image path to label path: .../images/split/name.jpg -> .../labels/split/name.txt
fn image_to_label_path(image_path: &Path) -> PathBuf {
    let s = image_path.to_string_lossy();
    let s = s.replace("/images/", "/labels/");
    let p = PathBuf::from(s);
    p.with_extension("txt")
}

fn split_labels(labels: &[Label]) -> (Vec<f32>, Vec<[f32; 4]>) {
    let cls: Vec<f32> = labels.iter().map(|l| l.class as f32).collect();
    let bbox: Vec<[f32; 4]> = labels
        .iter()
        .map(|l| [l.cx, l.cy, l.w, l.h])
        .collect();
    (cls, bbox)
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
pub struct Batch {
    /// [B, 3, H, W]
    pub images: candle_core::Tensor,
    /// Flat class labels [total_targets].
    pub cls: candle_core::Tensor,
    /// Flat bounding boxes [total_targets, 4].
    pub bbox: candle_core::Tensor,
    /// Batch index per target [total_targets].
    pub idx: candle_core::Tensor,
}

pub fn collate(
    samples: &[Sample],
    device: &candle_core::Device,
) -> candle_core::Result<Batch> {
    let imgs: Vec<&candle_core::Tensor> = samples.iter().map(|s| &s.image).collect();
    let images = candle_core::Tensor::stack(&imgs, 0)?;

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
    let cls = candle_core::Tensor::from_vec(all_cls, (n, 1), device)?;
    let bbox = candle_core::Tensor::from_vec(all_bbox, (n, 4), device)?;
    let idx = candle_core::Tensor::from_vec(all_idx, n, device)?;

    Ok(Batch { images, cls, bbox, idx })
}
