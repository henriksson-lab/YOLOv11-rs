use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Full configuration parsed from args.yaml.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Config {
    pub min_lr: f64,
    pub max_lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub warmup_epochs: f64,
    pub box_: Option<f64>,
    pub cls: Option<f64>,
    pub dfl: Option<f64>,
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
    pub names: HashMap<u32, String>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        // The yaml has "box" as key which is a Rust keyword, handle via raw parsing
        let raw: serde_yaml::Value = serde_yaml::from_str(&content)?;
        let map = raw.as_mapping().unwrap();

        let get_f64 = |key: &str| -> f64 {
            map.get(serde_yaml::Value::String(key.to_string()))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        };
        let get_f32 = |key: &str| -> f32 { get_f64(key) as f32 };

        let names_val = map
            .get(serde_yaml::Value::String("names".to_string()))
            .unwrap();
        let mut names = HashMap::new();
        if let Some(nm) = names_val.as_mapping() {
            for (k, v) in nm {
                let idx = k.as_u64().unwrap_or(0) as u32;
                let name = v.as_str().unwrap_or("").to_string();
                names.insert(idx, name);
            }
        }

        Ok(Config {
            min_lr: get_f64("min_lr"),
            max_lr: get_f64("max_lr"),
            momentum: get_f64("momentum"),
            weight_decay: get_f64("weight_decay"),
            warmup_epochs: get_f64("warmup_epochs"),
            box_: Some(get_f64("box")),
            cls: Some(get_f64("cls")),
            dfl: Some(get_f64("dfl")),
            hsv_h: get_f32("hsv_h"),
            hsv_s: get_f32("hsv_s"),
            hsv_v: get_f32("hsv_v"),
            degrees: get_f32("degrees"),
            translate: get_f32("translate"),
            scale: get_f32("scale"),
            shear: get_f32("shear"),
            flip_ud: get_f32("flip_ud"),
            flip_lr: get_f32("flip_lr"),
            mosaic: get_f32("mosaic"),
            mix_up: get_f32("mix_up"),
            names,
        })
    }

    pub fn num_classes(&self) -> usize {
        self.names.len()
    }

    pub fn box_gain(&self) -> f64 {
        self.box_.unwrap_or(7.5)
    }
    pub fn cls_gain(&self) -> f64 {
        self.cls.unwrap_or(0.5)
    }
    pub fn dfl_gain(&self) -> f64 {
        self.dfl.unwrap_or(1.5)
    }

    pub fn to_augment_params(&self) -> crate::data::dataset::AugmentParams {
        crate::data::dataset::AugmentParams {
            hsv_h: self.hsv_h,
            hsv_s: self.hsv_s,
            hsv_v: self.hsv_v,
            degrees: self.degrees,
            translate: self.translate,
            scale: self.scale,
            shear: self.shear,
            flip_ud: self.flip_ud,
            flip_lr: self.flip_lr,
            mosaic: self.mosaic,
            mix_up: self.mix_up,
        }
    }
}
