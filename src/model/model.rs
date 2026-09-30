use burn::module::{Module, ParamGroup};
use burn::prelude::*;

use crate::model::backbone::DarkNet;
use crate::model::conv::{with_frozen_batch_norm, FuseModule};
use crate::model::head::{Head, HeadOutput};
use crate::model::neck::DarkFPN;

#[derive(Module, Debug)]
pub struct YOLO {
    net: DarkNet,
    fpn: DarkFPN,
    head: Head,
    pub stride: Tensor<1>,
}

#[derive(Debug)]
pub enum YOLOOutput {
    Train(Vec<Tensor<4>>),
    Infer(Tensor<3>),
}

impl YOLO {
    pub fn new(
        width: &[usize],
        depth: &[usize],
        csp: &[bool],
        num_classes: usize,
        device: &Device,
    ) -> Self {
        let net = DarkNet::new(width, depth, csp, device);
        let fpn = DarkFPN::new(width, depth, csp, device);

        // Compute strides via dummy forward pass
        let dummy = Tensor::<4>::zeros([1, width[0], 256, 256], device);
        let (p3, p4, p5) = net.forward(dummy);
        let (p3, p4, p5) = fpn.forward(p3, p4, p5);

        let s3 = 256.0 / p3.dims()[3] as f64;
        let s4 = 256.0 / p4.dims()[3] as f64;
        let s5 = 256.0 / p5.dims()[3] as f64;
        let stride = Tensor::<1>::from_floats([s3 as f32, s4 as f32, s5 as f32].as_slice(), device);

        let filters = [width[3], width[4], width[5]];
        let mut head = Head::new(num_classes, &filters, stride.clone(), device);
        head.initialize_biases();

        Self {
            net,
            fpn,
            head,
            stride,
        }
    }

    pub fn forward(&self, x: Tensor<4>, training: bool) -> YOLOOutput {
        self.forward_with_batch_norm(x, training, false)
    }

    /// Freeze exactly the DarkNet feature extractor parameters.
    pub fn freeze_backbone(self) -> Self {
        let group = ParamGroup::ids_from_module(self.net.clone());
        self.freeze_group(group)
    }

    /// Freeze exactly the DarkNet and feature-pyramid parameters.
    pub fn freeze_backbone_and_neck(self) -> Self {
        let group = ParamGroup::ids_from_module(self.net.clone())
            .fuse(&ParamGroup::ids_from_module(self.fpn.clone()));
        self.freeze_group(group)
    }

    /// Run a forward pass while optionally keeping all BatchNorm population
    /// statistics fixed. Gradients still flow through BatchNorm affine
    /// parameters and surrounding layers.
    pub fn forward_with_batch_norm(
        &self,
        x: Tensor<4>,
        training: bool,
        frozen_batch_norm: bool,
    ) -> YOLOOutput {
        with_frozen_batch_norm(frozen_batch_norm, || self.forward_inner(x, training))
    }

    /// Return the three feature-pyramid tensors before the 2D detection head.
    ///
    /// This is the stable transfer-learning boundary for consumers which reuse
    /// the microscopy encoder for a different prediction task, such as fusing
    /// features from adjacent Z slices before a 3D center-detection head.
    pub fn forward_features_with_batch_norm(
        &self,
        x: Tensor<4>,
        frozen_batch_norm: bool,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        with_frozen_batch_norm(frozen_batch_norm, || {
            let (p3, p4, p5) = self.net.forward(x);
            self.fpn.forward(p3, p4, p5)
        })
    }

    fn forward_inner(&self, x: Tensor<4>, training: bool) -> YOLOOutput {
        let profile = std::env::var_os("YOLOV11_PROFILE").is_some();
        let device = x.device();
        let started = std::time::Instant::now();
        let (p3, p4, p5) = self.net.forward(x);
        if profile {
            device.sync().expect("synchronizing YOLO backbone");
        }
        let backbone_elapsed = started.elapsed();
        let fpn_started = std::time::Instant::now();
        let (p3, p4, p5) = self.fpn.forward(p3, p4, p5);
        if profile {
            device.sync().expect("synchronizing YOLO FPN");
        }
        let fpn_elapsed = fpn_started.elapsed();
        let head_started = std::time::Instant::now();
        let output = match self.head.forward(&[p3, p4, p5], training) {
            HeadOutput::Train(outputs) => YOLOOutput::Train(outputs),
            HeadOutput::Infer(output) => YOLOOutput::Infer(output),
        };
        if profile {
            device.sync().expect("synchronizing YOLO head");
            eprintln!(
                "YOLOV11_PROFILE backbone_us={} fpn_us={} head_us={}",
                backbone_elapsed.as_micros(),
                fpn_elapsed.as_micros(),
                head_started.elapsed().as_micros(),
            );
        }
        output
    }

    pub fn fuse(self) -> Self {
        Self {
            net: self.net.fuse_module(),
            fpn: self.fpn.fuse_module(),
            head: self.head.fuse_module(),
            stride: self.stride,
        }
    }
}

// ---------------------------------------------------------------------------
// Model variant constructors
// ---------------------------------------------------------------------------

pub fn yolo_v11_n(num_classes: usize, device: &Device) -> YOLO {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 16, 32, 64, 128, 256];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_t(num_classes: usize, device: &Device) -> YOLO {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 24, 48, 96, 192, 384];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_s(num_classes: usize, device: &Device) -> YOLO {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 32, 64, 128, 256, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_m(num_classes: usize, device: &Device) -> YOLO {
    let csp = [true, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 64, 128, 256, 512, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_l(num_classes: usize, device: &Device) -> YOLO {
    let csp = [true, true];
    let depth = [2, 2, 2, 2, 2, 2];
    let width = [3, 64, 128, 256, 512, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_x(num_classes: usize, device: &Device) -> YOLO {
    let csp = [true, true];
    let depth = [2, 2, 2, 2, 2, 2];
    let width = [3, 96, 192, 384, 768, 768];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

#[cfg(test)]
mod tests {
    use super::{yolo_v11_n, YOLOOutput};
    use burn::module::{ModuleVisitor, Param};
    use burn::prelude::*;

    #[derive(Default)]
    struct FreezeVisitor {
        frozen: usize,
        trainable: usize,
    }

    impl ModuleVisitor for FreezeVisitor {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if param.val().is_require_grad() {
                self.trainable += 1;
            } else {
                self.frozen += 1;
            }
        }
    }

    #[test]
    fn yolo_v11_n_forward_training_smoke_matches_three_scale_head() {
        let device = burn::tensor::Device::flex().autodiff();
        let model = yolo_v11_n(2, &device);
        let x = Tensor::<4>::zeros([1, 3, 64, 64], &device);

        let (p3, p4, p5) = model.forward_features_with_batch_norm(x.clone(), true);
        assert_eq!(p3.dims(), [1, 64, 8, 8]);
        assert_eq!(p4.dims(), [1, 128, 4, 4]);
        assert_eq!(p5.dims(), [1, 256, 2, 2]);

        match model.forward(x, true) {
            YOLOOutput::Train(outputs) => {
                assert_eq!(outputs.len(), 3);
                assert_eq!(outputs[0].dims()[0], 1);
                assert_eq!(outputs[1].dims()[0], 1);
                assert_eq!(outputs[2].dims()[0], 1);
                assert_eq!(outputs[0].dims()[1], 66);
                assert_eq!(outputs[1].dims()[1], 66);
                assert_eq!(outputs[2].dims()[1], 66);
            }
            YOLOOutput::Infer(_) => panic!("training forward returned inference output"),
        }
    }

    #[test]
    fn yolo_fuse_preserves_training_forward_shape() {
        let device = burn::tensor::Device::flex();
        let model = yolo_v11_n(2, &device).fuse();
        let x = Tensor::<4>::zeros([1, 3, 64, 64], &device);

        match model.forward(x, true) {
            YOLOOutput::Train(outputs) => {
                assert_eq!(outputs.len(), 3);
                assert_eq!(outputs[0].dims()[1], 66);
                assert_eq!(outputs[1].dims()[1], 66);
                assert_eq!(outputs[2].dims()[1], 66);
            }
            YOLOOutput::Infer(_) => panic!("training forward returned inference output"),
        }
    }

    #[test]
    fn backbone_group_freezes_only_darknet_parameters() {
        let device = burn::tensor::Device::flex().autodiff();
        let model = yolo_v11_n(1, &device).freeze_backbone();
        let mut visitor = FreezeVisitor::default();
        model.visit(&mut visitor);

        assert!(visitor.frozen > 0);
        assert!(visitor.trainable > 0);
    }

    #[test]
    fn backbone_and_neck_group_leaves_detection_head_trainable() {
        let device = burn::tensor::Device::flex().autodiff();
        let model = yolo_v11_n(1, &device);
        let mut head_before = FreezeVisitor::default();
        model.head.visit(&mut head_before);
        let model = model.freeze_backbone_and_neck();
        let mut backbone = FreezeVisitor::default();
        model.net.visit(&mut backbone);
        let mut neck = FreezeVisitor::default();
        model.fpn.visit(&mut neck);
        let mut head = FreezeVisitor::default();
        model.head.visit(&mut head);

        assert!(backbone.frozen > 0);
        assert_eq!(backbone.trainable, 0);
        assert!(neck.frozen > 0);
        assert_eq!(neck.trainable, 0);
        // RunningState tensors are inherently non-trainable, so compare the
        // head with itself before freezing rather than expecting zero frozen
        // tensors in a module containing BatchNorm.
        assert_eq!(head.frozen, head_before.frozen);
        assert_eq!(head.trainable, head_before.trainable);
        assert!(head.trainable > 0);
    }
}
