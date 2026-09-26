/// Non-maximum suppression on CPU.
use burn::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct NmsOptions {
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub max_detections: usize,
    pub max_candidates: usize,
}

impl Default for NmsOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            max_detections: 300,
            max_candidates: 30_000,
        }
    }
}

/// Apply NMS to model outputs in Python's `[B, 4 + nc, A]` layout.
pub fn non_max_suppression(
    outputs: &Tensor<3>,
    confidence_threshold: f32,
    iou_threshold: f32,
) -> Vec<Vec<[f32; 6]>> {
    non_max_suppression_with_options(
        outputs,
        NmsOptions {
            confidence_threshold,
            iou_threshold,
            ..NmsOptions::default()
        },
    )
}

/// Apply NMS with explicit candidate and output limits.
///
/// Every image in the batch is always processed. A wall-clock deadline would
/// make metrics depend on batch ordering and host load, and can silently turn
/// unprocessed images into false negatives.
pub fn non_max_suppression_with_options(
    outputs: &Tensor<3>,
    options: NmsOptions,
) -> Vec<Vec<[f32; 6]>> {
    const MAX_WH: f32 = 7680.0;

    let [batch, stride, num_anchors] = outputs.dims();
    let mut output = vec![Vec::new(); batch];

    for index in 0..batch {
        let x = outputs
            .clone()
            .narrow(0, index, 1)
            .squeeze_dim::<2>(0)
            .swap_dims(0, 1);
        let preds: Vec<f32> = x.to_data().try_to_vec::<f32>().unwrap();

        let mut candidates: Vec<[f32; 6]> = Vec::new();
        for i in 0..num_anchors {
            let base = i * stride;
            let cls_scores = &preds[base + 4..base + stride];
            if !cls_scores
                .iter()
                .any(|&score| score > options.confidence_threshold)
            {
                continue;
            }

            let cx = preds[base];
            let cy = preds[base + 1];
            let w = preds[base + 2];
            let h = preds[base + 3];
            let x1 = cx - w * 0.5;
            let y1 = cy - h * 0.5;
            let x2 = cx + w * 0.5;
            let y2 = cy + h * 0.5;

            if cls_scores.len() > 1 {
                for (class, &confidence) in cls_scores.iter().enumerate() {
                    if confidence > options.confidence_threshold {
                        candidates.push([x1, y1, x2, y2, confidence, class as f32]);
                    }
                }
            } else {
                let confidence = cls_scores[0];
                if confidence > options.confidence_threshold {
                    candidates.push([x1, y1, x2, y2, confidence, 0.0]);
                }
            }
        }

        if candidates.is_empty() {
            continue;
        }

        candidates.sort_unstable_by(|a, b| b[4].partial_cmp(&a[4]).unwrap());
        candidates.truncate(options.max_candidates);

        let mut suppressed = vec![false; candidates.len()];
        let mut keep: Vec<[f32; 6]> = Vec::with_capacity(options.max_detections);
        for i in 0..candidates.len() {
            if suppressed[i] {
                continue;
            }
            if keep.len() >= options.max_detections {
                break;
            }

            for j in (i + 1)..candidates.len() {
                if suppressed[j] {
                    continue;
                }

                let offset_i = candidates[i][5] * MAX_WH;
                let offset_j = candidates[j][5] * MAX_WH;
                let inter_x1 = (candidates[i][0] + offset_i).max(candidates[j][0] + offset_j);
                let inter_y1 = candidates[i][1].max(candidates[j][1]);
                let inter_x2 = (candidates[i][2] + offset_i).min(candidates[j][2] + offset_j);
                let inter_y2 = candidates[i][3].min(candidates[j][3]);
                let inter = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
                let area_i =
                    (candidates[i][2] - candidates[i][0]) * (candidates[i][3] - candidates[i][1]);
                let area_j =
                    (candidates[j][2] - candidates[j][0]) * (candidates[j][3] - candidates[j][1]);
                let union = area_i + area_j - inter;
                let iou = if union > 0.0 { inter / union } else { 0.0 };
                if iou > options.iou_threshold {
                    suppressed[j] = true;
                }
            }

            keep.push(candidates[i]);
        }

        output[index] = keep;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::{non_max_suppression, non_max_suppression_with_options, NmsOptions};
    use burn::prelude::*;

    fn outputs(anchor_major: &[f32], stride: usize, num_anchors: usize) -> Tensor<3> {
        let device = burn::tensor::Device::flex();
        let mut channel_major = Vec::with_capacity(anchor_major.len());
        for c in 0..stride {
            for a in 0..num_anchors {
                channel_major.push(anchor_major[a * stride + c]);
            }
        }
        Tensor::<1>::from_floats(channel_major.as_slice(), &device).reshape([
            1,
            stride,
            num_anchors,
        ])
    }

    #[test]
    fn non_max_suppression_preserves_python_batch_output_slots() {
        let device = burn::tensor::Device::flex();
        let output = Tensor::<3>::zeros([3, 5, 0], &device);
        let detections = non_max_suppression(&output, 0.5, 0.65);

        assert_eq!(detections.len(), 3);
        assert!(detections.iter().all(Vec::is_empty));
    }

    #[test]
    fn non_max_suppression_keeps_multiclass_candidates() {
        let preds = [10.0, 10.0, 4.0, 4.0, 0.8, 0.7];
        let detections = non_max_suppression(&outputs(&preds, 6, 1), 0.5, 0.65);
        let detections = &detections[0];

        assert_eq!(detections.len(), 2);
        assert!(detections.iter().any(|d| d[5] == 0.0 && d[4] == 0.8));
        assert!(detections.iter().any(|d| d[5] == 1.0 && d[4] == 0.7));
    }

    #[test]
    fn explicit_detection_limit_supports_dense_images() {
        let preds = [
            10.0, 10.0, 2.0, 2.0, 0.9, //
            20.0, 20.0, 2.0, 2.0, 0.8, //
            30.0, 30.0, 2.0, 2.0, 0.7,
        ];
        let detections = non_max_suppression_with_options(
            &outputs(&preds, 5, 3),
            NmsOptions {
                confidence_threshold: 0.5,
                iou_threshold: 0.65,
                max_detections: 2,
                max_candidates: 30_000,
            },
        );

        assert_eq!(detections[0].len(), 2);
    }

    #[test]
    fn non_max_suppression_applies_python_max_nms_cap_before_greedy_nms() {
        let num_anchors = 30_001;
        let stride = 6;
        let mut preds = Vec::with_capacity(num_anchors * stride);
        for i in 0..num_anchors {
            if i == 0 {
                preds.extend_from_slice(&[100.0, 100.0, 1.0, 1.0, 0.0, 0.000001]);
            } else {
                preds.extend_from_slice(&[0.0, 0.0, 1.0, 1.0, i as f32 / num_anchors as f32, 0.0]);
            }
        }

        let detections = non_max_suppression(&outputs(&preds, stride, num_anchors), 0.0, 0.65);
        let detections = &detections[0];

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0][5], 0.0);
    }

    #[test]
    fn non_max_suppression_uses_python_class_offset_boxes() {
        let preds = [
            8005.0, 5.0, 10.0, 10.0, 0.9, 0.0, //
            325.0, 5.0, 10.0, 10.0, 0.0, 0.8,
        ];
        let detections = non_max_suppression(&outputs(&preds, 6, 2), 0.5, 0.65);
        let detections = &detections[0];

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0][5], 0.0);
    }

    #[test]
    fn non_max_suppression_keeps_boxes_at_exact_python_iou_threshold() {
        let preds = [
            1.0, 1.0, 2.0, 2.0, 0.9, //
            2.0, 1.0, 2.0, 2.0, 0.8,
        ];
        let detections = non_max_suppression(&outputs(&preds, 5, 2), 0.5, 1.0 / 3.0);
        let detections = &detections[0];

        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0][4], 0.9);
        assert_eq!(detections[1][4], 0.8);
    }

    #[test]
    fn non_max_suppression_matches_python_multiclass_fixture() {
        let preds = [
            10.0, 10.0, 4.0, 4.0, 0.9, 0.2, //
            10.0, 10.0, 4.0, 4.0, 0.8, 0.1, //
            10.0, 10.0, 4.0, 4.0, 0.7, 0.75, //
            30.0, 30.0, 4.0, 4.0, 0.001, 0.5, //
            50.0, 50.0, 4.0, 4.0, 0.6, 0.65,
        ];

        let detections = non_max_suppression(&outputs(&preds, 6, 5), 0.5, 0.65);
        let detections = &detections[0];

        assert_eq!(detections.len(), 4);
        let expected = [
            (8.0, 8.0, 12.0, 12.0, 0.9, 0),
            (8.0, 8.0, 12.0, 12.0, 0.75, 1),
            (48.0, 48.0, 52.0, 52.0, 0.65, 1),
            (48.0, 48.0, 52.0, 52.0, 0.6, 0),
        ];
        for (actual, expected) in detections.iter().zip(expected) {
            assert!((actual[0] - expected.0).abs() < 1e-6);
            assert!((actual[1] - expected.1).abs() < 1e-6);
            assert!((actual[2] - expected.2).abs() < 1e-6);
            assert!((actual[3] - expected.3).abs() < 1e-6);
            assert!((actual[4] - expected.4).abs() < 1e-6);
            assert_eq!(actual[5], expected.5 as f32);
        }
    }
}
