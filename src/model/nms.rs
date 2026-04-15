/// Non-maximum suppression on CPU.

use burn::prelude::*;

#[derive(Debug, Clone)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class: usize,
}

/// Apply NMS to model output for a single image.
///
/// `preds` is a flat slice of [num_anchors * stride] where stride = 4 + nc.
/// Returns up to `max_det` detections per image.
pub fn non_max_suppression(
    preds: &[f32],
    stride: usize,
    num_anchors: usize,
    conf_threshold: f32,
    iou_threshold: f32,
    max_det: usize,
) -> Vec<Detection> {
    let mut candidates: Vec<Detection> = Vec::new();
    for i in 0..num_anchors {
        let base = i * stride;
        let cx = preds[base];
        let cy = preds[base + 1];
        let w = preds[base + 2];
        let h = preds[base + 3];

        let cls_scores = &preds[base + 4..base + stride];
        let mut max_score = 0.0f32;
        let mut max_cls = 0usize;
        for (c, &score) in cls_scores.iter().enumerate() {
            if score > max_score {
                max_score = score;
                max_cls = c;
            }
        }

        if max_score < conf_threshold {
            continue;
        }

        candidates.push(Detection {
            x1: cx - w * 0.5,
            y1: cy - h * 0.5,
            x2: cx + w * 0.5,
            y2: cy + h * 0.5,
            confidence: max_score,
            class: max_cls,
        });
    }

    candidates.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    // Greedy NMS using a suppressed flag instead of Vec::remove(0)
    let mut suppressed = vec![false; candidates.len()];
    let mut keep: Vec<Detection> = Vec::with_capacity(max_det);

    for i in 0..candidates.len() {
        if suppressed[i] {
            continue;
        }
        if keep.len() >= max_det {
            break;
        }
        // Suppress lower-confidence overlapping detections of the same class
        for j in (i + 1)..candidates.len() {
            if !suppressed[j]
                && candidates[j].class == candidates[i].class
                && iou_single(&candidates[i], &candidates[j]) >= iou_threshold
            {
                suppressed[j] = true;
            }
        }
        keep.push(candidates[i].clone());
    }

    keep
}

fn iou_single(a: &Detection, b: &Detection) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);
    let inter = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union = area_a + area_b - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Batch NMS: process model output tensor [B, 4+nc, A] into Vec<Vec<Detection>>.
pub fn batch_nms<B: Backend>(
    output: &Tensor<B, 3>,
    conf_threshold: f32,
    iou_threshold: f32,
    max_det: usize,
) -> Vec<Vec<Detection>> {
    let [batch, stride, num_a] = output.dims();
    let mut results = Vec::with_capacity(batch);

    for b in 0..batch {
        let img_out = output.clone().narrow(0, b, 1).squeeze::<2>(); // [stride, A]
        let img_out = img_out.swap_dims(0, 1); // [A, stride]
        let data: Vec<f32> = img_out.to_data().to_vec().unwrap();
        results.push(non_max_suppression(
            &data,
            stride,
            num_a,
            conf_threshold,
            iou_threshold,
            max_det,
        ));
    }
    results
}
