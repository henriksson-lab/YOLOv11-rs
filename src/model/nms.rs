/// Non-maximum suppression on CPU.

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
/// `preds` is shaped [4+nc, num_anchors] (cx, cy, w, h, cls_scores...).
/// Returns up to `max_det` detections per image.
pub fn non_max_suppression(
    preds: &[Vec<f32>],  // [num_anchors][4+nc]
    conf_threshold: f32,
    iou_threshold: f32,
    max_det: usize,
) -> Vec<Detection> {
    let nc = if preds.is_empty() {
        return Vec::new();
    } else {
        preds[0].len() - 4
    };

    // Collect candidates above confidence threshold
    let mut candidates: Vec<Detection> = Vec::new();
    for pred in preds {
        let cx = pred[0];
        let cy = pred[1];
        let w = pred[2];
        let h = pred[3];

        // Find max class score
        let mut max_score = 0.0f32;
        let mut max_cls = 0usize;
        for c in 0..nc {
            if pred[4 + c] > max_score {
                max_score = pred[4 + c];
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

    // Sort by confidence descending
    candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    // Greedy NMS per class
    let mut keep: Vec<Detection> = Vec::new();
    while !candidates.is_empty() && keep.len() < max_det {
        let best = candidates.remove(0);
        candidates.retain(|c| {
            if c.class != best.class {
                return true;
            }
            iou_single(&best, c) < iou_threshold
        });
        keep.push(best);
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
pub fn batch_nms(
    output: &candle_core::Tensor,
    conf_threshold: f32,
    iou_threshold: f32,
    max_det: usize,
) -> anyhow::Result<Vec<Vec<Detection>>> {
    let (batch, _no, num_a) = output.dims3()?;
    let mut results = Vec::with_capacity(batch);

    for b in 0..batch {
        let img_out = output.get(b)?; // [4+nc, A]
        let img_out = img_out.transpose(0, 1)?; // [A, 4+nc]
        let data: Vec<Vec<f32>> = (0..num_a)
            .map(|i| img_out.get(i).unwrap().to_vec1::<f32>().unwrap())
            .collect();
        results.push(non_max_suppression(&data, conf_threshold, iou_threshold, max_det));
    }
    Ok(results)
}
