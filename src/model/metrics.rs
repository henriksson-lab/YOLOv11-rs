/// COCO-style mAP evaluation on CPU.

/// A single prediction result matched against ground truth.
pub struct MatchResult {
    /// [num_preds, num_iou_thresholds] — true if pred matches a GT at that IoU threshold.
    pub tp: Vec<Vec<bool>>,
    /// Confidence scores [num_preds].
    pub conf: Vec<f32>,
    /// Predicted classes [num_preds].
    pub pred_cls: Vec<usize>,
    /// Ground truth classes [num_gt].
    pub gt_cls: Vec<usize>,
}

/// Compute TP/FP for predictions against targets at multiple IoU thresholds.
///
/// pred_boxes: [N, 6] = (x1, y1, x2, y2, conf, class).
/// gt_boxes: [M, 5] = (class, x1, y1, x2, y2).
/// iou_thresholds: e.g. [0.5, 0.55, ..., 0.95].
pub fn compute_metric(
    pred_boxes: &[[f32; 6]],
    gt_boxes: &[[f32; 5]],
    iou_thresholds: &[f32],
) -> MatchResult {
    let n_iou = iou_thresholds.len();
    let num_preds = pred_boxes.len();
    let num_gt = gt_boxes.len();

    let mut tp = vec![vec![false; n_iou]; num_preds];
    let conf: Vec<f32> = pred_boxes.iter().map(|p| p[4]).collect();
    let pred_cls: Vec<usize> = pred_boxes.iter().map(|p| p[5] as usize).collect();
    let gt_cls: Vec<usize> = gt_boxes.iter().map(|g| g[0] as usize).collect();

    if num_preds == 0 || num_gt == 0 {
        return MatchResult { tp, conf, pred_cls, gt_cls };
    }

    // For each pred, find best matching GT
    let mut gt_matched = vec![vec![false; n_iou]; num_gt];

    for i in 0..num_preds {
        let pc = pred_cls[i];
        let mut best_iou = 0.0f32;
        let mut best_j = 0usize;

        for j in 0..num_gt {
            if gt_cls[j] != pc {
                continue;
            }
            let iou = box_iou_single(
                &[pred_boxes[i][0], pred_boxes[i][1], pred_boxes[i][2], pred_boxes[i][3]],
                &[gt_boxes[j][1], gt_boxes[j][2], gt_boxes[j][3], gt_boxes[j][4]],
            );
            if iou > best_iou {
                best_iou = iou;
                best_j = j;
            }
        }

        for (k, &thr) in iou_thresholds.iter().enumerate() {
            if best_iou >= thr && !gt_matched[best_j][k] && gt_cls[best_j] == pc {
                tp[i][k] = true;
                gt_matched[best_j][k] = true;
            }
        }
    }

    MatchResult { tp, conf, pred_cls, gt_cls }
}

fn box_iou_single(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Compute AP (Average Precision) from accumulated match results.
///
/// Returns (mAP@0.5:0.95, mAP@0.5, mean_recall, mean_precision).
pub fn compute_ap(
    all_results: &[MatchResult],
    num_classes: usize,
) -> (f32, f32, f32, f32) {
    if all_results.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n_iou = all_results[0].tp[0].len();

    // Collect all predictions sorted by confidence
    let mut all_preds: Vec<(f32, usize, Vec<bool>)> = Vec::new();
    let mut gt_per_class = vec![0usize; num_classes];

    for r in all_results {
        for &c in &r.gt_cls {
            if c < num_classes {
                gt_per_class[c] += 1;
            }
        }
        for i in 0..r.conf.len() {
            all_preds.push((r.conf[i], r.pred_cls[i], r.tp[i].clone()));
        }
    }

    // Sort by confidence descending
    all_preds.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut ap_sum = 0.0f32;
    let mut ap50_sum = 0.0f32;
    let mut recall_sum = 0.0f32;
    let mut precision_sum = 0.0f32;
    let mut valid_classes = 0;

    for c in 0..num_classes {
        if gt_per_class[c] == 0 {
            continue;
        }
        valid_classes += 1;

        // Filter preds for this class
        let class_preds: Vec<&(f32, usize, Vec<bool>)> =
            all_preds.iter().filter(|p| p.1 == c).collect();

        let n = class_preds.len();
        if n == 0 {
            continue;
        }

        for iou_idx in 0..n_iou {
            let mut tp_cumsum = Vec::with_capacity(n);
            let mut fp_cumsum = Vec::with_capacity(n);
            let mut tp_sum = 0;
            let mut fp_sum = 0;
            for p in &class_preds {
                if p.2[iou_idx] {
                    tp_sum += 1;
                } else {
                    fp_sum += 1;
                }
                tp_cumsum.push(tp_sum as f32);
                fp_cumsum.push(fp_sum as f32);
            }

            let gt_count = gt_per_class[c] as f32;
            let recall: Vec<f32> = tp_cumsum.iter().map(|&tp| tp / gt_count).collect();
            let precision: Vec<f32> = tp_cumsum
                .iter()
                .zip(fp_cumsum.iter())
                .map(|(&tp, &fp)| tp / (tp + fp + 1e-16))
                .collect();

            // 101-point interpolation
            let ap = interpolated_ap(&recall, &precision);
            ap_sum += ap;
            if iou_idx == 0 {
                ap50_sum += ap;
                if let Some(&last_r) = recall.last() {
                    recall_sum += last_r;
                }
                if let Some(&last_p) = precision.last() {
                    precision_sum += last_p;
                }
            }
        }
    }

    let vc = valid_classes.max(1) as f32;
    let mean_ap = ap_sum / (vc * n_iou as f32);
    let map50 = ap50_sum / vc;
    let mean_recall = recall_sum / vc;
    let mean_precision = precision_sum / vc;

    (mean_ap, map50, mean_recall, mean_precision)
}

/// 101-point interpolated AP.
fn interpolated_ap(recall: &[f32], precision: &[f32]) -> f32 {
    let mut ap = 0.0f32;
    for i in 0..=100 {
        let r_thr = i as f32 / 100.0;
        let mut max_p = 0.0f32;
        for (j, &r) in recall.iter().enumerate() {
            if r >= r_thr && precision[j] > max_p {
                max_p = precision[j];
            }
        }
        ap += max_p;
    }
    ap / 101.0
}
