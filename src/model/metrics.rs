/// COCO-style mAP evaluation on CPU.

/// Compute TP/FP for predictions against targets at multiple IoU thresholds.
///
/// pred_boxes: [N, 6] = (x1, y1, x2, y2, conf, class).
/// gt_boxes: [M, 5] = (class, x1, y1, x2, y2).
/// iou_thresholds: e.g. [0.5, 0.55, ..., 0.95].
pub fn compute_metric(
    pred_boxes: &[[f32; 6]],
    gt_boxes: &[[f32; 5]],
    iou_thresholds: &[f32],
) -> Vec<Vec<bool>> {
    let n_iou = iou_thresholds.len();
    let num_preds = pred_boxes.len();
    let num_gt = gt_boxes.len();

    let mut tp = vec![vec![false; n_iou]; num_preds];
    let pred_cls: Vec<f32> = pred_boxes.iter().map(|p| p[5]).collect();
    let gt_cls: Vec<f32> = gt_boxes.iter().map(|g| g[0]).collect();

    if num_preds == 0 || num_gt == 0 {
        return tp;
    }

    let mut iou = vec![vec![0.0f32; num_preds]; num_gt];
    for label in 0..num_gt {
        for detect in 0..num_preds {
            let ix1 = gt_boxes[label][1].max(pred_boxes[detect][0]);
            let iy1 = gt_boxes[label][2].max(pred_boxes[detect][1]);
            let ix2 = gt_boxes[label][3].min(pred_boxes[detect][2]);
            let iy2 = gt_boxes[label][4].min(pred_boxes[detect][3]);
            let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
            let area_label = (gt_boxes[label][3] - gt_boxes[label][1])
                * (gt_boxes[label][4] - gt_boxes[label][2]);
            let area_detect = (pred_boxes[detect][2] - pred_boxes[detect][0])
                * (pred_boxes[detect][3] - pred_boxes[detect][1]);
            iou[label][detect] = inter / (area_label + area_detect - inter + 1e-7);
        }
    }

    for (threshold_index, &threshold) in iou_thresholds.iter().enumerate() {
        let mut matches = Vec::new();
        for label in 0..num_gt {
            for detect in 0..num_preds {
                if iou[label][detect] >= threshold && gt_cls[label] == pred_cls[detect] {
                    matches.push((label, detect, iou[label][detect]));
                }
            }
        }

        if matches.len() > 1 {
            matches.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            let mut unique_detect = Vec::new();
            for detect in 0..num_preds {
                if let Some(&match_) = matches.iter().find(|&&(_, d, _)| d == detect) {
                    unique_detect.push(match_);
                }
            }
            matches = unique_detect;
            let mut unique_label = Vec::new();
            for label in 0..num_gt {
                if let Some(&match_) = matches.iter().find(|&&(l, _, _)| l == label) {
                    unique_label.push(match_);
                }
            }
            matches = unique_label;
        }

        for (_, detect, _) in matches {
            tp[detect][threshold_index] = true;
        }
    }

    tp
}

/// Compute AP (Average Precision) from accumulated match arrays.
///
/// Returns `(tp, fp, m_pre, m_rec, map50, mean_ap)` to match the Python
/// `compute_ap` return order.
pub fn compute_ap(
    tp: &[Vec<bool>],
    conf: &[f32],
    output: &[f32],
    target: &[f32],
) -> (Vec<f32>, Vec<f32>, f32, f32, f32, f32) {
    if target.is_empty() {
        return (
            Vec::new(),
            Vec::new(),
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        );
    }
    let n_iou = tp.first().map(|row| row.len()).unwrap_or(10);

    let mut sorted: Vec<usize> = (0..conf.len()).collect();
    sorted.sort_by(|&a, &b| conf[b].partial_cmp(&conf[a]).unwrap());
    let all_preds: Vec<(f32, f32, Vec<bool>)> = sorted
        .iter()
        .map(|&i| (conf[i], output[i], tp[i].clone()))
        .collect();

    let mut unique_classes = target.to_vec();
    unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_classes.dedup();
    let mut nt = Vec::with_capacity(unique_classes.len());
    for &class in &unique_classes {
        nt.push(target.iter().filter(|&&gt| gt == class).count());
    }
    let nc = unique_classes.len();
    let mut p_curve = vec![vec![0.0f32; 1000]; nc];
    let mut r_curve = vec![vec![0.0f32; 1000]; nc];
    let mut ap = vec![vec![0.0f32; n_iou]; nc];
    let px: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();

    let interp_increasing = |x: f32, xp: &[f32], fp: &[f32], left: f32| -> f32 {
        if xp.is_empty() {
            return left;
        }
        if x < xp[0] {
            return left;
        }
        if x >= xp[xp.len() - 1] {
            return fp[fp.len() - 1];
        }
        for i in 1..xp.len() {
            if x <= xp[i] {
                let denom = xp[i] - xp[i - 1];
                if denom.abs() < 1e-16 {
                    return fp[i];
                }
                let t = (x - xp[i - 1]) / denom;
                return fp[i - 1] + t * (fp[i] - fp[i - 1]);
            }
        }
        fp[fp.len() - 1]
    };

    for (ci, &c) in unique_classes.iter().enumerate() {
        let class_preds: Vec<&(f32, f32, Vec<bool>)> =
            all_preds.iter().filter(|p| p.1 == c).collect();
        let nl = nt[ci] as f32;
        let no = class_preds.len();
        if no == 0 || nl == 0.0 {
            continue;
        }

        let conf: Vec<f32> = class_preds.iter().map(|p| p.0).collect();
        let mut tpc = vec![vec![0.0f32; n_iou]; no];
        let mut fpc = vec![vec![0.0f32; n_iou]; no];

        for i in 0..no {
            for j in 0..n_iou {
                let tp = if class_preds[i].2[j] { 1.0 } else { 0.0 };
                let fp = 1.0 - tp;
                tpc[i][j] = tp + if i > 0 { tpc[i - 1][j] } else { 0.0 };
                fpc[i][j] = fp + if i > 0 { fpc[i - 1][j] } else { 0.0 };
            }
        }

        let recall0: Vec<f32> = tpc.iter().map(|row| row[0] / (nl + 1e-16)).collect();
        let precision0: Vec<f32> = tpc
            .iter()
            .zip(fpc.iter())
            .map(|(tp, fp)| tp[0] / (tp[0] + fp[0] + 1e-16))
            .collect();

        for (pi, &x) in px.iter().enumerate() {
            let xp: Vec<f32> = conf.iter().map(|&c| -c).collect();
            p_curve[ci][pi] = interp_increasing(-x, &xp, &precision0, 1.0);
            r_curve[ci][pi] = interp_increasing(-x, &xp, &recall0, 0.0);
        }

        for iou_idx in 0..n_iou {
            let recall: Vec<f32> = tpc.iter().map(|row| row[iou_idx] / (nl + 1e-16)).collect();
            let precision: Vec<f32> = tpc
                .iter()
                .zip(fpc.iter())
                .map(|(tp, fp)| tp[iou_idx] / (tp[iou_idx] + fp[iou_idx] + 1e-16))
                .collect();

            let mut m_rec = Vec::with_capacity(recall.len() + 2);
            m_rec.push(0.0);
            m_rec.extend_from_slice(&recall);
            m_rec.push(1.0);

            let mut m_pre = Vec::with_capacity(precision.len() + 2);
            m_pre.push(1.0);
            m_pre.extend_from_slice(&precision);
            m_pre.push(0.0);

            for i in (0..m_pre.len() - 1).rev() {
                m_pre[i] = m_pre[i].max(m_pre[i + 1]);
            }

            let x: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
            let y: Vec<f32> = x
                .iter()
                .map(|&xi| interp_increasing(xi, &m_rec, &m_pre, 0.0))
                .collect();
            ap[ci][iou_idx] = x
                .windows(2)
                .zip(y.windows(2))
                .map(|(xw, yw)| (xw[1] - xw[0]) * (yw[0] + yw[1]) * 0.5)
                .sum();
        }
    }

    let f1_mean: Vec<f32> = (0..1000)
        .map(|i| {
            let p = p_curve.iter().map(|row| row[i]).sum::<f32>() / nc as f32;
            let r = r_curve.iter().map(|row| row[i]).sum::<f32>() / nc as f32;
            2.0 * p * r / (p + r + 1e-16)
        })
        .collect();
    let f1_smoothed = if f1_mean.is_empty() {
        Vec::new()
    } else {
        let nf = ((f1_mean.len() as f32 * 0.1 * 2.0).round() as usize) / 2 + 1;
        let pad = nf / 2;
        let mut yp = Vec::with_capacity(f1_mean.len() + pad * 2);
        yp.extend(std::iter::repeat(f1_mean[0]).take(pad));
        yp.extend_from_slice(&f1_mean);
        yp.extend(std::iter::repeat(*f1_mean.last().unwrap()).take(pad));
        (0..f1_mean.len())
            .map(|i| yp[i..i + nf].iter().sum::<f32>() / nf as f32)
            .collect()
    };
    let best_idx = f1_smoothed
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let p: Vec<f32> = p_curve.iter().map(|row| row[best_idx]).collect();
    let r: Vec<f32> = r_curve.iter().map(|row| row[best_idx]).collect();
    let tp: Vec<f32> = r
        .iter()
        .zip(nt.iter())
        .map(|(&recall, &class_count)| (recall * class_count as f32).round())
        .collect();
    let fp: Vec<f32> = tp
        .iter()
        .zip(p.iter())
        .map(|(&tp, &precision)| (tp / (precision + 1e-16) - tp).round())
        .collect();

    let m_pre = p.iter().sum::<f32>() / nc as f32;
    let m_rec = r.iter().sum::<f32>() / nc as f32;
    let map50 = ap.iter().map(|row| row[0]).sum::<f32>() / nc as f32;
    let mean_ap = ap
        .iter()
        .map(|row| row.iter().sum::<f32>() / n_iou as f32)
        .sum::<f32>()
        / nc as f32;

    (tp, fp, m_pre, m_rec, map50, mean_ap)
}

#[cfg(test)]
mod tests {
    use super::{compute_ap, compute_metric};

    #[test]
    fn compute_ap_returns_python_order() {
        let tp_in = vec![vec![true; 10]];
        let conf = vec![0.9];
        let pred_cls = vec![0.0];
        let gt_cls = vec![0.0];

        let (tp, fp, precision, recall, map50, mean_ap) =
            compute_ap(&tp_in, &conf, &pred_cls, &gt_cls);

        assert_eq!(tp, vec![1.0]);
        assert_eq!(fp, vec![0.0]);
        assert!((precision - 1.0).abs() < 1e-6);
        assert!((recall - 1.0).abs() < 1e-6);
        assert!(map50 > 0.99);
        assert!(mean_ap > 0.99);
    }

    #[test]
    fn compute_ap_empty_targets_match_python_empty_mean_behavior() {
        let (tp, fp, precision, recall, map50, mean_ap) = compute_ap(&[], &[], &[], &[]);

        assert!(tp.is_empty());
        assert!(fp.is_empty());
        assert!(precision.is_nan());
        assert!(recall.is_nan());
        assert!(map50.is_nan());
        assert!(mean_ap.is_nan());
    }

    #[test]
    fn compute_metric_matches_python_numpy_unique_reordering() {
        let pred_boxes = [
            [0.0, 0.0, 8.0, 8.0, 0.8, 0.0],
            [0.0, 0.0, 10.0, 10.0, 0.7, 0.0],
        ];
        let gt_boxes = [[0.0, 0.0, 0.0, 10.0, 10.0]];

        let tp = compute_metric(&pred_boxes, &gt_boxes, &[0.5]);

        assert_eq!(tp, vec![vec![true], vec![false]]);
    }

    #[test]
    fn compute_metric_compares_float_classes_like_python() {
        let pred_boxes = [[0.0, 0.0, 10.0, 10.0, 0.9, 1.0]];
        let gt_boxes = [[1.5, 0.0, 0.0, 10.0, 10.0]];

        let tp = compute_metric(&pred_boxes, &gt_boxes, &[0.5]);

        assert_eq!(tp, vec![vec![false]]);
    }

    #[test]
    fn compute_ap_compares_float_classes_like_python() {
        let tp_in = vec![vec![true; 10]];
        let (tp, fp, precision, recall, map50, mean_ap) =
            compute_ap(&tp_in, &[0.9], &[1.0], &[1.5]);

        assert_eq!(tp, vec![0.0]);
        assert_eq!(fp, vec![0.0]);
        assert_eq!(precision, 0.0);
        assert_eq!(recall, 0.0);
        assert_eq!(map50, 0.0);
        assert_eq!(mean_ap, 0.0);
    }

    #[test]
    fn compute_metric_and_ap_match_python_fixture() {
        let pred_boxes = [
            [0.0, 0.0, 10.0, 10.0, 0.90, 0.0],
            [1.0, 1.0, 9.0, 9.0, 0.80, 0.0],
            [20.0, 20.0, 30.0, 30.0, 0.70, 1.0],
            [40.0, 40.0, 50.0, 50.0, 0.60, 1.0],
        ];
        let gt_boxes = [[0.0, 0.0, 0.0, 10.0, 10.0], [1.0, 20.0, 20.0, 30.0, 30.0]];

        let tp_in = compute_metric(&pred_boxes, &gt_boxes, &[0.5, 0.75]);

        assert_eq!(
            tp_in,
            vec![
                vec![true, true],
                vec![false, false],
                vec![true, true],
                vec![false, false],
            ]
        );

        let (tp, fp, m_pre, m_rec, map50, mean_ap) = compute_ap(
            &tp_in,
            &[0.90, 0.80, 0.70, 0.60],
            &[0.0, 0.0, 1.0, 1.0],
            &[0.0, 1.0],
        );
        assert_eq!(tp, vec![1.0, 1.0]);
        assert_eq!(fp, vec![1.0, 0.0]);
        assert!((m_pre - 0.6241241).abs() < 1e-6);
        assert!((m_rec - 1.0).abs() < 1e-6);
        assert!((map50 - 0.995).abs() < 1e-6);
        assert!((mean_ap - 0.995).abs() < 1e-6);
    }
}
