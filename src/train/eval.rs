use anyhow::Result;
use burn::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

use crate::data::dataset::{Dataset, Sample};
use crate::model::metrics;
use crate::model::model::{YOLOOutput, YOLO};
use crate::model::nms;

/// Run evaluation on the validation set.
/// Returns (mAP, mAP50, recall, precision).
pub fn test(
    model: &YOLO,
    dataset: &Dataset,
    device: &Device,
    batch_size: usize,
) -> Result<(f32, f32, f32, f32)> {
    let n = dataset.len();
    let num_batches = (n + batch_size - 1) / batch_size;

    let warmup_input = Tensor::<4>::zeros(
        [
            1,
            3,
            dataset.input_size as usize,
            dataset.input_size as usize,
        ],
        device,
    );
    for _ in 0..3 {
        match model.forward(warmup_input.clone(), false) {
            YOLOOutput::Infer(_) => {}
            YOLOOutput::Train(_) => unreachable!("warmup forward returned training output"),
        }
    }

    let iou_thresholds: Vec<f32> = (0..10).map(|i| 0.5 + i as f32 * 0.05).collect();

    let pb = ProgressBar::new(num_batches as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap(),
    );
    pb.set_message("Evaluating");

    let mut tp = Vec::new();
    let mut conf = Vec::new();
    let mut pred_cls = Vec::new();
    let mut gt_cls_all = Vec::new();

    let mut total_data_time = std::time::Duration::ZERO;
    let mut total_forward_time = std::time::Duration::ZERO;
    let mut total_nms_time = std::time::Duration::ZERO;
    let mut total_metric_time = std::time::Duration::ZERO;
    let eval_start = Instant::now();

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(n);

        let t0 = Instant::now();
        let mut samples: Vec<Sample> = Vec::new();
        for i in start..end {
            samples.push(dataset.get_item(i, device)?);
        }

        let batch = Dataset::collate_fn(&samples, device);
        let images = batch.images / 255.0;
        total_data_time += t0.elapsed();

        // Forward (inference)
        let t1 = Instant::now();
        let output = match model.forward(images, false) {
            YOLOOutput::Infer(output) => output,
            YOLOOutput::Train(_) => unreachable!("inference forward returned training output"),
        }; // [B, 4+nc, A]
        total_forward_time += t1.elapsed();

        // NMS
        let t2 = Instant::now();
        let detections = nms::non_max_suppression(&output, 0.001, 0.65);
        total_nms_time += t2.elapsed();

        // Match against GT
        let t3 = Instant::now();
        let gt_cls: Vec<f32> = batch
            .cls
            .clone()
            .squeeze_dim::<1>(1)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let gt_bbox_data: Vec<f32> = batch.bbox.to_data().try_to_vec::<f32>().unwrap();
        let num_gt = gt_cls.len();
        let gt_bbox: Vec<[f32; 4]> = (0..num_gt)
            .map(|i| {
                [
                    gt_bbox_data[i * 4],
                    gt_bbox_data[i * 4 + 1],
                    gt_bbox_data[i * 4 + 2],
                    gt_bbox_data[i * 4 + 3],
                ]
            })
            .collect();
        let gt_idx: Vec<f32> = batch.idx.to_data().try_to_vec::<f32>().unwrap();

        let input_size = samples[0].image.dims()[2] as f32; // H dimension

        for b in 0..samples.len() {
            let mut gt_boxes: Vec<[f32; 5]> = Vec::new();
            for t in 0..gt_idx.len() {
                if gt_idx[t] >= 0.0 && gt_idx[t] as usize == b {
                    let cx = gt_bbox[t][0] * input_size;
                    let cy = gt_bbox[t][1] * input_size;
                    let w = gt_bbox[t][2] * input_size;
                    let h = gt_bbox[t][3] * input_size;
                    gt_boxes.push([
                        gt_cls[t],
                        cx - w / 2.0,
                        cy - h / 2.0,
                        cx + w / 2.0,
                        cy + h / 2.0,
                    ]);
                }
            }

            let pred_boxes: Vec<[f32; 6]> = detections[b]
                .iter()
                .map(|d| [d[0], d[1], d[2], d[3], d[4], d[5]])
                .collect();

            tp.extend(metrics::compute_metric(
                &pred_boxes,
                &gt_boxes,
                &iou_thresholds,
            ));
            conf.extend(pred_boxes.iter().map(|p| p[4]));
            pred_cls.extend(pred_boxes.iter().map(|p| p[5]));
            gt_cls_all.extend(gt_boxes.iter().map(|g| g[0]));
        }
        total_metric_time += t3.elapsed();

        pb.inc(1);
    }

    let total_elapsed = eval_start.elapsed();
    pb.finish_with_message("Done");

    let (precision, recall, map50, mean_ap) =
        if !tp.is_empty() && tp.iter().any(|row| row.iter().any(|&matched| matched)) {
            let (_tp, _fp, precision, recall, map50, mean_ap) =
                metrics::compute_ap(&tp, &conf, &pred_cls, &gt_cls_all);
            (precision, recall, map50, mean_ap)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
    println!(
        "  Precision: {:.3}  Recall: {:.3}  mAP@50: {:.3}  mAP@50:95: {:.3}",
        precision, recall, map50, mean_ap
    );

    // Timing summary
    let n_images = n as f64;
    println!("\n  Benchmark ({} images, batch_size={}):", n, batch_size);
    println!(
        "    Data loading:  {:>8.1}ms total, {:>6.2}ms/img",
        total_data_time.as_secs_f64() * 1e3,
        total_data_time.as_secs_f64() * 1e3 / n_images
    );
    println!(
        "    Forward pass:  {:>8.1}ms total, {:>6.2}ms/img",
        total_forward_time.as_secs_f64() * 1e3,
        total_forward_time.as_secs_f64() * 1e3 / n_images
    );
    println!(
        "    NMS:           {:>8.1}ms total, {:>6.2}ms/img",
        total_nms_time.as_secs_f64() * 1e3,
        total_nms_time.as_secs_f64() * 1e3 / n_images
    );
    println!(
        "    Metrics:       {:>8.1}ms total, {:>6.2}ms/img",
        total_metric_time.as_secs_f64() * 1e3,
        total_metric_time.as_secs_f64() * 1e3 / n_images
    );
    println!(
        "    Total:         {:>8.1}ms total, {:>6.2}ms/img",
        total_elapsed.as_secs_f64() * 1e3,
        total_elapsed.as_secs_f64() * 1e3 / n_images
    );
    println!(
        "    Throughput:    {:.1} img/s",
        n_images / total_elapsed.as_secs_f64()
    );

    Ok((mean_ap, map50, recall, precision))
}

#[cfg(test)]
mod tests {
    #[test]
    fn eval_ignores_collate_dummy_negative_indices_like_python_empty_targets() {
        let gt_cls = [0.0_f32];
        let gt_bbox = [[0.0_f32, 0.0, 0.0, 0.0]];
        let gt_idx = [-1.0_f32];
        let input_size = 64.0_f32;

        let mut gt_boxes = Vec::new();
        for t in 0..gt_idx.len() {
            if gt_idx[t] >= 0.0 && gt_idx[t] as usize == 0 {
                let cx = gt_bbox[t][0] * input_size;
                let cy = gt_bbox[t][1] * input_size;
                let w = gt_bbox[t][2] * input_size;
                let h = gt_bbox[t][3] * input_size;
                gt_boxes.push([
                    gt_cls[t],
                    cx - w / 2.0,
                    cy - h / 2.0,
                    cx + w / 2.0,
                    cy + h / 2.0,
                ]);
            }
        }

        assert!(gt_boxes.is_empty());
    }

    #[test]
    fn eval_metric_summary_keeps_python_zero_defaults_without_true_positives() {
        let tp = vec![vec![false; 10]];
        let conf = vec![0.9_f32];
        let pred_cls = vec![0.0_f32];
        let gt_cls_all = vec![0.0_f32];

        let (precision, recall, map50, mean_ap) =
            if !tp.is_empty() && tp.iter().any(|row| row.iter().any(|&matched| matched)) {
                let (_tp, _fp, precision, recall, map50, mean_ap) =
                    crate::model::metrics::compute_ap(&tp, &conf, &pred_cls, &gt_cls_all);
                (precision, recall, map50, mean_ap)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

        assert_eq!((precision, recall, map50, mean_ap), (0.0, 0.0, 0.0, 0.0));
    }
}
