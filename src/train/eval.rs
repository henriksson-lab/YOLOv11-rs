use anyhow::Result;
use burn::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

use crate::data::dataset::{self, Dataset, Sample};
use crate::model::metrics::{self, MatchResult};
use crate::model::model::YOLO;
use crate::model::nms;

/// Run evaluation on the validation set.
/// Returns (mAP, mAP50, recall, precision).
pub fn test<B: Backend>(
    model: &YOLO<B>,
    dataset: &Dataset,
    num_classes: usize,
    device: &B::Device,
    batch_size: usize,
) -> Result<(f32, f32, f32, f32)> {
    let n = dataset.len();
    let num_batches = (n + batch_size - 1) / batch_size;

    let iou_thresholds: Vec<f32> = (0..10).map(|i| 0.5 + i as f32 * 0.05).collect();

    let pb = ProgressBar::new(num_batches as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap(),
    );
    pb.set_message("Evaluating");

    let mut all_results: Vec<MatchResult> = Vec::new();

    let mut total_data_time = std::time::Duration::ZERO;
    let mut total_forward_time = std::time::Duration::ZERO;
    let mut total_nms_time = std::time::Duration::ZERO;
    let mut total_metric_time = std::time::Duration::ZERO;
    let eval_start = Instant::now();

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(n);

        let t0 = Instant::now();
        let mut samples: Vec<Sample<B>> = Vec::new();
        for i in start..end {
            samples.push(dataset.get(i, device)?);
        }

        let batch = dataset::collate(&samples, device);
        let images = batch.images;
        total_data_time += t0.elapsed();

        // Forward (inference)
        let t1 = Instant::now();
        let output = model.forward_infer(images); // [B, 4+nc, A]
        total_forward_time += t1.elapsed();

        // NMS
        let t2 = Instant::now();
        let detections = nms::batch_nms(&output, 0.001, 0.65, 300);
        total_nms_time += t2.elapsed();

        // Match against GT
        let t3 = Instant::now();
        let gt_cls: Vec<f32> = batch.cls.clone().squeeze::<1>().to_data().to_vec().unwrap();
        let gt_bbox_data: Vec<f32> = batch.bbox.to_data().to_vec().unwrap();
        let num_gt = gt_cls.len();
        let gt_bbox: Vec<[f32; 4]> = (0..num_gt)
            .map(|i| [
                gt_bbox_data[i * 4],
                gt_bbox_data[i * 4 + 1],
                gt_bbox_data[i * 4 + 2],
                gt_bbox_data[i * 4 + 3],
            ])
            .collect();
        let gt_idx: Vec<f32> = batch.idx.to_data().to_vec().unwrap();

        let input_size = samples[0].image.dims()[2] as f32; // H dimension

        for b in 0..samples.len() {
            let mut gt_boxes: Vec<[f32; 5]> = Vec::new();
            for t in 0..gt_idx.len() {
                if gt_idx[t] as usize == b {
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
                .map(|d| [d.x1, d.y1, d.x2, d.y2, d.confidence, d.class as f32])
                .collect();

            let result = metrics::compute_metric(&pred_boxes, &gt_boxes, &iou_thresholds);
            all_results.push(result);
        }
        total_metric_time += t3.elapsed();

        pb.inc(1);
    }

    let total_elapsed = eval_start.elapsed();
    pb.finish_with_message("Done");

    let (mean_ap, map50, recall, precision) = metrics::compute_ap(&all_results, num_classes);
    println!(
        "  Precision: {:.3}  Recall: {:.3}  mAP@50: {:.3}  mAP@50:95: {:.3}",
        precision, recall, map50, mean_ap
    );

    // Timing summary
    let n_images = n as f64;
    println!("\n  Benchmark ({} images, batch_size={}):", n, batch_size);
    println!("    Data loading:  {:>8.1}ms total, {:>6.2}ms/img", total_data_time.as_secs_f64() * 1e3, total_data_time.as_secs_f64() * 1e3 / n_images);
    println!("    Forward pass:  {:>8.1}ms total, {:>6.2}ms/img", total_forward_time.as_secs_f64() * 1e3, total_forward_time.as_secs_f64() * 1e3 / n_images);
    println!("    NMS:           {:>8.1}ms total, {:>6.2}ms/img", total_nms_time.as_secs_f64() * 1e3, total_nms_time.as_secs_f64() * 1e3 / n_images);
    println!("    Metrics:       {:>8.1}ms total, {:>6.2}ms/img", total_metric_time.as_secs_f64() * 1e3, total_metric_time.as_secs_f64() * 1e3 / n_images);
    println!("    Total:         {:>8.1}ms total, {:>6.2}ms/img", total_elapsed.as_secs_f64() * 1e3, total_elapsed.as_secs_f64() * 1e3 / n_images);
    println!("    Throughput:    {:.1} img/s", n_images / total_elapsed.as_secs_f64());

    Ok((mean_ap, map50, recall, precision))
}

