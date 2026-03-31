use anyhow::Result;
use candle_core::Device;
use indicatif::{ProgressBar, ProgressStyle};

use yolo_data::dataset::{self, Dataset, Sample};
use yolo_model::metrics::{self, MatchResult};
use yolo_model::model::YOLO;
use yolo_model::nms;

/// Run evaluation on the validation set.
/// Returns (mAP, mAP50, recall, precision).
pub fn test(
    model: &YOLO,
    dataset: &Dataset,
    num_classes: usize,
    device: &Device,
    batch_size: usize,
) -> Result<(f32, f32, f32, f32)> {
    let n = dataset.len();
    let num_batches = (n + batch_size - 1) / batch_size;

    // IoU thresholds: 0.5 to 0.95 in steps of 0.05
    let iou_thresholds: Vec<f32> = (0..10).map(|i| 0.5 + i as f32 * 0.05).collect();

    let pb = ProgressBar::new(num_batches as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap(),
    );
    pb.set_message("Evaluating");

    let mut all_results: Vec<MatchResult> = Vec::new();

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(n);

        let mut samples: Vec<Sample> = Vec::new();
        for i in start..end {
            samples.push(dataset.get(i, device)?);
        }

        let batch = dataset::collate(&samples, device)?;
        let images = batch.images; // [B, 3, H, W] already in 0..1

        // Forward (inference)
        let output = model.forward_infer(&images)?; // [B, 4+nc, A]

        // NMS
        let detections = nms::batch_nms(&output, 0.001, 0.65, 300)?;

        // Match against GT
        let gt_cls: Vec<f32> = batch.cls.squeeze(1)?.to_vec1()?;
        let gt_bbox: Vec<Vec<f32>> = (0..batch.bbox.dims()[0])
            .map(|i| batch.bbox.get(i).unwrap().to_vec1::<f32>().unwrap())
            .collect();
        let gt_idx: Vec<f32> = batch.idx.to_vec1()?;

        let input_size = images.dims()[3] as f32;

        for b in 0..samples.len() {
            // Get GT for this image
            let mut gt_boxes: Vec<[f32; 5]> = Vec::new();
            for t in 0..gt_idx.len() {
                if gt_idx[t] as usize == b {
                    // Convert normalized xywh to pixel xyxy
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

            // Convert detections to metric format
            let pred_boxes: Vec<[f32; 6]> = detections[b]
                .iter()
                .map(|d| [d.x1, d.y1, d.x2, d.y2, d.confidence, d.class as f32])
                .collect();

            let result = metrics::compute_metric(&pred_boxes, &gt_boxes, &iou_thresholds);
            all_results.push(result);
        }

        pb.inc(1);
    }

    pb.finish_with_message("Done");

    let (mean_ap, map50, recall, precision) = metrics::compute_ap(&all_results, num_classes);
    println!(
        "  Precision: {:.3}  Recall: {:.3}  mAP@50: {:.3}  mAP@50:95: {:.3}",
        precision, recall, map50, mean_ap
    );

    Ok((mean_ap, map50, recall, precision))
}
