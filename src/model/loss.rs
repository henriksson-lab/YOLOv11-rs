use burn::prelude::*;
use burn::tensor::activation::{log_softmax, sigmoid};

use crate::model::anchors;

// ---------------------------------------------------------------------------
// CIoU loss (Complete IoU)
// ---------------------------------------------------------------------------

/// Compute CIoU between pred [N, 4] and target [N, 4] boxes in xyxy format.
/// Returns (ciou, iou) both [N].
pub fn compute_ciou<B: Backend>(pred: &Tensor<B, 2>, target: &Tensor<B, 2>) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let p_x1 = pred.clone().narrow(1, 0, 1);
    let p_y1 = pred.clone().narrow(1, 1, 1);
    let p_x2 = pred.clone().narrow(1, 2, 1);
    let p_y2 = pred.clone().narrow(1, 3, 1);

    let t_x1 = target.clone().narrow(1, 0, 1);
    let t_y1 = target.clone().narrow(1, 1, 1);
    let t_x2 = target.clone().narrow(1, 2, 1);
    let t_y2 = target.clone().narrow(1, 3, 1);

    // Intersection
    let inter_x1 = p_x1.clone().max_pair(t_x1.clone());
    let inter_y1 = p_y1.clone().max_pair(t_y1.clone());
    let inter_x2 = p_x2.clone().min_pair(t_x2.clone());
    let inter_y2 = p_y2.clone().min_pair(t_y2.clone());

    let inter_w = (inter_x2 - inter_x1).clamp_min(0.0);
    let inter_h = (inter_y2 - inter_y1).clamp_min(0.0);
    let inter = inter_w * inter_h;

    // Union
    let p_w = p_x2.clone() - p_x1.clone();
    let p_h = p_y2.clone() - p_y1.clone();
    let t_w = t_x2.clone() - t_x1.clone();
    let t_h = t_y2.clone() - t_y1.clone();
    let area_p = p_w * p_h;
    let area_t = t_w * t_h;
    let union = area_p + area_t - inter.clone() + 1e-7;
    let iou = inter / union;

    // Enclosing box
    let enc_x1 = p_x1.clone().min_pair(t_x1.clone());
    let enc_y1 = p_y1.clone().min_pair(t_y1.clone());
    let enc_x2 = p_x2.clone().max_pair(t_x2.clone());
    let enc_y2 = p_y2.clone().max_pair(t_y2.clone());
    let enc_w = enc_x2 - enc_x1;
    let enc_h = enc_y2 - enc_y1;
    let c2 = enc_w.clone().powf_scalar(2.0) + enc_h.powf_scalar(2.0); // diagonal squared

    // Center distance
    let p_cx = (p_x1 + p_x2) * 0.5;
    let p_cy = (p_y1 + p_y2) * 0.5;
    let t_cx = (t_x1 + t_x2) * 0.5;
    let t_cy = (t_y1 + t_y2) * 0.5;
    let d2 = (p_cx - t_cx).powf_scalar(2.0) + (p_cy - t_cy).powf_scalar(2.0);

    // DIoU
    let diou = iou.clone() - d2 / (c2 + 1e-7);

    let iou_flat = iou.squeeze::<1>(1);
    let diou_flat = diou.squeeze::<1>(1);
    (diou_flat, iou_flat)
}

// ---------------------------------------------------------------------------
// DFL loss  (distribution focal loss for box regression)
// ---------------------------------------------------------------------------

/// Distribution Focal Loss.
/// pred_dist: [N, 4*ch] predicted distribution logits.
/// target: [N, 4] target values (continuous, will be discretized).
/// ch: number of DFL bins (16).
pub fn df_loss<B: Backend>(
    pred_dist: &Tensor<B, 2>,
    target: &Tensor<B, 2>,
    ch: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let target_flat: Vec<f32> = target.clone().reshape([target.dims()[0] * 4]).to_data().to_vec().unwrap();
    let n = pred_dist.dims()[0];

    let target_left: Vec<f32> = target_flat
        .iter()
        .map(|&v| v.floor().clamp(0.0, (ch - 1) as f32))
        .collect();
    let target_right: Vec<f32> = target_left
        .iter()
        .map(|&v| (v + 1.0).clamp(0.0, (ch - 1) as f32))
        .collect();
    let weight_right: Vec<f32> = target_flat
        .iter()
        .zip(target_left.iter())
        .map(|(&t, &l)| t - l)
        .collect();
    let weight_left: Vec<f32> = weight_right.iter().map(|&wr| 1.0 - wr).collect();

    // pred_dist: [N, 4*ch] -> [N*4, ch]
    let pd = pred_dist.clone().reshape([n * 4, ch]);
    let log_sm = log_softmax(pd, 1);

    let log_sm_data: Vec<f32> = log_sm.to_data().to_vec().unwrap();

    let mut loss_vals = Vec::with_capacity(n * 4);
    for i in 0..n * 4 {
        let row_start = i * ch;
        let ll = -log_sm_data[row_start + target_left[i] as usize] * weight_left[i];
        let lr = -log_sm_data[row_start + target_right[i] as usize] * weight_right[i];
        loss_vals.push(ll + lr);
    }

    let loss = Tensor::<B, 1>::from_floats(loss_vals.as_slice(), device).reshape([n, 4]);
    loss.mean_dim(1).squeeze::<1>(1) // [N]
}

// ---------------------------------------------------------------------------
// BCE with logits
// ---------------------------------------------------------------------------

/// Binary cross-entropy with logits (numerically stable).
pub fn bce_with_logits<B: Backend>(pred: &Tensor<B, 2>, target: &Tensor<B, 2>) -> Tensor<B, 2> {
    // max(x, 0) - x*target + log(1 + exp(-|x|))
    let relu_x = pred.clone().clamp_min(0.0);
    let neg_abs = pred.clone().abs().neg();
    let log_term = (neg_abs.exp() + 1.0).log();
    relu_x - pred.clone() * target.clone() + log_term
}

// ---------------------------------------------------------------------------
// Assigner (task-aligned, runs on CPU with no gradient)
// ---------------------------------------------------------------------------

/// Task-aligned assignment result.
pub struct AssignResult {
    /// Target labels per anchor [total_anchors], -1 for background.
    pub target_labels: Vec<i64>,
    /// Target boxes per anchor [total_anchors, 4].
    pub target_bboxes: Vec<[f32; 4]>,
    /// Target scores per anchor (alignment metric) [total_anchors].
    pub target_scores: Vec<f32>,
    /// Foreground mask [total_anchors].
    pub fg_mask: Vec<bool>,
}

/// Run task-aligned assignment on CPU.
pub fn task_aligned_assign(
    pd_scores: &[Vec<f32>],   // [A, nc]
    pd_bboxes: &[[f32; 4]],   // [A, 4]
    anc_points: &[[f32; 2]],  // [A, 2]
    gt_labels: &[i64],        // [G]
    gt_bboxes: &[[f32; 4]],   // [G, 4]
    top_k: usize,
) -> AssignResult {
    let num_anchors = pd_scores.len();
    let num_gt = gt_labels.len();
    let nc = if num_anchors > 0 { pd_scores[0].len() } else { 0 };

    if num_gt == 0 {
        return AssignResult {
            target_labels: vec![-1; num_anchors],
            target_bboxes: vec![[0.0; 4]; num_anchors],
            target_scores: vec![0.0; num_anchors],
            fg_mask: vec![false; num_anchors],
        };
    }

    // Check which anchors are inside which GT boxes
    let mut is_inside = vec![vec![false; num_gt]; num_anchors];
    for a in 0..num_anchors {
        let (ax, ay) = (anc_points[a][0], anc_points[a][1]);
        for g in 0..num_gt {
            let b = &gt_bboxes[g];
            if ax >= b[0] && ax <= b[2] && ay >= b[1] && ay <= b[3] {
                is_inside[a][g] = true;
            }
        }
    }

    // Compute alignment metric
    let alpha = 0.5f32;
    let beta = 6.0f32;
    let mut metrics = vec![vec![0.0f32; num_gt]; num_anchors];
    for a in 0..num_anchors {
        for g in 0..num_gt {
            if !is_inside[a][g] {
                continue;
            }
            let cls = gt_labels[g] as usize;
            let score = if cls < nc { pd_scores[a][cls] } else { 0.0 };
            let iou = box_iou_single(&pd_bboxes[a], &gt_bboxes[g]);
            metrics[a][g] = score.powf(alpha) * iou.powf(beta);
        }
    }

    // Top-k selection per GT
    let mut assigned_gt = vec![-1i64; num_anchors];
    let mut assigned_metric = vec![0.0f32; num_anchors];

    for g in 0..num_gt {
        let mut candidates: Vec<(f32, usize)> = Vec::new();
        for a in 0..num_anchors {
            if is_inside[a][g] && metrics[a][g] > 0.0 {
                candidates.push((metrics[a][g], a));
            }
        }
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let k = top_k.min(candidates.len());
        for i in 0..k {
            let (m, a) = candidates[i];
            if m > assigned_metric[a] {
                assigned_gt[a] = g as i64;
                assigned_metric[a] = m;
            }
        }
    }

    // Build outputs
    let mut target_labels = vec![-1i64; num_anchors];
    let mut target_bboxes = vec![[0.0f32; 4]; num_anchors];
    let mut target_scores = vec![0.0f32; num_anchors];
    let mut fg_mask = vec![false; num_anchors];

    for a in 0..num_anchors {
        if assigned_gt[a] >= 0 {
            let g = assigned_gt[a] as usize;
            target_labels[a] = gt_labels[g];
            target_bboxes[a] = gt_bboxes[g];
            target_scores[a] = assigned_metric[a];
            fg_mask[a] = true;
        }
    }

    // Normalize scores per GT so max = IoU
    for g in 0..num_gt {
        let mut max_metric = 0.0f32;
        for a in 0..num_anchors {
            if assigned_gt[a] == g as i64 {
                max_metric = max_metric.max(assigned_metric[a]);
            }
        }
        if max_metric > 0.0 {
            for a in 0..num_anchors {
                if assigned_gt[a] == g as i64 {
                    let iou = box_iou_single(&pd_bboxes[a], &gt_bboxes[g]);
                    target_scores[a] = (assigned_metric[a] / max_metric) * iou;
                }
            }
        }
    }

    AssignResult {
        target_labels,
        target_bboxes,
        target_scores,
        fg_mask,
    }
}

fn box_iou_single(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter_x1 = a[0].max(b[0]);
    let inter_y1 = a[1].max(b[1]);
    let inter_x2 = a[2].min(b[2]);
    let inter_y2 = a[3].min(b[3]);
    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter = inter_w * inter_h;
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// ComputeLoss — main training loss
// ---------------------------------------------------------------------------

pub struct ComputeLoss {
    pub nc: usize,
    pub ch: usize,   // DFL channels (16)
    pub nl: usize,   // number of detection layers
    pub stride: Vec<f32>,
    pub box_gain: f64,
    pub cls_gain: f64,
    pub dfl_gain: f64,
}

impl ComputeLoss {
    pub fn new<B: Backend>(
        nc: usize,
        nl: usize,
        stride: &Tensor<B, 1>,
        box_gain: f64,
        cls_gain: f64,
        dfl_gain: f64,
    ) -> Self {
        let stride_vec: Vec<f32> = stride.to_data().to_vec().unwrap();
        Self {
            nc,
            ch: 16,
            nl,
            stride: stride_vec,
            box_gain,
            cls_gain,
            dfl_gain,
        }
    }

    /// Compute training loss.
    pub fn compute<B: Backend>(
        &self,
        outputs: &[Tensor<B, 4>],
        target_cls: &Tensor<B, 2>,  // [num_targets, 1]
        target_box: &Tensor<B, 2>,  // [num_targets, 4]  normalized xywh
        target_idx: &Tensor<B, 1>,  // [num_targets]  batch index
        batch_size: usize,
        img_size: usize,
        device: &B::Device,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let no = self.nc + self.ch * 4;

        // Flatten outputs: [B, no, total_A]
        let mut flat: Vec<Tensor<B, 3>> = Vec::new();
        for o in outputs {
            let [b, _, h, w] = o.dims();
            flat.push(o.clone().reshape([b, no, h * w]));
        }
        let x = Tensor::cat(flat, 2); // [B, no, total_A]

        let box_ch = self.ch * 4;
        let pred_dist = x.clone().narrow(1, 0, box_ch);       // [B, 4*ch, A]
        let pred_scores = x.narrow(1, box_ch, self.nc);        // [B, nc, A]

        // Generate anchors
        let stride_t = Tensor::<B, 1>::from_floats(self.stride.as_slice(), device);
        let (anchor_points, strides) =
            crate::model::anchors::make_anchors::<B>(outputs, &stride_t, 0.5, device);

        let total_a = anchor_points.dims()[0];

        // Decode predicted boxes
        let pred_bboxes = self.box_decode::<B>(&pred_dist, &anchor_points, &strides, device);

        let pred_scores_t = pred_scores.swap_dims(1, 2); // [B, A, nc]

        // Convert targets to pixel coords
        let target_box_xyxy = anchors::wh2xy(target_box);
        let target_box_pixel = target_box_xyxy * (img_size as f32);

        // Per-image loss computation
        let mut total_box_loss = Tensor::<B, 1>::zeros([1], device);
        let mut total_cls_loss = Tensor::<B, 1>::zeros([1], device);
        let mut total_dfl_loss = Tensor::<B, 1>::zeros([1], device);
        let mut total_fg = 0usize;

        let target_idx_vec: Vec<f32> = target_idx.to_data().to_vec().unwrap();

        // Handle empty targets (dummy idx=-1 from collate): return zero losses
        if target_idx_vec.is_empty() || (target_idx_vec.len() == 1 && target_idx_vec[0] < 0.0) {
            let zero = Tensor::<B, 1>::zeros([1], device);
            return (zero.clone(), zero.clone(), zero);
        }

        let target_cls_vec: Vec<f32> = target_cls.clone().squeeze::<1>(1).to_data().to_vec().unwrap();
        let target_box_data: Vec<f32> = target_box_pixel.to_data().to_vec().unwrap();
        let num_targets = target_idx_vec.len();
        let target_box_vec: Vec<[f32; 4]> = (0..num_targets)
            .map(|i| {
                [
                    target_box_data[i * 4],
                    target_box_data[i * 4 + 1],
                    target_box_data[i * 4 + 2],
                    target_box_data[i * 4 + 3],
                ]
            })
            .collect();

        let anc_data: Vec<f32> = anchor_points.to_data().to_vec().unwrap();
        let anc_vec: Vec<[f32; 2]> = (0..total_a)
            .map(|i| [anc_data[i * 2], anc_data[i * 2 + 1]])
            .collect();

        for b in 0..batch_size {
            // Get targets for this image
            let mut gt_labels = Vec::new();
            let mut gt_bboxes = Vec::new();
            for t in 0..num_targets {
                if target_idx_vec[t] as usize == b {
                    gt_labels.push(target_cls_vec[t] as i64);
                    gt_bboxes.push(target_box_vec[t]);
                }
            }

            // Extract per-image predictions
            let pb = pred_bboxes.clone().narrow(0, b, 1).squeeze::<2>(0); // [A, 4]
            let ps = pred_scores_t.clone().narrow(0, b, 1).squeeze::<2>(0); // [A, nc]
            let ps_sigmoid = sigmoid(ps.clone());

            let ps_data: Vec<f32> = ps_sigmoid.to_data().to_vec().unwrap();
            let ps_vec: Vec<Vec<f32>> = (0..total_a)
                .map(|i| {
                    let start = i * self.nc;
                    ps_data[start..start + self.nc].to_vec()
                })
                .collect();

            let pb_data: Vec<f32> = pb.to_data().to_vec().unwrap();
            let pb_vec: Vec<[f32; 4]> = (0..total_a)
                .map(|i| {
                    [
                        pb_data[i * 4],
                        pb_data[i * 4 + 1],
                        pb_data[i * 4 + 2],
                        pb_data[i * 4 + 3],
                    ]
                })
                .collect();

            let assign = task_aligned_assign(&ps_vec, &pb_vec, &anc_vec, &gt_labels, &gt_bboxes, 10);

            let fg_count: usize = assign.fg_mask.iter().filter(|&&v| v).count();
            total_fg += fg_count;

            if fg_count == 0 {
                let cls_target = Tensor::<B, 2>::zeros([total_a, self.nc], device);
                let cls_loss = bce_with_logits(&ps, &cls_target).sum_dim(1).sum_dim(0).squeeze(0);
                total_cls_loss = total_cls_loss + cls_loss;
                continue;
            }

            // Classification target
            let mut cls_target_data = vec![0.0f32; total_a * self.nc];
            for (a_idx, fg) in assign.fg_mask.iter().enumerate() {
                if *fg {
                    let cls = assign.target_labels[a_idx] as usize;
                    if cls < self.nc {
                        cls_target_data[a_idx * self.nc + cls] = assign.target_scores[a_idx];
                    }
                }
            }
            let cls_target = Tensor::<B, 1>::from_floats(cls_target_data.as_slice(), device)
                .reshape([total_a, self.nc]);
            let cls_loss = bce_with_logits(&ps, &cls_target).sum_dim(1).sum_dim(0).squeeze(0);
            total_cls_loss = total_cls_loss + cls_loss;

            // Box loss (CIoU) on foreground
            let fg_indices: Vec<usize> = assign
                .fg_mask
                .iter()
                .enumerate()
                .filter(|(_, &v)| v)
                .map(|(i, _)| i)
                .collect();

            let fg_idx_tensor = Tensor::<B, 1, Int>::from_ints(
                fg_indices.iter().map(|&i| i as i32).collect::<Vec<_>>().as_slice(),
                device,
            );
            let fg_pred_boxes = pb.select(0, fg_idx_tensor.clone());

            let fg_target_boxes_data: Vec<f32> = fg_indices
                .iter()
                .flat_map(|&a| assign.target_bboxes[a].iter().copied())
                .collect();
            let fg_target_boxes =
                Tensor::<B, 1>::from_floats(fg_target_boxes_data.as_slice(), device)
                    .reshape([fg_count, 4]);

            let (ciou, _iou) = compute_ciou(&fg_pred_boxes, &fg_target_boxes);
            let box_loss = ((-ciou) + 1.0).mean();
            total_box_loss = total_box_loss + box_loss;

            // DFL loss on foreground
            let pred_dist_b = pred_dist.clone().narrow(0, b, 1).squeeze::<2>(0); // [4*ch, A]
            let pred_dist_b = pred_dist_b.swap_dims(0, 1); // [A, 4*ch]
            let fg_pred_dist = pred_dist_b.select(0, fg_idx_tensor.clone());

            // Target for DFL
            let strides_data: Vec<f32> = strides.to_data().to_vec().unwrap();
            let fg_anc_data: Vec<f32> = fg_indices
                .iter()
                .flat_map(|&a| [anc_vec[a][0], anc_vec[a][1]])
                .collect();
            let fg_anc = Tensor::<B, 1>::from_floats(fg_anc_data.as_slice(), device)
                .reshape([fg_count, 2]);

            let fg_strides_data: Vec<f32> = fg_indices
                .iter()
                .map(|&a| strides_data[a])
                .collect();
            let fg_strides = Tensor::<B, 1>::from_floats(fg_strides_data.as_slice(), device)
                .reshape([fg_count, 1]);

            let fg_tb_xy1 = fg_target_boxes.clone().narrow(1, 0, 2);
            let fg_tb_xy2 = fg_target_boxes.narrow(1, 2, 2);
            let lt = fg_anc.clone() - fg_tb_xy1 / fg_strides.clone();
            let rb = fg_tb_xy2 / fg_strides - fg_anc;
            let dfl_target = Tensor::cat(vec![lt, rb], 1); // [fg, 4]
            let dfl_target = dfl_target.clamp(0.0, (self.ch - 1) as f64);

            let dfl_l = df_loss::<B>(&fg_pred_dist, &dfl_target, self.ch, device);
            let dfl_loss = dfl_l.mean();
            total_dfl_loss = total_dfl_loss + dfl_loss;
        }

        // Normalize by total foreground count
        let norm = 1.0 / total_fg.max(1) as f64;
        let loss_box = total_box_loss * (norm * self.box_gain);
        let loss_cls = total_cls_loss * (norm * self.cls_gain);
        let loss_dfl = total_dfl_loss * (norm * self.dfl_gain);

        (loss_box, loss_cls, loss_dfl)
    }

    /// Decode DFL distribution to box coordinates.
    fn box_decode<B: Backend>(
        &self,
        pred_dist: &Tensor<B, 3>,
        anchor_points: &Tensor<B, 2>,
        strides: &Tensor<B, 2>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, _, a] = pred_dist.dims();

        let pd = pred_dist.clone().reshape([b, 4, self.ch, a]);
        let pd = pd.swap_dims(2, 3); // [B, 4, A, ch]
        let pd = burn::tensor::activation::softmax(pd, 3);
        let bins: Vec<f32> = (0..self.ch).map(|i| i as f32).collect();
        let bins = Tensor::<B, 1>::from_floats(bins.as_slice(), device).reshape([1, 1, 1, self.ch]);
        let pd = pd * bins; // [B, 4, A, ch] * [1, 1, 1, ch]
        let pd = pd.sum_dim(3).squeeze::<3>(3); // [B, 4, A]
        let pd = pd.swap_dims(1, 2); // [B, A, 4]

        // Split into lt, rb
        let lt = pd.clone().narrow(2, 0, 2); // [B, A, 2]
        let rb = pd.narrow(2, 2, 2);          // [B, A, 2]

        let anc = anchor_points.clone().unsqueeze_dim(0); // [1, A, 2]
        let x1y1 = anc.clone() - lt;
        let x2y2 = anc + rb;
        let boxes = Tensor::cat(vec![x1y1, x2y2], 2); // [B, A, 4]
        let st = strides.clone().unsqueeze_dim(0); // [1, A, 1]
        boxes * st
    }
}
