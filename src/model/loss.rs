use burn::prelude::*;
use burn::tensor::activation::{log_sigmoid, log_softmax, sigmoid};

use crate::model::anchors;

// ---------------------------------------------------------------------------
// CIoU loss (Complete IoU)
// ---------------------------------------------------------------------------

/// Compute CIoU between pred [N, 4] and target [N, 4] boxes in xyxy format.
/// Returns (ciou, iou) both [N].
pub fn compute_iou(pred: &Tensor<2>, target: &Tensor<2>) -> (Tensor<1>, Tensor<1>) {
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

    // Python adds eps to heights here before area/ratio computations.
    let p_w = p_x2.clone() - p_x1.clone();
    let p_h = p_y2.clone() - p_y1.clone() + 1e-7;
    let t_w = t_x2.clone() - t_x1.clone();
    let t_h = t_y2.clone() - t_y1.clone() + 1e-7;
    let area_p = p_w.clone() * p_h.clone();
    let area_t = t_w.clone() * t_h.clone();
    let union = area_p + area_t - inter.clone() + 1e-7;
    let iou = inter / union;

    // Enclosing box
    let enc_x1 = p_x1.clone().min_pair(t_x1.clone());
    let enc_y1 = p_y1.clone().min_pair(t_y1.clone());
    let enc_x2 = p_x2.clone().max_pair(t_x2.clone());
    let enc_y2 = p_y2.clone().max_pair(t_y2.clone());
    let enc_w = enc_x2 - enc_x1;
    let enc_h = enc_y2 - enc_y1;
    let c2 = enc_w.clone().powf_scalar(2.0) + enc_h.powf_scalar(2.0) + 1e-7;

    // Center distance
    let rho2 = ((t_x1.clone() + t_x2.clone() - p_x1.clone() - p_x2.clone()).powf_scalar(2.0)
        + (t_y1.clone() + t_y2.clone() - p_y1.clone() - p_y2.clone()).powf_scalar(2.0))
        / 4.0;

    let v = ((t_w / t_h).atan() - (p_w / p_h).atan()).powf_scalar(2.0)
        * (4.0 / std::f64::consts::PI.powi(2));
    // The reference computes CIoU's alpha under `torch.no_grad()`. Letting
    // gradients flow through this ratio can explode when its denominator is
    // small, even though the forward loss is identical.
    let alpha = (v.clone() / (v.clone() - iou.clone() + (1.0 + 1e-7))).detach();
    let ciou = iou.clone() - (rho2 / c2 + v * alpha);

    let n = pred.dims()[0];
    let iou_flat = iou.reshape([n]);
    let ciou_flat = ciou.reshape([n]);
    (ciou_flat, iou_flat)
}

// ---------------------------------------------------------------------------
// DFL loss  (distribution focal loss for box regression)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Assigner (task-aligned, runs on CPU with no gradient)
// ---------------------------------------------------------------------------

pub struct Assigner {
    top_k: usize,
    nc: usize,
    alpha: f32,
    beta: f32,
    eps: f32,
}

impl Assigner {
    pub fn new(nc: usize, top_k: usize, alpha: f32, beta: f32, eps: f32) -> Self {
        Self {
            top_k,
            nc,
            alpha,
            beta,
            eps,
        }
    }

    pub fn forward(
        &self,
        pd_scores: &[Vec<f32>],
        pd_bboxes: &[[f32; 4]],
        anc_points: &[[f32; 2]],
        gt_labels: &[i64],
        gt_bboxes: &[[f32; 4]],
    ) -> (Vec<[f32; 4]>, Vec<Vec<f32>>, Vec<bool>) {
        let top_k = self.top_k;
        let alpha = self.alpha;
        let beta = self.beta;
        let eps = self.eps;
        let num_anchors = pd_scores.len();
        let num_gt = gt_labels.len();

        if num_gt == 0 {
            return (
                vec![[0.0; 4]; num_anchors],
                vec![vec![0.0; self.nc]; num_anchors],
                vec![false; num_anchors],
            );
        }
        assert!(top_k <= num_anchors, "selected index k out of range");

        // Check which anchors are strictly inside which GT boxes, matching
        // `amin(...).gt_(eps)` in the Python assigner.
        let mut is_inside = vec![vec![false; num_gt]; num_anchors];
        for a in 0..num_anchors {
            let (ax, ay) = (anc_points[a][0], anc_points[a][1]);
            for g in 0..num_gt {
                let b = &gt_bboxes[g];
                is_inside[a][g] = (ax - b[0]).min(ay - b[1]).min(b[2] - ax).min(b[3] - ay) > eps;
            }
        }

        // Compute alignment metric
        let mut overlaps = vec![vec![0.0f32; num_gt]; num_anchors];
        let mut metrics = vec![vec![0.0f32; num_gt]; num_anchors];
        for a in 0..num_anchors {
            for g in 0..num_gt {
                if !is_inside[a][g] {
                    continue;
                }
                let cls = gt_labels[g] as usize;
                let score = pd_scores[a][cls];
                let inter_x1 = gt_bboxes[g][0].max(pd_bboxes[a][0]);
                let inter_y1 = gt_bboxes[g][1].max(pd_bboxes[a][1]);
                let inter_x2 = gt_bboxes[g][2].min(pd_bboxes[a][2]);
                let inter_y2 = gt_bboxes[g][3].min(pd_bboxes[a][3]);
                let inter_w = (inter_x2 - inter_x1).max(0.0);
                let inter_h = (inter_y2 - inter_y1).max(0.0);
                let inter = inter_w * inter_h;
                let gt_w = gt_bboxes[g][2] - gt_bboxes[g][0];
                let gt_h = gt_bboxes[g][3] - gt_bboxes[g][1] + 1e-7;
                let pd_w = pd_bboxes[a][2] - pd_bboxes[a][0];
                let pd_h = pd_bboxes[a][3] - pd_bboxes[a][1] + 1e-7;
                let union = gt_w * gt_h + pd_w * pd_h - inter + 1e-7;
                let iou = inter / union;
                let cw =
                    gt_bboxes[g][2].max(pd_bboxes[a][2]) - gt_bboxes[g][0].min(pd_bboxes[a][0]);
                let ch =
                    gt_bboxes[g][3].max(pd_bboxes[a][3]) - gt_bboxes[g][1].min(pd_bboxes[a][1]);
                let c2 = cw.powi(2) + ch.powi(2) + 1e-7;
                let rho2 =
                    ((pd_bboxes[a][0] + pd_bboxes[a][2] - gt_bboxes[g][0] - gt_bboxes[g][2])
                        .powi(2)
                        + (pd_bboxes[a][1] + pd_bboxes[a][3] - gt_bboxes[g][1] - gt_bboxes[g][3])
                            .powi(2))
                        / 4.0;
                let v = (4.0 / std::f32::consts::PI.powi(2))
                    * ((pd_w / pd_h).atan() - (gt_w / gt_h).atan()).powi(2);
                let alpha_ciou = v / (v - iou + (1.0 + 1e-7));
                let ciou = iou - (rho2 / c2 + v * alpha_ciou);
                let overlap = ciou.max(0.0);
                overlaps[a][g] = overlap;
                metrics[a][g] = score.powf(alpha) * overlap.powf(beta);
            }
        }

        // Top-k selection per GT.
        let mut mask_pos = vec![vec![false; num_gt]; num_anchors];
        for g in 0..num_gt {
            let mut candidates: Vec<(f32, usize)> = Vec::with_capacity(num_anchors);
            for a in 0..num_anchors {
                candidates.push((metrics[a][g], a));
            }
            candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
            for i in 0..top_k {
                let (_, a) = candidates[i];
                if is_inside[a][g] {
                    mask_pos[a][g] = true;
                }
            }
        }

        // If an anchor is assigned to multiple GT boxes, keep the GT with the
        // largest overlap, matching Python's `overlaps.argmax(1)` resolution.
        let mut assigned_gt = vec![-1i64; num_anchors];
        let mut assigned_metric = vec![0.0f32; num_anchors];
        for a in 0..num_anchors {
            let mut selected: Vec<usize> = (0..num_gt).filter(|&g| mask_pos[a][g]).collect();
            if selected.len() > 1 {
                selected.sort_by(|&g1, &g2| overlaps[a][g2].total_cmp(&overlaps[a][g1]));
                selected.truncate(1);
            }
            if let Some(&g) = selected.first() {
                assigned_gt[a] = g as i64;
                assigned_metric[a] = metrics[a][g];
            }
        }

        // Build outputs
        let mut target_labels = vec![-1i64; num_anchors];
        let mut target_bboxes = vec![[0.0f32; 4]; num_anchors];
        let mut target_scores = vec![vec![0.0f32; self.nc]; num_anchors];
        let mut fg_mask = vec![false; num_anchors];

        for a in 0..num_anchors {
            if assigned_gt[a] >= 0 {
                let g = assigned_gt[a] as usize;
                target_labels[a] = gt_labels[g];
                target_bboxes[a] = gt_bboxes[g];
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
                        let cls = target_labels[a].max(0) as usize;
                        target_scores[a][cls] = (assigned_metric[a] / max_metric) * overlaps[a][g];
                    }
                }
            }
        }

        (target_bboxes, target_scores, fg_mask)
    }
}

/// Batched task-aligned assignment on the active tensor backend. Assignment
/// does not require gradients, but keeping it on the device avoids downloading
/// every anchor prediction and stalling the training stream once per image.
struct TensorAssigner {
    top_k: usize,
    nc: usize,
    alpha: f64,
    beta: f64,
    eps: f64,
}

impl TensorAssigner {
    fn new(nc: usize, top_k: usize, alpha: f64, beta: f64, eps: f64) -> Self {
        Self {
            top_k,
            nc,
            alpha,
            beta,
            eps,
        }
    }

    fn ciou(gt: Tensor<4>, pred: Tensor<4>) -> Tensor<3> {
        let gx1 = gt.clone().narrow(3, 0, 1);
        let gy1 = gt.clone().narrow(3, 1, 1);
        let gx2 = gt.clone().narrow(3, 2, 1);
        let gy2 = gt.narrow(3, 3, 1);
        let px1 = pred.clone().narrow(3, 0, 1);
        let py1 = pred.clone().narrow(3, 1, 1);
        let px2 = pred.clone().narrow(3, 2, 1);
        let py2 = pred.narrow(3, 3, 1);

        let inter = (gx2.clone().min_pair(px2.clone()) - gx1.clone().max_pair(px1.clone()))
            .clamp_min(0.0)
            * (gy2.clone().min_pair(py2.clone()) - gy1.clone().max_pair(py1.clone()))
                .clamp_min(0.0);
        let gw = gx2.clone() - gx1.clone();
        let gh = gy2.clone() - gy1.clone() + 1e-7;
        let pw = px2.clone() - px1.clone();
        let ph = py2.clone() - py1.clone() + 1e-7;
        let union = gw.clone() * gh.clone() + pw.clone() * ph.clone() - inter.clone() + 1e-7;
        let iou = inter / union;
        let cw = gx2.clone().max_pair(px2.clone()) - gx1.clone().min_pair(px1.clone());
        let ch = gy2.clone().max_pair(py2.clone()) - gy1.clone().min_pair(py1.clone());
        let c2 = cw.powf_scalar(2.0) + ch.powf_scalar(2.0) + 1e-7;
        let rho2 = ((px1.clone() + px2.clone() - gx1.clone() - gx2.clone()).powf_scalar(2.0)
            + (py1.clone() + py2.clone() - gy1.clone() - gy2.clone()).powf_scalar(2.0))
            / 4.0;
        let v = ((pw / ph).atan() - (gw / gh).atan()).powf_scalar(2.0)
            * (4.0 / std::f64::consts::PI.powi(2));
        let alpha = v.clone() / (v.clone() - iou.clone() + (1.0 + 1e-7));
        (iou - (rho2 / c2 + v * alpha))
            .clamp_min(0.0)
            .squeeze_dim::<3>(3)
    }

    fn forward(
        &self,
        pd_scores: Tensor<3>,
        pd_bboxes: Tensor<3>,
        anc_points: Tensor<2>,
        gt_labels: Tensor<2, Int>,
        gt_bboxes: Tensor<3>,
        mask_gt: Tensor<2, Bool>,
    ) -> (Tensor<3>, Tensor<3>, Tensor<2, Bool>) {
        let [batch_size, num_anchors, _] = pd_scores.dims();
        let num_gt = gt_bboxes.dims()[1];
        let device = pd_scores.device();

        let anc = anc_points.reshape([1, 1, num_anchors, 2]);
        let lt = gt_bboxes.clone().narrow(2, 0, 2).unsqueeze_dim::<4>(2);
        let rb = gt_bboxes.clone().narrow(2, 2, 2).unsqueeze_dim::<4>(2);
        let inside = Tensor::cat(vec![anc.clone() - lt, rb - anc], 3)
            .min_dim(3)
            .squeeze_dim::<3>(3)
            .greater_elem(self.eps);
        let valid = inside.bool_and(
            mask_gt
                .clone()
                .unsqueeze_dim::<3>(2)
                .repeat_dim(2, num_anchors),
        );

        let label_indices = gt_labels
            .clone()
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, num_anchors);
        let bbox_scores =
            pd_scores.detach().swap_dims(1, 2).gather(1, label_indices) * valid.clone().float();
        let overlaps = Self::ciou(
            gt_bboxes.clone().unsqueeze_dim::<4>(2),
            pd_bboxes.detach().unsqueeze_dim::<4>(1),
        ) * valid.clone().float();
        let align_metric =
            bbox_scores.powf_scalar(self.alpha) * overlaps.clone().powf_scalar(self.beta);

        let k = self.top_k.min(num_anchors);
        let (_, top_indices) = align_metric.clone().topk_with_indices(k, 2);
        let top_values = mask_gt
            .clone()
            .float()
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, k);
        let mask_top_k = Tensor::<3>::zeros([batch_size, num_gt, num_anchors], &device).scatter(
            2,
            top_indices,
            top_values,
            burn::tensor::IndexingUpdateOp::Add,
        );
        let mut mask_pos = mask_top_k * valid.clone().float();

        let assignment_profile = std::env::var_os("YOLOV11_ASSIGNMENT_PROFILE").is_some();
        let valid_per_gt =
            assignment_profile.then(|| valid.clone().float().sum_dim(2).squeeze_dim::<2>(2));
        let selected_before_conflict =
            assignment_profile.then(|| mask_pos.clone().sum_dim(2).squeeze_dim::<2>(2));

        let fg_count = mask_pos.clone().sum_dim(1).squeeze_dim::<2>(1);
        let multi = fg_count
            .clone()
            .greater_elem(1.0)
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, num_gt);
        let max_overlap_idx = overlaps.clone().argmax(1);
        let is_max = Tensor::<3>::zeros([batch_size, num_gt, num_anchors], &device).scatter(
            1,
            max_overlap_idx,
            Tensor::ones([batch_size, 1, num_anchors], &device),
            burn::tensor::IndexingUpdateOp::Add,
        );
        mask_pos = mask_pos.mask_where(multi, is_max);
        if assignment_profile {
            let valid_gt = mask_gt.clone().to_data().try_to_vec::<bool>().unwrap();
            let valid_counts = valid_per_gt.unwrap().to_data().try_to_vec::<f32>().unwrap();
            let pre_counts = selected_before_conflict
                .unwrap()
                .to_data()
                .try_to_vec::<f32>()
                .unwrap();
            let post_counts = mask_pos
                .clone()
                .sum_dim(2)
                .squeeze_dim::<2>(2)
                .to_data()
                .try_to_vec::<f32>()
                .unwrap();
            let contested = fg_count
                .greater_elem(1.0)
                .float()
                .sum()
                .to_data()
                .try_to_vec::<f32>()
                .unwrap()[0] as usize;
            let mut ground_truth = 0usize;
            let mut zero_valid = 0usize;
            let mut zero_selected = 0usize;
            let mut valid_sum = 0.0f32;
            let mut pre_sum = 0.0f32;
            let mut post_sum = 0.0f32;
            for index in 0..valid_gt.len() {
                if !valid_gt[index] {
                    continue;
                }
                ground_truth += 1;
                zero_valid += usize::from(valid_counts[index] == 0.0);
                zero_selected += usize::from(post_counts[index] == 0.0);
                valid_sum += valid_counts[index];
                pre_sum += pre_counts[index];
                post_sum += post_counts[index];
            }
            let denominator = ground_truth.max(1) as f32;
            eprintln!(
                "YOLOV11_ASSIGNMENT_PROFILE gt={ground_truth} zero_valid={zero_valid} \
                 zero_selected={zero_selected} valid_mean={:.3} pre_mean={:.3} \
                 post_mean={:.3} contested_anchors={contested}",
                valid_sum / denominator,
                pre_sum / denominator,
                post_sum / denominator,
            );
        }
        let fg_mask = mask_pos
            .clone()
            .sum_dim(1)
            .squeeze_dim::<2>(1)
            .greater_elem(0.0);
        let target_gt_idx = mask_pos.clone().argmax(1).squeeze_dim::<2>(1);

        let offsets = Tensor::<1, Int>::from_ints(
            (0..batch_size)
                .map(|batch| (batch * num_gt) as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([batch_size, 1])
        .repeat_dim(1, num_anchors);
        let flat_indices = (target_gt_idx + offsets).reshape([batch_size * num_anchors]);
        let target_bboxes = gt_bboxes
            .reshape([batch_size * num_gt, 4])
            .select(0, flat_indices.clone())
            .reshape([batch_size, num_anchors, 4]);
        let target_labels = gt_labels
            .reshape([batch_size * num_gt])
            .select(0, flat_indices)
            .reshape([batch_size, num_anchors]);
        let target_scores = Tensor::<3>::zeros([batch_size, num_anchors, self.nc], &device)
            .scatter(
                2,
                target_labels.unsqueeze_dim::<3>(2),
                fg_mask.clone().float().unsqueeze_dim::<3>(2),
                burn::tensor::IndexingUpdateOp::Add,
            );

        let masked_metric = align_metric * mask_pos.clone();
        let pos_align = masked_metric.clone().max_dim(2);
        let pos_overlap = (overlaps * mask_pos).max_dim(2);
        let norm = (masked_metric * pos_overlap / (pos_align + self.eps))
            .max_dim(1)
            .swap_dims(1, 2);
        (target_bboxes, target_scores * norm, fg_mask)
    }
}

pub struct BoxLoss {
    dfl_ch: usize,
}

impl BoxLoss {
    pub fn new(dfl_ch: usize) -> Self {
        Self { dfl_ch }
    }

    pub fn forward(
        &self,
        pred_dist: &Tensor<2>,
        pred_bboxes: &Tensor<2>,
        anchor_points: &Tensor<2>,
        target_bboxes: &Tensor<2>,
        target_scores: &Tensor<2>,
        target_scores_sum: f32,
        fg_mask: &[bool],
        device: &Device,
    ) -> (Tensor<1>, Tensor<1>) {
        let fg_indices: Vec<usize> = fg_mask
            .iter()
            .enumerate()
            .filter(|(_, &v)| v)
            .map(|(i, _)| i)
            .collect();

        if fg_indices.is_empty() {
            let zero = Tensor::<1>::zeros([1], device);
            return (zero.clone(), zero);
        }

        let fg_count = fg_indices.len();
        let fg_idx_tensor = Tensor::<1, Int>::from_ints(
            fg_indices
                .iter()
                .map(|&i| i as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );

        let pred_bboxes = pred_bboxes.clone().select(0, fg_idx_tensor.clone());
        let target_bboxes_fg = target_bboxes.clone().select(0, fg_idx_tensor.clone());
        let weight = target_scores
            .clone()
            .sum_dim(1)
            .select(0, fg_idx_tensor.clone())
            .reshape([fg_count, 1]);
        let target_scores_sum = target_scores_sum as f64;

        let (ciou, _iou) = compute_iou(&pred_bboxes, &target_bboxes_fg);
        let loss_box = ((ciou.neg() + 1.0).reshape([fg_count, 1]) * weight.clone())
            .sum_dim(0)
            .reshape([1])
            / target_scores_sum;

        let pred_dist = pred_dist.clone().select(0, fg_idx_tensor.clone());
        let anchor_points = anchor_points.clone().select(0, fg_idx_tensor);
        let a = target_bboxes_fg.clone().narrow(1, 0, 2);
        let b = target_bboxes_fg.narrow(1, 2, 2);
        let target = Tensor::cat(vec![anchor_points.clone() - a, b - anchor_points], 1)
            .clamp(0.0, self.dfl_ch as f64 - 0.01);
        let loss_dfl = (Self::df_loss(&pred_dist, &target, self.dfl_ch + 1, device)
            .reshape([fg_count, 1])
            * weight)
            .sum_dim(0)
            .reshape([1])
            / target_scores_sum;

        (loss_box, loss_dfl)
    }

    pub fn df_loss(
        pred_dist: &Tensor<2>,
        target: &Tensor<2>,
        ch: usize,
        _device: &Device,
    ) -> Tensor<1> {
        let n = pred_dist.dims()[0];
        let target = target.clone().reshape([n * 4]);
        let target_left = target.clone().floor();
        let weight_right = target - target_left.clone();
        let weight_left = weight_right.clone().neg() + 1.0;
        let left_indices = target_left.clone().int().reshape([n * 4, 1]);
        let right_indices = (target_left + 1.0).int().reshape([n * 4, 1]);
        let pd = pred_dist.clone().reshape([n * 4, ch]);
        let log_sm = log_softmax(pd, 1);
        let loss = (log_sm
            .clone()
            .gather(1, left_indices)
            .reshape([n * 4])
            .neg()
            * weight_left
            + log_sm.gather(1, right_indices).reshape([n * 4]).neg() * weight_right)
            .reshape([n, 4]);
        loss.mean_dim(1).reshape([n])
    }
}

pub struct QFL {
    beta: f64,
}

impl QFL {
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }

    pub fn forward(&self, outputs: &Tensor<2>, targets: &Tensor<2>) -> Tensor<2> {
        let bce_loss =
            (targets.clone().neg() + 1.0) * outputs.clone() - log_sigmoid(outputs.clone());
        (targets.clone() - sigmoid(outputs.clone()))
            .abs()
            .powf_scalar(self.beta)
            * bce_loss
    }
}

pub struct VFL {
    alpha: f64,
    gamma: f64,
    iou_weighted: bool,
}

impl VFL {
    pub fn new(alpha: f64, gamma: f64, iou_weighted: bool) -> Self {
        Self {
            alpha,
            gamma,
            iou_weighted,
        }
    }

    pub fn forward(&self, outputs: &Tensor<2>, targets: &Tensor<2>) -> Tensor<2> {
        let loss = (targets.clone().neg() + 1.0) * outputs.clone() - log_sigmoid(outputs.clone());
        let distance = (sigmoid(outputs.clone()) - targets.clone())
            .abs()
            .powf_scalar(self.gamma);
        let negative_weight = distance * self.alpha;
        if self.iou_weighted {
            let focal_weight =
                negative_weight.mask_where(targets.clone().greater_elem(0.0), targets.clone());
            loss * focal_weight
        } else {
            let positive_weight = targets.clone() * 0.0 + 1.0;
            let focal_weight =
                negative_weight.mask_where(targets.clone().greater_elem(0.0), positive_weight);
            loss * focal_weight
        }
    }
}

pub struct FocalLoss {
    alpha: f64,
    gamma: f64,
}

impl FocalLoss {
    pub fn new(alpha: f64, gamma: f64) -> Self {
        Self { alpha, gamma }
    }

    pub fn forward(&self, outputs: &Tensor<2>, targets: &Tensor<2>) -> Tensor<2> {
        let mut loss =
            (targets.clone().neg() + 1.0) * outputs.clone() - log_sigmoid(outputs.clone());
        if self.alpha > 0.0 {
            let alpha_factor =
                targets.clone() * self.alpha + (1.0 - targets.clone()) * (1.0 - self.alpha);
            loss = loss * alpha_factor;
        }
        if self.gamma > 0.0 {
            let outputs_sigmoid = sigmoid(outputs.clone());
            let p_t: Tensor<2> = targets.clone() * outputs_sigmoid.clone()
                + (targets.clone().neg() + 1.0) * (outputs_sigmoid.neg() + 1.0);
            loss = loss * (p_t.neg() + 1.0).powf_scalar(self.gamma);
        }
        loss
    }
}

// ---------------------------------------------------------------------------
// ComputeLoss — main training loss
// ---------------------------------------------------------------------------

pub struct ComputeLoss {
    nc: usize,
    ch: usize,
    nl: usize,
    stride: Vec<f32>,
    box_gain: f64,
    cls_gain: f64,
    dfl_gain: f64,
}

impl ComputeLoss {
    pub fn new(
        nc: usize,
        nl: usize,
        stride: &Tensor<1>,
        box_gain: f64,
        cls_gain: f64,
        dfl_gain: f64,
    ) -> Self {
        let stride_vec: Vec<f32> = stride.to_data().try_to_vec::<f32>().unwrap();
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

    pub fn call(
        &self,
        outputs: &[Tensor<4>],
        target_cls: &Tensor<2>, // [num_targets, 1]
        target_box: &Tensor<2>, // [num_targets, 4]  normalized xywh
        target_idx: &Tensor<1>, // [num_targets]  batch index
        batch_size: usize,
        img_size: usize,
        device: &Device,
    ) -> (Tensor<1>, Tensor<1>, Tensor<1>) {
        let no = self.nc + self.ch * 4;

        // Flatten outputs: [B, no, total_A]
        let mut flat: Vec<Tensor<3>> = Vec::new();
        for o in outputs {
            let [b, _, h, w] = o.dims();
            flat.push(o.clone().reshape([b, no, h * w]));
        }
        let x = Tensor::cat(flat, 2); // [B, no, total_A]

        let box_ch = self.ch * 4;
        let pred_dist = x.clone().narrow(1, 0, box_ch); // [B, 4*ch, A]
        let pred_scores = x.narrow(1, box_ch, self.nc); // [B, nc, A]

        // Generate anchors
        let stride_t = Tensor::<1>::from_floats(self.stride.as_slice(), device);
        let (anchor_points, strides) =
            crate::model::anchors::make_anchors(outputs, &stride_t, 0.5, device);

        let total_a = anchor_points.dims()[0];

        // Decode predicted boxes in anchor units, matching Python's box_decode.
        let pred_bboxes = self.box_decode(&pred_dist, &anchor_points, &strides, device);

        let pred_scores_t = pred_scores.swap_dims(1, 2); // [B, A, nc]

        // Convert targets to pixel coords
        let target_box_xyxy = anchors::wh2xy(target_box);
        let target_box_pixel = target_box_xyxy * (img_size as f32);

        let target_idx_vec: Vec<f32> = target_idx.to_data().try_to_vec::<f32>().unwrap();

        let target_cls_vec: Vec<f32> = target_cls
            .clone()
            .squeeze_dim::<1>(1)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let target_box_data: Vec<f32> = target_box_pixel.to_data().try_to_vec::<f32>().unwrap();
        let valid_target_indices: Vec<usize> = target_idx_vec
            .iter()
            .enumerate()
            .filter_map(|(i, &idx)| if idx >= 0.0 { Some(i) } else { None })
            .collect();
        let target_idx_vec: Vec<f32> = valid_target_indices
            .iter()
            .map(|&i| target_idx_vec[i])
            .collect();
        let target_cls_vec: Vec<f32> = valid_target_indices
            .iter()
            .map(|&i| target_cls_vec[i])
            .collect();
        let target_box_vec: Vec<[f32; 4]> = valid_target_indices
            .iter()
            .map(|&i| {
                [
                    target_box_data[i * 4],
                    target_box_data[i * 4 + 1],
                    target_box_data[i * 4 + 2],
                    target_box_data[i * 4 + 3],
                ]
            })
            .collect();

        // The assigner works in pixels: `gt_bboxes` are pixels and `pd_bboxes`
        // are multiplied by their stride below, so the anchor points must be
        // too. `make_anchors` returns them in grid units — 0..40 at stride 8 on
        // a 320 px image — and comparing those against pixel boxes makes
        // `is_inside` true only near the origin, so almost nothing is ever
        // assigned and the model learns to predict nothing. Python does the
        // same multiplication (`anchor_points * stride_tensor`) at the same
        // point.
        let mut per_batch = vec![Vec::<usize>::new(); batch_size];
        for (target, &batch) in target_idx_vec.iter().enumerate() {
            per_batch[batch as usize].push(target);
        }
        let max_gt = per_batch.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let mut padded_labels = vec![0i32; batch_size * max_gt];
        let mut padded_boxes = vec![0.0f32; batch_size * max_gt * 4];
        let mut padded_mask = vec![false; batch_size * max_gt];
        for (batch, targets) in per_batch.iter().enumerate() {
            for (slot, &target) in targets.iter().enumerate() {
                let output = batch * max_gt + slot;
                padded_labels[output] = target_cls_vec[target] as i32;
                padded_boxes[output * 4..output * 4 + 4].copy_from_slice(&target_box_vec[target]);
                padded_mask[output] = true;
            }
        }
        let gt_labels = Tensor::<1, Int>::from_ints(padded_labels.as_slice(), device)
            .reshape([batch_size, max_gt]);
        let gt_bboxes = Tensor::<1>::from_floats(padded_boxes.as_slice(), device)
            .reshape([batch_size, max_gt, 4]);
        let mask_gt = Tensor::<1, Bool>::from_bool(padded_mask.as_slice(), device)
            .reshape([batch_size, max_gt]);

        let anchor_points_pixel = anchor_points.clone() * strides.clone();
        let pred_bboxes_pixel = pred_bboxes.clone() * strides.clone().unsqueeze_dim::<3>(0);
        let (target_bboxes, target_scores, fg_mask) =
            TensorAssigner::new(self.nc, 10, 0.5, 6.0, 1e-9).forward(
                // The PyTorch task-aligned assigner runs under `no_grad`.
                // Assignment chooses fixed targets for this optimization step;
                // differentiating through its scores changes, and can reverse,
                // the classification and box gradients.
                sigmoid(pred_scores_t.clone()).detach(),
                pred_bboxes_pixel.detach(),
                anchor_points_pixel,
                gt_labels,
                gt_bboxes,
                mask_gt,
            );
        let target_scores_sum = target_scores.clone().sum().reshape([1]).clamp_min(1.0);

        let total_cls_loss = ((target_scores.clone().neg() + 1.0) * pred_scores_t.clone()
            - log_sigmoid(pred_scores_t.clone()))
        .sum()
        .reshape([1])
            / target_scores_sum.clone();

        let target_bboxes_grid = target_bboxes / strides.clone().unsqueeze_dim::<3>(0);
        let flat_fg = fg_mask.reshape([batch_size * total_a]);
        let fg_indices = flat_fg.nonzero().into_iter().next();
        let (total_box_loss, total_dfl_loss) = if let Some(fg_indices) = fg_indices {
            let fg_count = fg_indices.dims()[0];
            let weights = target_scores
                .clone()
                .sum_dim(2)
                .reshape([batch_size * total_a])
                .select(0, fg_indices.clone())
                .reshape([fg_count, 1]);
            let pred_bboxes_fg = pred_bboxes
                .clone()
                .reshape([batch_size * total_a, 4])
                .select(0, fg_indices.clone());
            let target_bboxes_fg = target_bboxes_grid
                .clone()
                .reshape([batch_size * total_a, 4])
                .select(0, fg_indices.clone());
            let (ciou, _) = compute_iou(&pred_bboxes_fg, &target_bboxes_fg);
            let box_loss = ((ciou.neg() + 1.0).reshape([fg_count, 1]) * weights.clone())
                .sum()
                .reshape([1])
                / target_scores_sum.clone();

            let anchors = anchor_points
                .reshape([1, total_a, 2])
                .repeat_dim(0, batch_size);
            let target_fg = Tensor::cat(
                vec![
                    anchors.clone() - target_bboxes_grid.clone().narrow(2, 0, 2),
                    target_bboxes_grid.narrow(2, 2, 2) - anchors,
                ],
                2,
            )
            .clamp(0.0, self.ch as f64 - 1.01)
            .reshape([batch_size * total_a, 4])
            .select(0, fg_indices.clone());
            let pred_dist_fg = pred_dist
                .swap_dims(1, 2)
                .reshape([batch_size * total_a, self.ch * 4])
                .select(0, fg_indices);
            let dfl =
                BoxLoss::df_loss(&pred_dist_fg, &target_fg, self.ch, device).reshape([fg_count, 1]);
            let dfl_loss = (dfl * weights).sum().reshape([1]) / target_scores_sum;
            (box_loss, dfl_loss)
        } else {
            let zero = Tensor::<1>::zeros([1], device);
            (zero.clone(), zero)
        };

        let loss_box = total_box_loss * self.box_gain;
        let loss_cls = total_cls_loss * self.cls_gain;
        let loss_dfl = total_dfl_loss * self.dfl_gain;

        (loss_box, loss_cls, loss_dfl)
    }

    /// Decode DFL distribution to box coordinates.
    pub fn box_decode(
        &self,
        pred_dist: &Tensor<3>,
        anchor_points: &Tensor<2>,
        _strides: &Tensor<2>,
        device: &Device,
    ) -> Tensor<3> {
        let [b, _, a] = pred_dist.dims();

        let pd = pred_dist.clone().reshape([b, 4, self.ch, a]);
        let pd = pd.swap_dims(2, 3); // [B, 4, A, ch]
        let pd = burn::tensor::activation::softmax(pd, 3);
        let bins: Vec<f32> = (0..self.ch).map(|i| i as f32).collect();
        let bins = Tensor::<1>::from_floats(bins.as_slice(), device).reshape([1, 1, 1, self.ch]);
        let pd = pd * bins; // [B, 4, A, ch] * [1, 1, 1, ch]
        let pd = pd.sum_dim(3).squeeze_dim::<3>(3); // [B, 4, A]
        let pd = pd.swap_dims(1, 2); // [B, A, 4]

        // Split into lt, rb
        let lt = pd.clone().narrow(2, 0, 2); // [B, A, 2]
        let rb = pd.narrow(2, 2, 2); // [B, A, 2]

        let anc = anchor_points.clone().unsqueeze_dim(0); // [1, A, 2]
        let x1y1 = anc.clone() - lt;
        let x2y2 = anc + rb;
        Tensor::cat(vec![x1y1, x2y2], 2) // [B, A, 4]
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_iou, ComputeLoss};
    use burn::prelude::*;

    fn ciou_scalar(box1: [f32; 4], box2: [f32; 4]) -> (f32, f32) {
        let eps = 1e-7f32;
        let (b1_x1, b1_y1, b1_x2, b1_y2) = (box1[0], box1[1], box1[2], box1[3]);
        let (b2_x1, b2_y1, b2_x2, b2_y2) = (box2[0], box2[1], box2[2], box2[3]);
        let w1 = b1_x2 - b1_x1;
        let h1 = b1_y2 - b1_y1 + eps;
        let w2 = b2_x2 - b2_x1;
        let h2 = b2_y2 - b2_y1 + eps;

        let inter = (b1_x2.min(b2_x2) - b1_x1.max(b2_x1)).max(0.0)
            * (b1_y2.min(b2_y2) - b1_y1.max(b2_y1)).max(0.0);
        let union = w1 * h1 + w2 * h2 - inter + eps;
        let iou = inter / union;
        let cw = b1_x2.max(b2_x2) - b1_x1.min(b2_x1);
        let ch = b1_y2.max(b2_y2) - b1_y1.min(b2_y1);
        let c2 = cw.powi(2) + ch.powi(2) + eps;
        let rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).powi(2)
            + (b2_y1 + b2_y2 - b1_y1 - b1_y2).powi(2))
            / 4.0;
        let v =
            (4.0 / std::f32::consts::PI.powi(2)) * ((w2 / h2).atan() - (w1 / h1).atan()).powi(2);
        let alpha = v / (v - iou + (1.0 + eps));
        (iou - (rho2 / c2 + v * alpha), iou)
    }

    #[test]
    fn compute_iou_matches_python_ciou_formula() {
        let device = burn::tensor::Device::flex();
        let pred = Tensor::<1>::from_floats([0.0, 0.0, 2.0, 3.0], &device).reshape([1, 4]);
        let target = Tensor::<1>::from_floats([0.5, 0.25, 2.5, 2.0], &device).reshape([1, 4]);

        let (ciou, iou) = compute_iou(&pred, &target);
        let ciou = ciou.to_data().try_to_vec::<f32>().unwrap()[0];
        let iou = iou.to_data().try_to_vec::<f32>().unwrap()[0];
        let (expected_ciou, expected_iou) =
            ciou_scalar([0.0, 0.0, 2.0, 3.0], [0.5, 0.25, 2.5, 2.0]);

        assert!((ciou - expected_ciou).abs() < 1e-5);
        assert!((iou - expected_iou).abs() < 1e-5);
    }

    #[test]
    fn assigner_excludes_boundary_anchor_points() {
        let assigner = super::Assigner::new(1, 1, 0.5, 6.0, 1e-9);
        let (_target_bboxes, _target_scores, fg_mask) = assigner.forward(
            &[vec![1.0]],
            &[[0.0, 0.0, 10.0, 10.0]],
            &[[0.0, 5.0]],
            &[0],
            &[[0.0, 0.0, 10.0, 10.0]],
        );

        assert_eq!(fg_mask, vec![false]);
    }

    #[test]
    fn assigner_resolves_multi_gt_anchor_by_overlap() {
        let assigner = super::Assigner::new(1, 1, 0.5, 6.0, 1e-9);
        let (target_bboxes, _target_scores, fg_mask) = assigner.forward(
            &[vec![1.0]],
            &[[4.0, 4.0, 8.0, 8.0]],
            &[[5.0, 5.0]],
            &[0, 0],
            &[[0.0, 0.0, 10.0, 10.0], [4.0, 4.0, 8.0, 8.0]],
        );

        assert_eq!(fg_mask, vec![true]);
        assert_eq!(target_bboxes[0], [4.0, 4.0, 8.0, 8.0]);
    }

    #[test]
    #[should_panic(expected = "selected index k out of range")]
    fn assigner_rejects_top_k_larger_than_anchor_dimension_like_python_topk() {
        let assigner = super::Assigner::new(1, 2, 0.5, 6.0, 1e-9);
        let _ = assigner.forward(
            &[vec![1.0]],
            &[[0.0, 0.0, 10.0, 10.0]],
            &[[5.0, 5.0]],
            &[0],
            &[[0.0, 0.0, 10.0, 10.0]],
        );
    }

    #[test]
    fn assigner_applies_top_k_before_inside_mask_like_python() {
        let assigner = super::Assigner::new(1, 1, 1.0, 1.0, 1e-9);
        let (_target_bboxes, target_scores, fg_mask) = assigner.forward(
            &[vec![1.0], vec![0.5]],
            &[[0.0, 0.0, 20.0, 20.0], [20.0, 20.0, 30.0, 30.0]],
            &[[20.0, 20.0], [5.0, 5.0]],
            &[0],
            &[[0.0, 0.0, 10.0, 10.0]],
        );

        assert_eq!(fg_mask, vec![false, false]);
        assert_eq!(target_scores, vec![vec![0.0], vec![0.0]]);
    }

    #[test]
    fn assigner_ranks_overlaps_by_python_ciou_not_plain_iou() {
        let assigner = super::Assigner::new(1, 1, 1.0, 1.0, 1e-9);
        let (_target_bboxes, target_scores, fg_mask) = assigner.forward(
            &[vec![1.0], vec![1.0]],
            &[[0.0, 0.0, 8.0, 10.0], [1.0, 0.0, 9.0, 10.0]],
            &[[5.0, 5.0], [5.0, 5.0]],
            &[0],
            &[[0.0, 0.0, 10.0, 10.0]],
        );

        assert_eq!(fg_mask, vec![false, true]);
        assert_eq!(target_scores[0], vec![0.0]);
        assert!((target_scores[1][0] - 0.7998798).abs() < 1e-6);
    }

    #[test]
    fn assigner_forward_matches_python_score_fixture() {
        let assigner = super::Assigner::new(2, 2, 0.5, 6.0, 1e-9);
        let (target_bboxes, target_scores, fg_mask) = assigner.forward(
            &[
                vec![0.8, 0.1],
                vec![0.6, 0.4],
                vec![0.2, 0.9],
                vec![0.3, 0.7],
            ],
            &[
                [0.0, 0.0, 4.0, 4.0],
                [1.0, 1.0, 5.0, 5.0],
                [5.0, 0.0, 9.0, 4.0],
                [6.0, 1.0, 10.0, 5.0],
            ],
            &[[2.0, 2.0], [3.0, 3.0], [7.0, 2.0], [8.0, 3.0]],
            &[0, 1],
            &[[0.0, 0.0, 5.0, 5.0], [5.0, 0.0, 10.0, 5.0]],
        );

        assert_eq!(fg_mask, vec![true, true, true, true]);
        assert_eq!(
            target_bboxes,
            vec![
                [0.0, 0.0, 5.0, 5.0],
                [0.0, 0.0, 5.0, 5.0],
                [5.0, 0.0, 10.0, 5.0],
                [5.0, 0.0, 10.0, 5.0],
            ]
        );

        let expected_scores = [
            [0.62999994, 0.0],
            [0.545596, 0.0],
            [0.0, 0.62999994],
            [0.0, 0.55560774],
        ];
        for (actual_row, expected_row) in target_scores.iter().zip(expected_scores) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                assert!((*actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn vfl_matches_python_positive_negative_weighting() {
        let device = burn::tensor::Device::flex();
        let outputs = Tensor::<1>::from_floats([0.0, 0.0, 0.0], &device).reshape([1, 3]);
        let targets = Tensor::<1>::from_floats([1.0, 0.0, 0.4], &device).reshape([1, 3]);
        let loss = super::VFL::new(0.75, 2.0, true).forward(&outputs, &targets);
        let loss = loss.to_data().try_to_vec::<f32>().unwrap();

        let bce = std::f32::consts::LN_2;
        let expected = [bce, bce * 0.75 * 0.5f32.powi(2), bce * 0.4];
        for (actual, expected) in loss.iter().zip(expected) {
            assert!((*actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn qfl_and_focal_loss_match_python_fixture() {
        let device = burn::tensor::Device::flex();
        let outputs =
            Tensor::<1>::from_floats([-2.0, 0.0, 1.5, 3.0, -1.0, 0.25], &device).reshape([2, 3]);
        let targets =
            Tensor::<1>::from_floats([0.0, 1.0, 0.4, 1.0, 0.0, 0.75], &device).reshape([2, 3]);

        let qfl = super::QFL::new(2.0)
            .forward(&outputs, &targets)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let focal = super::FocalLoss::new(0.25, 1.5)
            .forward(&outputs, &targets)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let bce = super::FocalLoss::new(0.0, 0.0)
            .forward(&outputs, &targets)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();

        let expected_qfl = [
            0.0018035639,
            0.1732868,
            0.1920517,
            0.00010928329,
            0.022658063,
            0.022522647,
        ];
        let expected_focal = [
            0.0039178636,
            0.061266132,
            0.25625426,
            0.00012545448,
            0.03276839,
            0.07687543,
        ];
        let expected_bce = [
            0.12692809,
            0.6931472,
            1.1014134,
            0.048587352,
            0.31326175,
            0.6384394,
        ];

        for (actual, expected) in qfl.iter().zip(expected_qfl) {
            assert!((*actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in focal.iter().zip(expected_focal) {
            assert!((*actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in bce.iter().zip(expected_bce) {
            assert!((*actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn box_loss_df_loss_matches_python_fixture() {
        let device = burn::tensor::Device::flex();
        let pred_dist = Tensor::<1>::from_floats(
            [
                0.2, -0.1, 0.4, 1.0, 1.2, 0.3, -0.7, 0.0, -0.5, 0.8, 0.1, -1.0, 0.0, 0.0, 0.0, 0.0,
                0.3, 0.1, -0.2, 0.5, 0.7, -0.4, 0.2, 0.0, -0.1, 0.2, 0.4, -0.8, 1.0, 0.0, -0.5,
                0.2,
            ],
            &device,
        )
        .reshape([2, 16]);
        let target = Tensor::<1>::from_floats([0.2, 1.7, 2.4, 2.99, 0.0, 0.5, 1.2, 2.8], &device)
            .reshape([2, 4]);

        let loss = super::BoxLoss::df_loss(&pred_dist, &target, 4, &device)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let expected = [1.7778859, 1.3875707];

        for (actual, expected) in loss.iter().zip(expected) {
            assert!((*actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic]
    fn box_loss_df_loss_out_of_range_target_matches_python_index_error() {
        let device = burn::tensor::Device::flex();
        let pred_dist = Tensor::<1>::from_floats([0.0; 4], &device).reshape([1, 4]);
        let target = Tensor::<1>::from_floats([3.0, 0.0, 0.0, 0.0], &device).reshape([1, 4]);

        let _ = super::BoxLoss::df_loss(&pred_dist, &target, 4, &device);
    }

    #[test]
    fn compute_loss_box_decode_matches_python_fixture() {
        let device = burn::tensor::Device::flex();
        let loss = ComputeLoss {
            nc: 1,
            ch: 4,
            nl: 1,
            stride: vec![8.0],
            box_gain: 7.5,
            cls_gain: 0.5,
            dfl_gain: 1.5,
        };
        let pred_dist = Tensor::<1>::from_floats(
            [
                0.2, 0.3, -0.1, 0.1, 0.4, -0.2, 1.0, 0.5, 1.2, 0.7, 0.3, -0.4, -0.7, 0.2, 0.0, 0.0,
                -0.5, -0.1, 0.8, 0.2, 0.1, 0.4, -1.0, -0.8, 0.0, 1.0, 0.0, 0.0, 0.0, -0.5, 0.0,
                0.2,
            ],
            &device,
        )
        .reshape([1, 16, 2]);
        let anchor_points = Tensor::<1>::from_floats([2.5, 3.5, 6.0, 1.0], &device).reshape([2, 2]);
        let strides = Tensor::<1>::from_floats([8.0, 8.0], &device).reshape([2, 1]);

        let decoded = loss
            .box_decode(&pred_dist, &anchor_points, &strides, &device)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let expected = [
            0.59932555, 2.633548, 3.786728, 5.0, 4.4380245, -0.2461841, 7.3652573, 2.0596902,
        ];

        for (actual, expected) in decoded.iter().zip(expected) {
            assert!((*actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn box_loss_forward_returns_weighted_box_and_dfl_losses() {
        let device = burn::tensor::Device::flex();
        let box_loss = super::BoxLoss::new(3);
        let pred_dist = Tensor::<1>::from_floats([0.0; 32], &device).reshape([2, 16]);
        let pred_bboxes =
            Tensor::<1>::from_floats([0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0], &device)
                .reshape([2, 4]);
        let anchor_points = Tensor::<1>::from_floats([1.0, 1.0, 3.0, 3.0], &device).reshape([2, 2]);
        let target_bboxes =
            Tensor::<1>::from_floats([0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0], &device)
                .reshape([2, 4]);
        let target_scores = Tensor::<1>::from_floats([0.5, 0.0], &device).reshape([2, 1]);

        let (loss_box, loss_dfl) = box_loss.forward(
            &pred_dist,
            &pred_bboxes,
            &anchor_points,
            &target_bboxes,
            &target_scores,
            0.5,
            &[true, false],
            &device,
        );

        let loss_box = loss_box.to_data().try_to_vec::<f32>().unwrap()[0];
        let loss_dfl = loss_dfl.to_data().try_to_vec::<f32>().unwrap()[0];
        assert!(loss_box.abs() < 1e-5);
        assert!((loss_dfl - 4.0_f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn compute_loss_call_keeps_background_cls_loss_for_empty_targets() {
        let device = burn::tensor::Device::flex();
        let stride = Tensor::<1>::from_floats([8.0], &device);
        let loss = ComputeLoss::new(1, 1, &stride, 7.5, 0.5, 1.5);
        let outputs = vec![Tensor::<1>::zeros([65], &device).reshape([1, 65, 1, 1])];
        let target_cls = Tensor::<1>::zeros([1], &device).reshape([1, 1]);
        let target_box = Tensor::<1>::zeros([4], &device).reshape([1, 4]);
        let target_idx = Tensor::<1>::from_floats([-1.0], &device);

        let (loss_box, loss_cls, loss_dfl) = loss.call(
            &outputs,
            &target_cls,
            &target_box,
            &target_idx,
            1,
            8,
            &device,
        );
        let loss_box = loss_box.to_data().try_to_vec::<f32>().unwrap()[0];
        let loss_cls = loss_cls.to_data().try_to_vec::<f32>().unwrap()[0];
        let loss_dfl = loss_dfl.to_data().try_to_vec::<f32>().unwrap()[0];

        assert_eq!(loss_box, 0.0);
        assert!((loss_cls - 0.5 * std::f32::consts::LN_2).abs() < 1e-6);
        assert_eq!(loss_dfl, 0.0);
    }

    #[test]
    fn compute_loss_call_returns_finite_losses_for_matched_target() {
        let device = burn::tensor::Device::flex();
        let stride = Tensor::<1>::from_floats([8.0], &device);
        let loss = ComputeLoss::new(1, 1, &stride, 7.5, 0.5, 1.5);
        let outputs = vec![Tensor::<1>::zeros([65 * 4 * 4], &device).reshape([1, 65, 4, 4])];
        let target_cls = Tensor::<1>::zeros([1], &device).reshape([1, 1]);
        let target_box = Tensor::<1>::from_floats([0.5, 0.5, 1.0, 1.0], &device).reshape([1, 4]);
        let target_idx = Tensor::<1>::from_floats([0.0], &device);

        let (loss_box, loss_cls, loss_dfl) = loss.call(
            &outputs,
            &target_cls,
            &target_box,
            &target_idx,
            1,
            32,
            &device,
        );
        let losses = [
            loss_box.to_data().try_to_vec::<f32>().unwrap()[0],
            loss_cls.to_data().try_to_vec::<f32>().unwrap()[0],
            loss_dfl.to_data().try_to_vec::<f32>().unwrap()[0],
        ];

        assert!(losses.iter().all(|loss| loss.is_finite()));
        assert!(losses[1] > 0.0);
    }

    #[test]
    fn compute_loss_call_assigns_targets_away_from_the_image_origin() {
        // The assigner compares anchor points against ground-truth boxes, and
        // both have to be in the same units. `make_anchors` produces grid units
        // (0..40 at stride 8 on a 320 px image) while the boxes are pixels, so
        // if the stride multiplication is dropped `is_inside` can only be true
        // near the origin — and a centred box assigns nothing at all.
        //
        // The older tests do not catch this because they use a 32 px image,
        // where the grid extent and the pixel extent are close enough that
        // every anchor still lands inside the box either way.
        let device = burn::tensor::Device::flex();
        let stride = Tensor::<1>::from_floats([8.0], &device);
        let loss = ComputeLoss::new(1, 1, &stride, 7.5, 0.5, 1.5);

        // 320 px at stride 8 is a 40 x 40 grid.
        let outputs = vec![Tensor::<1>::zeros([65 * 40 * 40], &device).reshape([1, 65, 40, 40])];
        let target_cls = Tensor::<1>::zeros([1], &device).reshape([1, 1]);
        // Centred, a tenth of the image across: pixels 144..176, nowhere near
        // the origin.
        let target_box = Tensor::<1>::from_floats([0.5, 0.5, 0.1, 0.1], &device).reshape([1, 4]);
        let target_idx = Tensor::<1>::from_floats([0.0], &device);

        let (loss_box, _loss_cls, loss_dfl) = loss.call(
            &outputs,
            &target_cls,
            &target_box,
            &target_idx,
            1,
            320,
            &device,
        );

        // Box and DFL losses are computed only over assigned anchors, so a
        // non-zero value here is the assertion that anything was assigned.
        let box_value = loss_box.to_data().try_to_vec::<f32>().unwrap()[0];
        let dfl_value = loss_dfl.to_data().try_to_vec::<f32>().unwrap()[0];
        assert!(
            box_value > 0.0,
            "no anchor was assigned to a centred box: box loss {box_value}"
        );
        assert!(
            dfl_value > 0.0,
            "no anchor was assigned to a centred box: dfl loss {dfl_value}"
        );
    }

    #[test]
    fn compute_loss_call_uses_python_batch_global_target_score_sum() {
        let device = burn::tensor::Device::flex();
        let stride = Tensor::<1>::from_floats([8.0], &device);
        let loss = ComputeLoss::new(1, 1, &stride, 1.0, 1.0, 1.0);

        let single_outputs = vec![Tensor::<1>::zeros([65 * 8 * 8], &device).reshape([1, 65, 8, 8])];
        let single_cls = Tensor::<1>::zeros([1], &device).reshape([1, 1]);
        let single_box = Tensor::<1>::from_floats([0.5, 0.5, 1.0, 1.0], &device).reshape([1, 4]);
        let single_idx = Tensor::<1>::from_floats([0.0], &device);
        let single_losses = loss.call(
            &single_outputs,
            &single_cls,
            &single_box,
            &single_idx,
            1,
            64,
            &device,
        );

        let batch_outputs =
            vec![Tensor::<1>::zeros([2 * 65 * 8 * 8], &device).reshape([2, 65, 8, 8])];
        let batch_cls = Tensor::<1>::zeros([2], &device).reshape([2, 1]);
        let batch_box = Tensor::<1>::from_floats([0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0], &device)
            .reshape([2, 4]);
        let batch_idx = Tensor::<1>::from_floats([0.0, 1.0], &device);
        let batch_losses = loss.call(
            &batch_outputs,
            &batch_cls,
            &batch_box,
            &batch_idx,
            2,
            64,
            &device,
        );

        let single = [
            single_losses.0.to_data().try_to_vec::<f32>().unwrap()[0],
            single_losses.1.to_data().try_to_vec::<f32>().unwrap()[0],
            single_losses.2.to_data().try_to_vec::<f32>().unwrap()[0],
        ];
        let batch = [
            batch_losses.0.to_data().try_to_vec::<f32>().unwrap()[0],
            batch_losses.1.to_data().try_to_vec::<f32>().unwrap()[0],
            batch_losses.2.to_data().try_to_vec::<f32>().unwrap()[0],
        ];

        for (single, batch) in single.iter().zip(batch) {
            assert!(
                (single - batch).abs() < 1e-5,
                "single={single}, batch={batch}"
            );
        }
    }
}
