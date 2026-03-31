use candle_core::{DType, Result, Tensor};

use crate::anchors;

// ---------------------------------------------------------------------------
// CIoU loss (Complete IoU)
// ---------------------------------------------------------------------------

/// Compute CIoU between pred [N, 4] and target [N, 4] boxes in xyxy format.
/// Returns (ciou, iou) both [N].
pub fn compute_ciou(pred: &Tensor, target: &Tensor) -> Result<(Tensor, Tensor)> {
    let eps = 1e-7;

    let p_x1 = pred.narrow(1, 0, 1)?;
    let p_y1 = pred.narrow(1, 1, 1)?;
    let p_x2 = pred.narrow(1, 2, 1)?;
    let p_y2 = pred.narrow(1, 3, 1)?;

    let t_x1 = target.narrow(1, 0, 1)?;
    let t_y1 = target.narrow(1, 1, 1)?;
    let t_x2 = target.narrow(1, 2, 1)?;
    let t_y2 = target.narrow(1, 3, 1)?;

    // Intersection
    let inter_x1 = p_x1.maximum(&t_x1)?;
    let inter_y1 = p_y1.maximum(&t_y1)?;
    let inter_x2 = p_x2.minimum(&t_x2)?;
    let inter_y2 = p_y2.minimum(&t_y2)?;

    let zero = Tensor::zeros(1, DType::F32, pred.device())?;
    let inter_w = inter_x2.sub(&inter_x1)?.maximum(&zero)?;
    let inter_h = inter_y2.sub(&inter_y1)?.maximum(&zero)?;
    let inter = inter_w.mul(&inter_h)?;

    // Union
    let p_w = p_x2.sub(&p_x1)?;
    let p_h = p_y2.sub(&p_y1)?;
    let t_w = t_x2.sub(&t_x1)?;
    let t_h = t_y2.sub(&t_y1)?;
    let area_p = p_w.mul(&p_h)?;
    let area_t = t_w.mul(&t_h)?;
    let union = (area_p.add(&area_t)?.sub(&inter)? + eps)?;
    let iou = inter.div(&union)?;

    // Enclosing box
    let enc_x1 = p_x1.minimum(&t_x1)?;
    let enc_y1 = p_y1.minimum(&t_y1)?;
    let enc_x2 = p_x2.maximum(&t_x2)?;
    let enc_y2 = p_y2.maximum(&t_y2)?;
    let enc_w = enc_x2.sub(&enc_x1)?;
    let enc_h = enc_y2.sub(&enc_y1)?;
    let c2 = enc_w.sqr()?.add(&enc_h.sqr()?)?; // diagonal squared

    // Center distance
    let p_cx = (p_x1.add(&p_x2)? * 0.5)?;
    let p_cy = (p_y1.add(&p_y2)? * 0.5)?;
    let t_cx = (t_x1.add(&t_x2)? * 0.5)?;
    let t_cy = (t_y1.add(&t_y2)? * 0.5)?;
    let d2 = (p_cx.sub(&t_cx)?.sqr()?.add(&p_cy.sub(&t_cy)?.sqr()?))?;

    // DIoU (skipping aspect-ratio term since atan is unavailable in candle)
    let diou = iou.sub(&d2.div(&(c2 + eps)?)?)?;

    let iou_flat = iou.squeeze(1)?;
    let diou_flat = diou.squeeze(1)?;
    Ok((diou_flat, iou_flat))
}

// ---------------------------------------------------------------------------
// DFL loss  (distribution focal loss for box regression)
// ---------------------------------------------------------------------------

/// Distribution Focal Loss.
/// pred_dist: [N, 4*ch] predicted distribution logits.
/// target: [N, 4] target values (continuous, will be discretized).
/// ch: number of DFL bins (16).
pub fn df_loss(pred_dist: &Tensor, target: &Tensor, ch: usize) -> Result<Tensor> {
    let target_flat = target.flatten_all()?;
    let target_left = target_flat.floor()?.clamp(0.0, (ch - 1) as f64)?;
    let target_right = (&target_left + 1.0)?.clamp(0.0, (ch - 1) as f64)?;
    let weight_right = target_flat.sub(&target_left)?;
    let weight_left = (1.0 - &weight_right)?;

    // pred_dist: [N, 4*ch] -> [N*4, ch]
    let n = pred_dist.dims()[0];
    let pd = pred_dist.reshape((n * 4, ch))?;
    let log_sm = candle_nn::ops::log_softmax(&pd, 1)?;

    // Gather left and right indices
    let tl = target_left.to_dtype(DType::U32)?;
    let tr = target_right.to_dtype(DType::U32)?;

    // Manual gather: for each row, index into log_softmax
    let tl_1d: Vec<u32> = tl.to_vec1()?;
    let tr_1d: Vec<u32> = tr.to_vec1()?;
    let wl: Vec<f32> = weight_left.to_vec1()?;
    let wr: Vec<f32> = weight_right.to_vec1()?;

    let log_sm_2d: Vec<Vec<f32>> = (0..n * 4)
        .map(|i| {
            let row = log_sm.get(i).unwrap();
            row.to_vec1::<f32>().unwrap()
        })
        .collect();

    let mut loss_vals = Vec::with_capacity(n * 4);
    for i in 0..n * 4 {
        let ll = -log_sm_2d[i][tl_1d[i] as usize] * wl[i];
        let lr = -log_sm_2d[i][tr_1d[i] as usize] * wr[i];
        loss_vals.push(ll + lr);
    }

    let loss = Tensor::from_vec(loss_vals, (n, 4), pred_dist.device())?;
    loss.mean(1) // [N]
}

// ---------------------------------------------------------------------------
// BCE with logits
// ---------------------------------------------------------------------------

/// Binary cross-entropy with logits (numerically stable).
pub fn bce_with_logits(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    // max(x, 0) - x*target + log(1 + exp(-|x|))
    let zero = Tensor::zeros(1, DType::F32, pred.device())?;
    let relu_x = pred.maximum(&zero)?;
    let neg_abs = pred.abs()?.neg()?;
    let log_term = neg_abs.exp()?.affine(1.0, 1.0)?.log()?;
    relu_x.sub(&pred.mul(target)?)?.add(&log_term)
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
///
/// - pd_scores: [num_anchors, nc] predicted class scores (sigmoid).
/// - pd_bboxes: [num_anchors, 4] predicted boxes (xyxy).
/// - anc_points: [num_anchors, 2] anchor center points.
/// - gt_labels: [num_gt] ground-truth class labels.
/// - gt_bboxes: [num_gt, 4] ground-truth boxes (xyxy).
///
/// Returns assignment for each anchor.
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
    let mut is_inside = vec![vec![false; num_gt]; num_anchors]; // [A, G]
    for a in 0..num_anchors {
        let (ax, ay) = (anc_points[a][0], anc_points[a][1]);
        for g in 0..num_gt {
            let b = &gt_bboxes[g];
            if ax >= b[0] && ax <= b[2] && ay >= b[1] && ay <= b[3] {
                is_inside[a][g] = true;
            }
        }
    }

    // Compute alignment metric: score^alpha * iou^beta  (alpha=0.5, beta=6.0 typical)
    let alpha = 0.5f32;
    let beta = 6.0f32;
    let mut metrics = vec![vec![0.0f32; num_gt]; num_anchors]; // [A, G]
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
    let mut assigned_gt = vec![-1i64; num_anchors]; // which GT each anchor is assigned to
    let mut assigned_metric = vec![0.0f32; num_anchors];

    for g in 0..num_gt {
        // Collect (metric, anchor_idx) for this GT, only valid anchors
        let mut candidates: Vec<(f32, usize)> = Vec::new();
        for a in 0..num_anchors {
            if is_inside[a][g] && metrics[a][g] > 0.0 {
                candidates.push((metrics[a][g], a));
            }
        }
        // Sort descending by metric
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
    pub stride: Tensor,
    pub box_gain: f64,
    pub cls_gain: f64,
    pub dfl_gain: f64,
}

impl ComputeLoss {
    pub fn new(
        nc: usize,
        nl: usize,
        stride: Tensor,
        box_gain: f64,
        cls_gain: f64,
        dfl_gain: f64,
    ) -> Self {
        Self {
            nc,
            ch: 16,
            nl,
            stride,
            box_gain,
            cls_gain,
            dfl_gain,
        }
    }

    /// Decode DFL distribution to box coordinates.
    /// pred_dist: [B, 4*ch, A], anchor_points: [A, 2], strides: [A, 1].
    /// Returns [B, A, 4] decoded boxes in pixel coords (xyxy).
    fn box_decode(
        &self,
        pred_dist: &Tensor,
        anchor_points: &Tensor,
        strides: &Tensor,
    ) -> Result<Tensor> {
        let (b, _, a) = pred_dist.dims3()?;

        // DFL: [B, 4*ch, A] -> [B, 4, ch, A] -> softmax -> weighted sum -> [B, 4, A]
        let pd = pred_dist.reshape((b, 4, self.ch, a))?;
        let pd = pd.transpose(2, 3)?; // [B, 4, A, ch]
        let pd = candle_nn::ops::softmax(&pd, 3)?;
        let bins: Vec<f32> = (0..self.ch).map(|i| i as f32).collect();
        let bins = Tensor::from_vec(bins, self.ch, pred_dist.device())?;
        let pd = pd.broadcast_mul(&bins)?.sum(3)?; // [B, 4, A]
        let pd = pd.transpose(1, 2)?; // [B, A, 4]

        // Split into lt, rb (left-top, right-bottom distances)
        let lt = pd.narrow(2, 0, 2)?; // [B, A, 2]
        let rb = pd.narrow(2, 2, 2)?; // [B, A, 2]

        let anc = anchor_points.unsqueeze(0)?; // [1, A, 2]
        let x1y1 = anc.broadcast_sub(&lt)?;
        let x2y2 = anc.broadcast_add(&rb)?;
        let boxes = Tensor::cat(&[&x1y1, &x2y2], 2)?; // [B, A, 4]
        let st = strides.unsqueeze(0)?; // [1, A, 1]
        boxes.broadcast_mul(&st)
    }

    /// Compute training loss.
    /// outputs: per-scale feature maps from Head (training mode).
    /// targets: dict-like with cls [num_targets, 1], box [num_targets, 4], idx [num_targets].
    ///
    /// Returns (loss_box, loss_cls, loss_dfl).
    pub fn compute(
        &self,
        outputs: &[Tensor],
        target_cls: &Tensor,  // [num_targets, 1]
        target_box: &Tensor,  // [num_targets, 4]  normalized xywh
        target_idx: &Tensor,  // [num_targets]  batch index
        batch_size: usize,
        img_size: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let device = outputs[0].device();
        let no = self.nc + self.ch * 4;

        // Flatten outputs: [B, no, total_A]
        let mut flat = Vec::new();
        for o in outputs {
            let (b, _, h, w) = o.dims4()?;
            flat.push(o.reshape((b, no, h * w))?);
        }
        let refs: Vec<&Tensor> = flat.iter().collect();
        let x = Tensor::cat(&refs, 2)?; // [B, no, total_A]

        let box_ch = self.ch * 4;
        let pred_dist = x.narrow(1, 0, box_ch)?;       // [B, 4*ch, A]
        let pred_scores = x.narrow(1, box_ch, self.nc)?; // [B, nc, A]

        // Generate anchors
        let (anchor_points, strides) =
            crate::anchors::make_anchors(outputs, &self.stride, 0.5)?;
        // anchor_points: [A, 2], strides: [A, 1]

        let total_a = anchor_points.dims()[0];

        // Decode predicted boxes
        let pred_bboxes = self.box_decode(&pred_dist, &anchor_points, &strides)?;
        // [B, A, 4] in pixel coords

        let pred_scores_t = pred_scores.transpose(1, 2)?; // [B, A, nc]

        // Convert targets to pixel coords
        let target_box_xyxy = anchors::wh2xy(target_box)?; // [T, 4] normalized xyxy
        let target_box_pixel = (target_box_xyxy * img_size as f64)?;

        // Per-image loss computation
        let mut total_box_loss = Tensor::zeros(1, DType::F32, &device)?;
        let mut total_cls_loss = Tensor::zeros(1, DType::F32, &device)?;
        let mut total_dfl_loss = Tensor::zeros(1, DType::F32, &device)?;
        let mut total_fg = 0usize;

        let target_idx_vec: Vec<f32> = target_idx.to_vec1()?;
        let target_cls_vec: Vec<f32> = target_cls.squeeze(1)?.to_vec1()?;
        let target_box_vec: Vec<Vec<f32>> = (0..target_box_pixel.dims()[0])
            .map(|i| target_box_pixel.get(i).unwrap().to_vec1::<f32>().unwrap())
            .collect();

        for b in 0..batch_size {
            // Get targets for this image
            let mut gt_labels = Vec::new();
            let mut gt_bboxes = Vec::new();
            for t in 0..target_idx_vec.len() {
                if target_idx_vec[t] as usize == b {
                    gt_labels.push(target_cls_vec[t] as i64);
                    gt_bboxes.push([
                        target_box_vec[t][0],
                        target_box_vec[t][1],
                        target_box_vec[t][2],
                        target_box_vec[t][3],
                    ]);
                }
            }

            // Extract per-image predictions (detached for assigner)
            let pb = pred_bboxes.get(b)?;        // [A, 4]
            let ps = pred_scores_t.get(b)?;      // [A, nc]
            let ps_sigmoid = candle_nn::ops::sigmoid(&ps)?;

            let ps_vec: Vec<Vec<f32>> = (0..total_a)
                .map(|i| ps_sigmoid.get(i).unwrap().to_vec1::<f32>().unwrap())
                .collect();
            let pb_vec: Vec<[f32; 4]> = (0..total_a)
                .map(|i| {
                    let row: Vec<f32> = pb.get(i).unwrap().to_vec1().unwrap();
                    [row[0], row[1], row[2], row[3]]
                })
                .collect();
            let anc_vec: Vec<[f32; 2]> = (0..total_a)
                .map(|i| {
                    let row: Vec<f32> = anchor_points.get(i).unwrap().to_vec1().unwrap();
                    [row[0], row[1]]
                })
                .collect();

            let assign = task_aligned_assign(&ps_vec, &pb_vec, &anc_vec, &gt_labels, &gt_bboxes, 10);

            // Count foreground
            let fg_count: usize = assign.fg_mask.iter().filter(|&&v| v).count();
            total_fg += fg_count;

            if fg_count == 0 {
                // No foreground — cls loss is BCE against zeros
                let cls_target = Tensor::zeros((total_a, self.nc), DType::F32, &device)?;
                let cls_loss = bce_with_logits(&ps, &cls_target)?.sum_all()?;
                total_cls_loss = total_cls_loss.add(&cls_loss)?;
                continue;
            }

            // Build target tensors from assignment
            let fg_indices: Vec<usize> = assign
                .fg_mask
                .iter()
                .enumerate()
                .filter(|(_, &v)| v)
                .map(|(i, _)| i)
                .collect();

            // Classification target: one-hot * score
            let mut cls_target_data = vec![0.0f32; total_a * self.nc];
            for &a in &fg_indices {
                let cls = assign.target_labels[a] as usize;
                if cls < self.nc {
                    cls_target_data[a * self.nc + cls] = assign.target_scores[a];
                }
            }
            let cls_target = Tensor::from_vec(cls_target_data, (total_a, self.nc), &device)?;
            let cls_loss = bce_with_logits(&ps, &cls_target)?.sum_all()?;
            total_cls_loss = total_cls_loss.add(&cls_loss)?;

            // Box loss (CIoU) on foreground only
            let fg_idx_tensor = Tensor::from_vec(
                fg_indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                fg_count,
                &device,
            )?;
            let fg_pred_boxes = pb.index_select(&fg_idx_tensor, 0)?;

            let fg_target_boxes_data: Vec<f32> = fg_indices
                .iter()
                .flat_map(|&a| assign.target_bboxes[a].iter().copied())
                .collect();
            let fg_target_boxes =
                Tensor::from_vec(fg_target_boxes_data, (fg_count, 4), &device)?;

            let (ciou, _iou) = compute_ciou(&fg_pred_boxes, &fg_target_boxes)?;
            let box_loss = (1.0 - ciou)?.mean_all()?;
            total_box_loss = total_box_loss.add(&box_loss)?;

            // DFL loss on foreground
            let pred_dist_b = pred_dist.get(b)?; // [4*ch, A]
            let pred_dist_b = pred_dist_b.transpose(0, 1)?; // [A, 4*ch]
            let fg_pred_dist = pred_dist_b.index_select(&fg_idx_tensor, 0)?; // [fg, 4*ch]

            // Target for DFL: convert target boxes back to lt/rb distances in stride units
            let fg_anc_data: Vec<f32> = fg_indices
                .iter()
                .flat_map(|&a| anc_vec[a].iter().copied())
                .collect();
            let fg_anc = Tensor::from_vec(fg_anc_data, (fg_count, 2), &device)?;

            let fg_strides = strides.index_select(&fg_idx_tensor, 0)?; // [fg, 1]
            let fg_tb_xy1 = fg_target_boxes.narrow(1, 0, 2)?;
            let fg_tb_xy2 = fg_target_boxes.narrow(1, 2, 2)?;
            let lt = fg_anc.sub(&fg_tb_xy1.div(&fg_strides)?)?;
            let rb = fg_tb_xy2.div(&fg_strides)?.sub(&fg_anc)?;
            let dfl_target = Tensor::cat(&[&lt, &rb], 1)?; // [fg, 4]
            let dfl_target = dfl_target.clamp(0.0, (self.ch - 1) as f64)?;

            let dfl_l = df_loss(&fg_pred_dist, &dfl_target, self.ch)?;
            let dfl_loss = dfl_l.mean_all()?;
            total_dfl_loss = total_dfl_loss.add(&dfl_loss)?;
        }

        // Normalize by total foreground count
        let norm = (total_fg.max(1) as f64).recip();
        let loss_box = (total_box_loss * (norm * self.box_gain))?;
        let loss_cls = (total_cls_loss * (norm * self.cls_gain))?;
        let loss_dfl = (total_dfl_loss * (norm * self.dfl_gain))?;

        Ok((loss_box, loss_cls, loss_dfl))
    }
}
