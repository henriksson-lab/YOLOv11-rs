use candle_core::{DType, Result, Tensor};

/// Generate anchor points and stride tensors for multi-scale feature maps.
///
/// Returns (anchors [total, 2], strides [total, 1]).
pub fn make_anchors(
    feature_maps: &[Tensor],
    strides: &Tensor,
    offset: f64,
) -> Result<(Tensor, Tensor)> {
    let device = feature_maps[0].device();
    let mut anchor_list = Vec::new();
    let mut stride_list = Vec::new();

    let strides_vec: Vec<f32> = strides.to_vec1()?;

    for (i, fm) in feature_maps.iter().enumerate() {
        let (_, _, h, w) = fm.dims4()?;
        let stride_val = strides_vec[i];

        // Create grid coordinates
        let sx: Vec<f32> = (0..w).map(|x| x as f32 + offset as f32).collect();
        let sy: Vec<f32> = (0..h).map(|y| y as f32 + offset as f32).collect();
        let sx = Tensor::from_vec(sx, w, device)?;
        let sy = Tensor::from_vec(sy, h, device)?;

        // Meshgrid: expand sy to [H, W] and sx to [H, W]
        let sy = sy.reshape((h, 1))?.expand((h, w))?;
        let sx = sx.reshape((1, w))?.expand((h, w))?;

        // Stack and flatten to [H*W, 2]
        let sy_flat = sy.reshape(h * w)?;
        let sx_flat = sx.reshape(h * w)?;
        let anchors = Tensor::stack(&[&sx_flat, &sy_flat], 1)?; // [H*W, 2]
        anchor_list.push(anchors);

        // Stride tensor [H*W, 1]
        let stride_t = Tensor::full(stride_val, (h * w, 1), device)?;
        stride_list.push(stride_t);
    }

    let anchor_refs: Vec<&Tensor> = anchor_list.iter().collect();
    let stride_refs: Vec<&Tensor> = stride_list.iter().collect();
    let anchors = Tensor::cat(&anchor_refs, 0)?;
    let strides = Tensor::cat(&stride_refs, 0)?;
    Ok((anchors, strides))
}

/// Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2].
pub fn wh2xy(x: &Tensor) -> Result<Tensor> {
    let cx = x.narrow(1, 0, 1)?;
    let cy = x.narrow(1, 1, 1)?;
    let w = x.narrow(1, 2, 1)?;
    let h = x.narrow(1, 3, 1)?;
    let half_w = (&w * 0.5)?;
    let half_h = (&h * 0.5)?;
    let x1 = cx.sub(&half_w)?;
    let y1 = cy.sub(&half_h)?;
    let x2 = cx.add(&half_w)?;
    let y2 = cy.add(&half_h)?;
    Tensor::cat(&[&x1, &y1, &x2, &y2], 1)
}

/// Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h].
pub fn xy2wh(x: &Tensor) -> Result<Tensor> {
    let x1 = x.narrow(1, 0, 1)?;
    let y1 = x.narrow(1, 1, 1)?;
    let x2 = x.narrow(1, 2, 1)?;
    let y2 = x.narrow(1, 3, 1)?;
    let cx = ((&x1 + &x2)? * 0.5)?;
    let cy = ((&y1 + &y2)? * 0.5)?;
    let w = x2.sub(&x1)?;
    let h = y2.sub(&y1)?;
    Tensor::cat(&[&cx, &cy, &w, &h], 1)
}

/// Compute IoU between two sets of boxes in [x1, y1, x2, y2] format.
/// box1: [N, 4], box2: [M, 4]. Returns [N, M].
pub fn box_iou(box1: &Tensor, box2: &Tensor) -> Result<Tensor> {
    let _n = box1.dims()[0];
    let _m = box2.dims()[0];

    // Expand for broadcasting: box1 [N, 1, 4], box2 [1, M, 4]
    let b1 = box1.unsqueeze(1)?; // [N, 1, 4]
    let b2 = box2.unsqueeze(0)?; // [1, M, 4]

    let b1_x1 = b1.narrow(2, 0, 1)?;
    let b1_y1 = b1.narrow(2, 1, 1)?;
    let b1_x2 = b1.narrow(2, 2, 1)?;
    let b1_y2 = b1.narrow(2, 3, 1)?;

    let b2_x1 = b2.narrow(2, 0, 1)?;
    let b2_y1 = b2.narrow(2, 1, 1)?;
    let b2_x2 = b2.narrow(2, 2, 1)?;
    let b2_y2 = b2.narrow(2, 3, 1)?;

    // Intersection
    let inter_x1 = b1_x1.broadcast_maximum(&b2_x1)?;
    let inter_y1 = b1_y1.broadcast_maximum(&b2_y1)?;
    let inter_x2 = b1_x2.broadcast_minimum(&b2_x2)?;
    let inter_y2 = b1_y2.broadcast_minimum(&b2_y2)?;

    let zero = Tensor::zeros(1, DType::F32, box1.device())?;
    let inter_w = inter_x2.sub(&inter_x1)?.broadcast_maximum(&zero)?;
    let inter_h = inter_y2.sub(&inter_y1)?.broadcast_maximum(&zero)?;
    let inter = inter_w.mul(&inter_h)?; // [N, M, 1]

    // Areas
    let area1 = (b1_x2.sub(&b1_x1)?.mul(&b1_y2.sub(&b1_y1)?))?; // [N, 1, 1]
    let area2 = (b2_x2.sub(&b2_x1)?.mul(&b2_y2.sub(&b2_y1)?))?; // [1, M, 1]

    let union = (area1.broadcast_add(&area2)?.sub(&inter))?;
    let iou = inter.broadcast_div(&(union + 1e-7)?)?;
    iou.squeeze(2) // [N, M]
}
