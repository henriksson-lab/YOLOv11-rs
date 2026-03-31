use burn::prelude::*;

/// Generate anchor points and stride tensors for multi-scale feature maps.
///
/// Returns (anchors [total, 2], strides [total, 1]).
pub fn make_anchors<B: Backend>(
    feature_maps: &[Tensor<B, 4>],
    strides: &Tensor<B, 1>,
    offset: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let strides_vec: Vec<f32> = strides.to_data().to_vec().unwrap();

    let mut anchor_list: Vec<Tensor<B, 2>> = Vec::new();
    let mut stride_list: Vec<Tensor<B, 2>> = Vec::new();

    for (i, fm) in feature_maps.iter().enumerate() {
        let [_, _, h, w] = fm.dims();
        let stride_val = strides_vec[i];

        // Create grid coordinates
        let sx: Vec<f32> = (0..w).map(|x| x as f32 + offset as f32).collect();
        let sy: Vec<f32> = (0..h).map(|y| y as f32 + offset as f32).collect();

        // Build anchors [H*W, 2]
        let mut anchor_data = Vec::with_capacity(h * w * 2);
        for y in 0..h {
            for x in 0..w {
                anchor_data.push(sx[x]);
                anchor_data.push(sy[y]);
            }
        }
        let anchors =
            Tensor::<B, 1>::from_floats(anchor_data.as_slice(), device).reshape([h * w, 2]);
        anchor_list.push(anchors);

        // Stride tensor [H*W, 1]
        let stride_data: Vec<f32> = vec![stride_val; h * w];
        let stride_t =
            Tensor::<B, 1>::from_floats(stride_data.as_slice(), device).reshape([h * w, 1]);
        stride_list.push(stride_t);
    }

    let anchors = Tensor::cat(anchor_list, 0);
    let strides = Tensor::cat(stride_list, 0);
    (anchors, strides)
}

/// Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2].
pub fn wh2xy<B: Backend>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
    let cx = x.clone().narrow(1, 0, 1);
    let cy = x.clone().narrow(1, 1, 1);
    let w = x.clone().narrow(1, 2, 1);
    let h = x.clone().narrow(1, 3, 1);
    let half_w = w.clone() * 0.5;
    let half_h = h.clone() * 0.5;
    let x1 = cx.clone() - half_w.clone();
    let y1 = cy.clone() - half_h.clone();
    let x2 = cx + half_w;
    let y2 = cy + half_h;
    Tensor::cat(vec![x1, y1, x2, y2], 1)
}

/// Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h].
pub fn xy2wh<B: Backend>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
    let x1 = x.clone().narrow(1, 0, 1);
    let y1 = x.clone().narrow(1, 1, 1);
    let x2 = x.clone().narrow(1, 2, 1);
    let y2 = x.clone().narrow(1, 3, 1);
    let cx = (x1.clone() + x2.clone()) * 0.5;
    let cy = (y1.clone() + y2.clone()) * 0.5;
    let w = x2 - x1;
    let h = y2 - y1;
    Tensor::cat(vec![cx, cy, w, h], 1)
}

/// Compute IoU between two sets of boxes in [x1, y1, x2, y2] format.
/// box1: [N, 4], box2: [M, 4]. Returns [N, M].
pub fn box_iou<B: Backend>(box1: &Tensor<B, 2>, box2: &Tensor<B, 2>) -> Tensor<B, 2> {
    // Expand for broadcasting: box1 [N, 1, 4], box2 [1, M, 4]
    let b1: Tensor<B, 3> = box1.clone().unsqueeze_dim(1); // [N, 1, 4]
    let b2: Tensor<B, 3> = box2.clone().unsqueeze_dim(0); // [1, M, 4]

    let b1_x1 = b1.clone().narrow(2, 0, 1);
    let b1_y1 = b1.clone().narrow(2, 1, 1);
    let b1_x2 = b1.clone().narrow(2, 2, 1);
    let b1_y2 = b1.narrow(2, 3, 1);

    let b2_x1 = b2.clone().narrow(2, 0, 1);
    let b2_y1 = b2.clone().narrow(2, 1, 1);
    let b2_x2 = b2.clone().narrow(2, 2, 1);
    let b2_y2 = b2.narrow(2, 3, 1);

    // Intersection
    let inter_x1 = b1_x1.clone().max_pair(b2_x1.clone());
    let inter_y1 = b1_y1.clone().max_pair(b2_y1.clone());
    let inter_x2 = b1_x2.clone().min_pair(b2_x2.clone());
    let inter_y2 = b1_y2.clone().min_pair(b2_y2.clone());

    let inter_w = (inter_x2 - inter_x1).clamp_min(0.0);
    let inter_h = (inter_y2 - inter_y1).clamp_min(0.0);
    let inter = inter_w * inter_h; // [N, M, 1]

    // Areas
    let area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1); // [N, 1, 1]
    let area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1); // [1, M, 1]

    let union = area1 + area2 - inter.clone() + 1e-7;
    let iou = inter / union;
    iou.squeeze::<2>(2) // [N, M]
}
