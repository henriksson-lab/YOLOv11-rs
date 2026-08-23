use burn::prelude::*;

/// Generate anchor points and stride tensors for multi-scale feature maps.
///
/// Returns (anchors [total, 2], strides [total, 1]).
pub fn make_anchors(
    feature_maps: &[Tensor<4>],
    strides: &Tensor<1>,
    offset: f64,
    device: &Device,
) -> (Tensor<2>, Tensor<2>) {
    let strides_vec: Vec<f32> = strides.to_data().try_to_vec::<f32>().unwrap();

    let mut anchor_list: Vec<Tensor<2>> = Vec::new();
    let mut stride_list: Vec<Tensor<2>> = Vec::new();

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
        let anchors = Tensor::<1>::from_floats(anchor_data.as_slice(), device).reshape([h * w, 2]);
        anchor_list.push(anchors);

        // Stride tensor [H*W, 1]
        let stride_data: Vec<f32> = vec![stride_val; h * w];
        let stride_t = Tensor::<1>::from_floats(stride_data.as_slice(), device).reshape([h * w, 1]);
        stride_list.push(stride_t);
    }

    let anchors = Tensor::cat(anchor_list, 0);
    let strides = Tensor::cat(stride_list, 0);
    (anchors, strides)
}

/// Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2].
pub fn wh2xy(x: &Tensor<2>) -> Tensor<2> {
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

#[cfg(test)]
mod tests {
    use super::{make_anchors, wh2xy};
    use burn::prelude::*;

    #[test]
    fn make_anchors_matches_python_grid_order_and_shapes() {
        let device = burn::tensor::Device::flex();
        let x0 = Tensor::<4>::zeros([1, 8, 2, 3], &device);
        let x1 = Tensor::<4>::zeros([1, 8, 1, 2], &device);
        let strides = Tensor::<1>::from_floats([8.0, 16.0], &device);

        let (anchors, stride_tensor) = make_anchors(&[x0, x1], &strides, 0.5, &device);

        assert_eq!(anchors.dims(), [8, 2]);
        assert_eq!(stride_tensor.dims(), [8, 1]);
        assert_eq!(
            anchors.to_data().try_to_vec::<f32>().unwrap(),
            vec![0.5, 0.5, 1.5, 0.5, 2.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 1.5, 0.5, 0.5, 1.5, 0.5,]
        );
        assert_eq!(
            stride_tensor.to_data().try_to_vec::<f32>().unwrap(),
            vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 16.0, 16.0]
        );
    }

    #[test]
    fn wh2xy_matches_python_edge_fixture() {
        let device = burn::tensor::Device::flex();
        let boxes = Tensor::<1>::from_floats(
            [
                10.0, 12.0, 4.0, 6.0, -2.0, 3.0, 8.0, 10.0, 0.0, 0.0, 0.0, 2.0,
            ],
            &device,
        )
        .reshape([3, 4]);

        let xy = wh2xy(&boxes);
        assert_eq!(
            xy.to_data().try_to_vec::<f32>().unwrap(),
            vec![8.0, 9.0, 12.0, 15.0, -6.0, -2.0, 2.0, 8.0, 0.0, -1.0, 0.0, 1.0]
        );
    }
}
