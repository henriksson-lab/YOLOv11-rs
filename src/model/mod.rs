pub mod anchors;
pub mod attention;
pub mod backbone;
pub mod blocks;
pub mod conv;
pub mod head;
pub mod loss;
pub mod metrics;
pub mod model;
pub mod neck;
pub mod nms;

/// Helper: split a tensor along `dim` into chunks of the given sizes.
pub fn tensor_split(
    t: &candle_core::Tensor,
    sizes: &[usize],
    dim: usize,
) -> candle_core::Result<Vec<candle_core::Tensor>> {
    let mut offset = 0;
    let mut out = Vec::with_capacity(sizes.len());
    for &s in sizes {
        out.push(t.narrow(dim, offset, s)?);
        offset += s;
    }
    Ok(out)
}
