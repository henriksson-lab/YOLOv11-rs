use anyhow::Result;
use image::{DynamicImage, GenericImageView, RgbImage, imageops::FilterType};

/// Resize and letterbox an image to `input_size x input_size`.
/// Returns (resized_image, (ratio_w, ratio_h), (pad_w, pad_h)).
pub fn letterbox(
    img: &DynamicImage,
    input_size: u32,
    augment: bool,
) -> Result<(RgbImage, (f32, f32), (f32, f32))> {
    let (w, h) = img.dimensions();
    let r = (input_size as f32 / h as f32).min(input_size as f32 / w as f32);
    let r = if augment { r } else { r.min(1.0) };

    let new_w = (w as f32 * r).round() as u32;
    let new_h = (h as f32 * r).round() as u32;

    let pad_w = (input_size - new_w) as f32 / 2.0;
    let pad_h = (input_size - new_h) as f32 / 2.0;

    let filter = if augment { FilterType::CatmullRom } else { FilterType::Triangle };
    let resized = img.resize_exact(new_w, new_h, filter);

    // Create output with zero-padding (gray/black border)
    let mut output = RgbImage::new(input_size, input_size);
    let left = pad_w.round() as i64;
    let top = pad_h.round() as i64;

    for y in 0..new_h {
        for x in 0..new_w {
            let px = resized.get_pixel(x, y).0;
            let ox = x as i64 + left;
            let oy = y as i64 + top;
            if ox >= 0 && ox < input_size as i64 && oy >= 0 && oy < input_size as i64 {
                output.put_pixel(ox as u32, oy as u32, image::Rgb([px[0], px[1], px[2]]));
            }
        }
    }

    Ok((output, (r, r), (pad_w, pad_h)))
}

/// Convert an RgbImage (HWC, u8) to a float tensor [C, H, W] in 0..1 range (RGB order).
pub fn image_to_tensor(
    img: &RgbImage,
    device: &candle_core::Device,
) -> candle_core::Result<candle_core::Tensor> {
    let (w, h) = img.dimensions();
    let raw = img.as_raw(); // [H*W*3] in RGB order

    // Rearrange HWC -> CHW
    let mut chw = vec![0.0f32; 3 * (h as usize) * (w as usize)];
    let hw = (h * w) as usize;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 3;
            chw[0 * hw + y * w as usize + x] = raw[idx] as f32 / 255.0;     // R
            chw[1 * hw + y * w as usize + x] = raw[idx + 1] as f32 / 255.0; // G
            chw[2 * hw + y * w as usize + x] = raw[idx + 2] as f32 / 255.0; // B
        }
    }

    candle_core::Tensor::from_vec(chw, (3, h as usize, w as usize), device)
}
