use anyhow::Result;
use image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage};

/// Returns (resized_image, (ratio_w, ratio_h), (pad_w, pad_h)).
pub fn resize(
    img: &DynamicImage,
    input_size: u32,
    augment: bool,
) -> Result<(RgbImage, (f32, f32), (f32, f32))> {
    let (w, h) = img.dimensions();
    let r = (input_size as f64 / h as f64).min(input_size as f64 / w as f64);
    let r = if augment { r } else { r.min(1.0) };

    let new_w = (w as f64 * r).round_ties_even() as u32;
    let new_h = (h as f64 * r).round_ties_even() as u32;

    let pad_w = (input_size - new_w) as f64 / 2.0;
    let pad_h = (input_size - new_h) as f64 / 2.0;

    let filter = if augment {
        resample()
    } else {
        FilterType::Triangle
    };
    let resized = img.resize_exact(new_w, new_h, filter);

    let mut output = RgbImage::new(input_size, input_size);
    let left = (pad_w - 0.1).round_ties_even() as i64;
    let top = (pad_h - 0.1).round_ties_even() as i64;

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

    Ok((output, (r as f32, r as f32), (pad_w as f32, pad_h as f32)))
}

pub fn resample() -> FilterType {
    let choices = [
        FilterType::Gaussian,
        FilterType::CatmullRom,
        FilterType::Triangle,
        FilterType::Nearest,
        FilterType::Lanczos3,
    ];
    choices[crate::rng::gen_range_usize(0..choices.len())]
}

#[cfg(test)]
mod tests {
    use super::{resample, resize};
    use image::{imageops::FilterType, DynamicImage, Rgb, RgbImage};

    #[test]
    fn resize_places_odd_padding_like_python_letterbox() {
        let _lock = crate::rng::test_lock();
        let mut img = RgbImage::new(2, 4);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([255, 0, 0]);
        }

        let (out, ratio, pad) = resize(&DynamicImage::ImageRgb8(img), 5, true).unwrap();

        assert_eq!(ratio, (1.25, 1.25));
        assert_eq!(pad, (1.5, 0.0));
        assert_eq!(out.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(out.get_pixel(1, 0).0, [255, 0, 0]);
        assert_eq!(out.get_pixel(2, 4).0, [255, 0, 0]);
        assert_eq!(out.get_pixel(3, 4).0, [0, 0, 0]);
        assert_eq!(out.get_pixel(4, 4).0, [0, 0, 0]);
    }

    #[test]
    fn resize_uses_python_ties_to_even_rounding_for_scaled_shape() {
        let _lock = crate::rng::test_lock();
        let mut img = RgbImage::new(5, 10);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([255, 0, 0]);
        }

        let (out, ratio, pad) = resize(&DynamicImage::ImageRgb8(img), 5, true).unwrap();

        assert_eq!(ratio, (0.5, 0.5));
        assert_eq!(pad, (1.5, 0.0));
        assert_eq!(out.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(out.get_pixel(1, 0).0, [255, 0, 0]);
        assert_eq!(out.get_pixel(2, 4).0, [255, 0, 0]);
        assert_eq!(out.get_pixel(3, 4).0, [0, 0, 0]);
        assert_eq!(out.get_pixel(4, 4).0, [0, 0, 0]);
    }

    #[test]
    fn resample_exposes_five_python_choice_equivalents() {
        let _lock = crate::rng::test_lock();
        let mut seen = Vec::new();
        for seed in 0..100 {
            crate::rng::set_seed(seed);
            let filter = resample();
            if !seen.contains(&filter) {
                seen.push(filter);
            }
            if seen.len() == 5 {
                break;
            }
        }

        assert!(seen.contains(&FilterType::Gaussian));
        assert!(seen.contains(&FilterType::CatmullRom));
        assert!(seen.contains(&FilterType::Triangle));
        assert!(seen.contains(&FilterType::Nearest));
        assert!(seen.contains(&FilterType::Lanczos3));
    }
}
