use image::RgbImage;
use rand_distr::Beta;

use crate::data::dataset::{AugmentParams, Label};

/// HSV color-space augmentation.
pub fn augment_hsv(img: &mut RgbImage, params: &AugmentParams) {
    let rh = crate::rng::gen_range_f32(-1.0..1.0) * params.hsv_h + 1.0;
    let rs = crate::rng::gen_range_f32(-1.0..1.0) * params.hsv_s + 1.0;
    let rv = crate::rng::gen_range_f32(-1.0..1.0) * params.hsv_v + 1.0;
    let lut_h = std::array::from_fn::<_, 256, _>(|x| (x as f32 * rh).rem_euclid(180.0) as u8);
    let lut_s = std::array::from_fn::<_, 256, _>(|x| (x as f32 * rs).clamp(0.0, 255.0) as u8);
    let lut_v = std::array::from_fn::<_, 256, _>(|x| (x as f32 * rv).clamp(0.0, 255.0) as u8);

    let bgr_to_hsv = |b: u8, g: u8, r: u8| -> (u8, u8, u8) {
        let rf = r as f32 / 255.0;
        let gf = g as f32 / 255.0;
        let bf = b as f32 / 255.0;
        let max = rf.max(gf).max(bf);
        let min = rf.min(gf).min(bf);
        let diff = max - min;

        let h = if diff == 0.0 {
            0.0
        } else if max == rf {
            60.0 * (((gf - bf) / diff) % 6.0)
        } else if max == gf {
            60.0 * ((bf - rf) / diff + 2.0)
        } else {
            60.0 * ((rf - gf) / diff + 4.0)
        };
        let h = if h < 0.0 { h + 360.0 } else { h };

        let s = if max == 0.0 { 0.0 } else { diff / max };
        let v = max;

        ((h / 2.0) as u8, (s * 255.0) as u8, (v * 255.0) as u8)
    };

    let hsv_to_bgr = |h: u8, s: u8, v: u8| -> (u8, u8, u8) {
        let h = h as f32 * 2.0;
        let s = s as f32 / 255.0;
        let v = v as f32 / 255.0;

        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        (
            ((b + m) * 255.0) as u8,
            ((g + m) * 255.0) as u8,
            ((r + m) * 255.0) as u8,
        )
    };

    for pixel in img.pixels_mut() {
        let [b, g, r] = pixel.0;
        let (h, s, v) = bgr_to_hsv(b, g, r);
        let h2 = lut_h[h as usize];
        let s2 = lut_s[s as usize];
        let v2 = lut_v[v as usize];
        let (b2, g2, r2) = hsv_to_bgr(h2, s2, v2);
        pixel.0 = [b2, g2, r2];
    }
}

/// Apply random perspective (affine) transformation.
/// `border` offsets for mosaic (typically [-input_size/2, -input_size/2]).
pub fn random_perspective(
    img: &RgbImage,
    labels: &[Label],
    params: &AugmentParams,
    border: &[i32; 2],
) -> (RgbImage, Vec<Label>) {
    let h = img.height() as f32 + border[0] as f32 * 2.0;
    let w = img.width() as f32 + border[1] as f32 * 2.0;
    let h = h as u32;
    let w = w as u32;

    let angle = (-params.degrees) + crate::rng::gen_f32() * (params.degrees - (-params.degrees));
    let s = (1.0 - params.scale)
        + crate::rng::gen_f32() * ((1.0 + params.scale) - (1.0 - params.scale));
    let sh_x = ((-params.shear) + crate::rng::gen_f32() * (params.shear - (-params.shear)))
        * std::f32::consts::PI
        / 180.0;
    let sh_x = sh_x.tan();
    let sh_y = ((-params.shear) + crate::rng::gen_f32() * (params.shear - (-params.shear)))
        * std::f32::consts::PI
        / 180.0;
    let sh_y = sh_y.tan();
    let tx = ((0.5 - params.translate)
        + crate::rng::gen_f32() * ((0.5 + params.translate) - (0.5 - params.translate)))
        * w as f32;
    let ty = ((0.5 - params.translate)
        + crate::rng::gen_f32() * ((0.5 + params.translate) - (0.5 - params.translate)))
        * h as f32;

    let cos_a = (angle * std::f32::consts::PI / 180.0).cos() * s;
    let sin_a = (angle * std::f32::consts::PI / 180.0).sin() * s;
    let cx = img.width() as f32 / 2.0;
    let cy = img.height() as f32 / 2.0;

    // Python computes translate @ shear @ rotate @ perspective @ center, with
    // cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s) providing:
    // [[cos(a)*s, sin(a)*s], [-sin(a)*s, cos(a)*s]].
    let m00 = cos_a - sh_x * sin_a;
    let m01 = sin_a + sh_x * cos_a;
    let m02 = tx - cx * m00 - cy * m01;
    let m10 = sh_y * cos_a - sin_a;
    let m11 = sh_y * sin_a + cos_a;
    let m12 = ty - cx * m10 - cy * m11;

    // Warp image using inverse mapping
    let mut output = RgbImage::new(w, h);
    // Compute inverse matrix for sampling
    let det = m00 * m11 - m01 * m10;
    if det.abs() < 1e-6 {
        return (output, Vec::new());
    }
    let inv_det = 1.0 / det;
    let i00 = m11 * inv_det;
    let i01 = -m01 * inv_det;
    let i02 = (m01 * m12 - m02 * m11) * inv_det;
    let i10 = -m10 * inv_det;
    let i11 = m00 * inv_det;
    let i12 = (m02 * m10 - m00 * m12) * inv_det;

    for dy in 0..h {
        for dx in 0..w {
            let sx = i00 * dx as f32 + i01 * dy as f32 + i02;
            let sy = i10 * dx as f32 + i11 * dy as f32 + i12;
            let sx0 = sx.floor() as i32;
            let sy0 = sy.floor() as i32;
            let wx = sx - sx0 as f32;
            let wy = sy - sy0 as f32;
            let mut bgr = [0.0_f32; 3];
            for (ox, x_weight) in [(0, 1.0 - wx), (1, wx)] {
                for (oy, y_weight) in [(0, 1.0 - wy), (1, wy)] {
                    let x = sx0 + ox;
                    let y = sy0 + oy;
                    if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                        let weight = x_weight * y_weight;
                        let pixel = img.get_pixel(x as u32, y as u32);
                        for c in 0..3 {
                            bgr[c] += pixel[c] as f32 * weight;
                        }
                    }
                }
            }
            output.put_pixel(
                dx,
                dy,
                image::Rgb([
                    bgr[0].round().clamp(0.0, 255.0) as u8,
                    bgr[1].round().clamp(0.0, 255.0) as u8,
                    bgr[2].round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    // Transform labels
    let mut new_labels = Vec::new();
    for l in labels {
        let x1 = l.cx;
        let y1 = l.cy;
        let x2 = l.w;
        let y2 = l.h;

        // Transform 4 corners
        let corners = [(x1, y1), (x2, y2), (x1, y2), (x2, y1)];
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for (px, py) in &corners {
            let nx = m00 * px + m01 * py + m02;
            let ny = m10 * px + m11 * py + m12;
            min_x = min_x.min(nx);
            min_y = min_y.min(ny);
            max_x = max_x.max(nx);
            max_y = max_y.max(ny);
        }

        // Clip to output
        min_x = min_x.clamp(0.0, w as f32);
        min_y = min_y.clamp(0.0, h as f32);
        max_x = max_x.clamp(0.0, w as f32);
        max_y = max_y.clamp(0.0, h as f32);

        let original = [x1 * s, y1 * s, x2 * s, y2 * s];
        let transformed = [min_x, min_y, max_x, max_y];
        if candidates(&original, &transformed) {
            new_labels.push(Label {
                class: l.class,
                cx: min_x,
                cy: min_y,
                w: max_x,
                h: max_y,
            });
        }
    }

    (output, new_labels)
}

pub fn candidates(box1: &[f32; 4], box2: &[f32; 4]) -> bool {
    let w1 = box1[2] - box1[0];
    let h1 = box1[3] - box1[1];
    let w2 = box2[2] - box2[0];
    let h2 = box2[3] - box2[1];
    let aspect_ratio = (w2 / (h2 + 1e-16)).max(h2 / (w2 + 1e-16));
    w2 > 2.0 && h2 > 2.0 && (w2 * h2 / (w1 * h1 + 1e-16)) > 0.1 && aspect_ratio < 100.0
}

/// Mix-up augmentation: blend two images with random alpha.
pub fn mix_up(
    img1: &RgbImage,
    labels1: &[Label],
    img2: &RgbImage,
    labels2: &[Label],
) -> (RgbImage, Vec<Label>) {
    let beta = Beta::new(32.0, 32.0).unwrap();
    let alpha: f32 = crate::rng::sample(beta) as f32;

    assert_eq!(
        img1.dimensions(),
        img2.dimensions(),
        "operands could not be broadcast together"
    );
    let (w, h) = img1.dimensions();
    let mut output = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p1 = img1.get_pixel(x, y);
            let p2 = img2.get_pixel(x, y);
            let r = (p1[0] as f32 * alpha + p2[0] as f32 * (1.0 - alpha)) as u8;
            let g = (p1[1] as f32 * alpha + p2[1] as f32 * (1.0 - alpha)) as u8;
            let b = (p1[2] as f32 * alpha + p2[2] as f32 * (1.0 - alpha)) as u8;
            output.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }

    let mut labels = labels1.to_vec();
    labels.extend_from_slice(labels2);
    (output, labels)
}

#[cfg(test)]
mod tests {
    use super::{augment_hsv, candidates, mix_up, random_perspective};
    use crate::data::dataset::{AugmentParams, Label};
    use image::{Rgb, RgbImage};
    use rand_distr::Beta;

    #[test]
    fn random_perspective_zero_params_matches_python_border_crop_geometry() {
        let _lock = crate::rng::test_lock();
        let params = AugmentParams {
            hsv_h: 0.0,
            hsv_s: 0.0,
            hsv_v: 0.0,
            degrees: 0.0,
            translate: 0.0,
            scale: 0.0,
            shear: 0.0,
            flip_ud: 0.0,
            flip_lr: 0.0,
            mosaic: 0.0,
            mix_up: 0.0,
        };
        let mut image = RgbImage::new(8, 8);
        for pixel in image.pixels_mut() {
            *pixel = Rgb([1, 2, 3]);
        }
        let labels = vec![Label {
            class: 4.0,
            cx: 2.0,
            cy: 2.0,
            w: 6.0,
            h: 6.0,
        }];

        let (image, labels) = random_perspective(&image, &labels, &params, &[-2, -2]);

        assert_eq!(image.dimensions(), (4, 4));
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].class, 4.0);
        assert!((labels[0].cx - 0.0).abs() < 1e-6);
        assert!((labels[0].cy - 0.0).abs() < 1e-6);
        assert!((labels[0].w - 4.0).abs() < 1e-6);
        assert!((labels[0].h - 4.0).abs() < 1e-6);
    }

    #[test]
    fn random_perspective_zero_params_consumes_python_rng_draw_sites() {
        let _lock = crate::rng::test_lock();
        let params = AugmentParams {
            hsv_h: 0.0,
            hsv_s: 0.0,
            hsv_v: 0.0,
            degrees: 0.0,
            translate: 0.0,
            scale: 0.0,
            shear: 0.0,
            flip_ud: 0.0,
            flip_lr: 0.0,
            mosaic: 0.0,
            mix_up: 0.0,
        };
        let image = RgbImage::new(8, 8);

        crate::rng::set_seed(2);
        let _ = random_perspective(&image, &[], &params, &[0, 0]);
        let next_after_perspective = crate::rng::gen_f32();

        crate::rng::set_seed(2);
        for _ in 0..6 {
            let _ = crate::rng::gen_f32();
        }
        let expected_next = crate::rng::gen_f32();

        assert_eq!(next_after_perspective, expected_next);
    }

    #[test]
    fn candidates_matches_python_edge_fixture() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let cases = [
            ([0.0, 0.0, 10.0, 10.0], true),
            ([0.0, 0.0, 1.9, 10.0], false),
            ([0.0, 0.0, 10.0, 1.0], false),
            ([0.0, 0.0, 10.0, 40.0], true),
            ([0.0, 0.0, 1000.0, 10.0], false),
        ];

        for (box2, expected) in cases {
            assert_eq!(candidates(&box1, &box2), expected, "box2={box2:?}");
        }
    }

    #[test]
    fn mix_up_uses_python_beta_alpha_and_concatenates_labels() {
        let _lock = crate::rng::test_lock();
        let mut img1 = RgbImage::new(1, 1);
        img1.put_pixel(0, 0, Rgb([100, 20, 200]));
        let mut img2 = RgbImage::new(1, 1);
        img2.put_pixel(0, 0, Rgb([10, 220, 40]));
        let labels1 = vec![Label {
            class: 1.0,
            cx: 2.0,
            cy: 3.0,
            w: 4.0,
            h: 5.0,
        }];
        let labels2 = vec![Label {
            class: 2.0,
            cx: 6.0,
            cy: 7.0,
            w: 8.0,
            h: 9.0,
        }];

        crate::rng::set_seed(7);
        let alpha = crate::rng::sample(Beta::new(32.0, 32.0).unwrap()) as f32;
        let expected = [
            (100.0 * alpha + 10.0 * (1.0 - alpha)) as u8,
            (20.0 * alpha + 220.0 * (1.0 - alpha)) as u8,
            (200.0 * alpha + 40.0 * (1.0 - alpha)) as u8,
        ];

        crate::rng::set_seed(7);
        let (mixed, labels) = mix_up(&img1, &labels1, &img2, &labels2);

        assert_eq!(mixed.get_pixel(0, 0).0, expected);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].class, 1.0);
        assert_eq!(labels[1].class, 2.0);
        assert_eq!(labels[0].cx, 2.0);
        assert_eq!(labels[1].w, 8.0);
    }

    #[test]
    #[should_panic(expected = "operands could not be broadcast together")]
    fn mix_up_rejects_shape_mismatch_like_python_broadcasting() {
        let _lock = crate::rng::test_lock();
        crate::rng::set_seed(0);
        let img1 = RgbImage::new(2, 2);
        let img2 = RgbImage::new(1, 2);

        let _ = mix_up(&img1, &[], &img2, &[]);
    }

    #[test]
    fn augment_hsv_applies_value_lut_to_bgr_grayscale_like_python() {
        let params = AugmentParams {
            hsv_h: 0.0,
            hsv_s: 0.0,
            hsv_v: 0.5,
            degrees: 0.0,
            translate: 0.0,
            scale: 0.0,
            shear: 0.0,
            flip_ud: 0.0,
            flip_lr: 0.0,
            mosaic: 0.0,
            mix_up: 0.0,
        };
        let _lock = crate::rng::test_lock();
        crate::rng::set_seed(0);
        let _ = crate::rng::gen_range_f32(-1.0..1.0);
        let _ = crate::rng::gen_range_f32(-1.0..1.0);
        let rv = crate::rng::gen_range_f32(-1.0..1.0) * 0.5 + 1.0;
        let expected = (100.0_f32 * rv).clamp(0.0, 255.0) as u8;

        crate::rng::set_seed(0);
        let mut image = RgbImage::new(1, 1);
        image.put_pixel(0, 0, Rgb([100, 100, 100]));
        augment_hsv(&mut image, &params);

        assert_eq!(image.get_pixel(0, 0).0, [expected, expected, expected]);
    }

    #[test]
    fn augment_hsv_applies_saturation_lut_before_hsv_to_bgr_like_python() {
        let params = AugmentParams {
            hsv_h: 0.0,
            hsv_s: 0.5,
            hsv_v: 0.0,
            degrees: 0.0,
            translate: 0.0,
            scale: 0.0,
            shear: 0.0,
            flip_ud: 0.0,
            flip_lr: 0.0,
            mosaic: 0.0,
            mix_up: 0.0,
        };
        let _lock = crate::rng::test_lock();
        crate::rng::set_seed(1);
        let _ = crate::rng::gen_range_f32(-1.0..1.0);
        let rs = crate::rng::gen_range_f32(-1.0..1.0) * 0.5 + 1.0;
        let _ = crate::rng::gen_range_f32(-1.0..1.0);
        let s = (255.0_f32 * rs).clamp(0.0, 255.0) as u8;
        let bg = ((1.0 - s as f32 / 255.0) * 255.0) as u8;

        crate::rng::set_seed(1);
        let mut image = RgbImage::new(1, 1);
        image.put_pixel(0, 0, Rgb([0, 0, 255]));
        augment_hsv(&mut image, &params);

        assert_eq!(image.get_pixel(0, 0).0, [bg, bg, 255]);
    }

    #[test]
    fn augment_hsv_wraps_negative_hue_lut_like_numpy_modulo() {
        let _lock = crate::rng::test_lock();
        crate::rng::set_seed(11);
        let draw = crate::rng::gen_range_f32(-1.0..1.0);
        let params = AugmentParams {
            hsv_h: -2.0 / draw,
            hsv_s: 0.0,
            hsv_v: 0.0,
            degrees: 0.0,
            translate: 0.0,
            scale: 0.0,
            shear: 0.0,
            flip_ud: 0.0,
            flip_lr: 0.0,
            mosaic: 0.0,
            mix_up: 0.0,
        };

        crate::rng::set_seed(11);
        let mut image = RgbImage::new(1, 1);
        image.put_pixel(0, 0, Rgb([0, 255, 0]));
        augment_hsv(&mut image, &params);

        assert_eq!(image.get_pixel(0, 0).0, [255, 0, 0]);
    }
}
