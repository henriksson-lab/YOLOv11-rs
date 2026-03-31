use image::RgbImage;
use rand::Rng;

use crate::data::dataset::Label;

/// HSV color-space augmentation.
pub fn augment_hsv(img: &mut RgbImage, h_gain: f32, s_gain: f32, v_gain: f32) {
    let mut rng = rand::thread_rng();
    let rh = rng.gen_range(-1.0..1.0) * h_gain + 1.0;
    let rs = rng.gen_range(-1.0..1.0) * s_gain + 1.0;
    let rv = rng.gen_range(-1.0..1.0) * v_gain + 1.0;

    for pixel in img.pixels_mut() {
        let [r, g, b] = pixel.0;
        let (h, s, v) = rgb_to_hsv(r, g, b);
        let h2 = ((h as f32 * rh) % 180.0) as u8;
        let s2 = (s as f32 * rs).clamp(0.0, 255.0) as u8;
        let v2 = (v as f32 * rv).clamp(0.0, 255.0) as u8;
        let (r2, g2, b2) = hsv_to_rgb(h2, s2, v2);
        pixel.0 = [r2, g2, b2];
    }
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
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
}

fn hsv_to_rgb(h: u8, s: u8, v: u8) -> (u8, u8, u8) {
    let h = h as f32 * 2.0; // 0..360
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
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Apply random perspective (affine) transformation.
/// `border` offsets for mosaic (typically [-input_size/2, -input_size/2]).
pub fn random_perspective(
    img: &RgbImage,
    labels: &[Label],
    _target_size: u32,
    degrees: f32,
    translate: f32,
    scale: f32,
    shear: f32,
    border: &[i32; 2],
) -> (RgbImage, Vec<Label>) {
    let mut rng = rand::thread_rng();
    let h = img.height() as f32 + border[0] as f32 * 2.0;
    let w = img.width() as f32 + border[1] as f32 * 2.0;
    let h = h as u32;
    let w = w as u32;

    // Build affine matrix components (guard against empty ranges when param is 0)
    let angle = if degrees > 0.0 { rng.gen_range(-degrees..degrees) } else { 0.0 };
    let s = if scale > 0.0 { rng.gen_range(1.0 - scale..1.0 + scale) } else { 1.0 };
    let sh_x = if shear > 0.0 { (rng.gen_range(-shear..shear) * std::f32::consts::PI / 180.0).tan() } else { 0.0 };
    let sh_y = if shear > 0.0 { (rng.gen_range(-shear..shear) * std::f32::consts::PI / 180.0).tan() } else { 0.0 };
    let tx = if translate > 0.0 { rng.gen_range(0.5 - translate..0.5 + translate) * w as f32 } else { 0.5 * w as f32 };
    let ty = if translate > 0.0 { rng.gen_range(0.5 - translate..0.5 + translate) * h as f32 } else { 0.5 * h as f32 };

    let cos_a = (angle * std::f32::consts::PI / 180.0).cos() * s;
    let sin_a = (angle * std::f32::consts::PI / 180.0).sin() * s;
    let cx = img.width() as f32 / 2.0;
    let cy = img.height() as f32 / 2.0;

    // Combined matrix: translate * shear * rotate * center
    // M = [[cos*s + shx*sin*s, -sin*s + shx*cos*s, tx - cx*(cos*s+shx*sin*s) + cy*(sin*s-shx*cos*s)],
    //      [shy*cos*s + sin*s, -shy*sin*s + cos*s, ty - cx*(shy*cos*s+sin*s) + cy*(shy*sin*s-cos*s)]]
    let m00 = cos_a + sh_x * sin_a;
    let m01 = -sin_a + sh_x * cos_a;
    let m02 = tx - cx * m00 + cy * (-m01);
    let m10 = sh_y * cos_a + sin_a;
    let m11 = -sh_y * sin_a + cos_a;
    let m12 = ty - cx * m10 + cy * (-m11);

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
            let sx_i = sx.floor() as i32;
            let sy_i = sy.floor() as i32;
            if sx_i >= 0
                && sx_i < img.width() as i32
                && sy_i >= 0
                && sy_i < img.height() as i32
            {
                let px = img.get_pixel(sx_i as u32, sy_i as u32);
                output.put_pixel(dx, dy, *px);
            }
        }
    }

    // Transform labels
    let mut new_labels = Vec::new();
    for l in labels {
        // Convert to corner points
        let x1 = (l.cx - l.w / 2.0) * img.width() as f32;
        let y1 = (l.cy - l.h / 2.0) * img.height() as f32;
        let x2 = (l.cx + l.w / 2.0) * img.width() as f32;
        let y2 = (l.cy + l.h / 2.0) * img.height() as f32;

        // Transform 4 corners
        let corners = [
            (x1, y1),
            (x2, y2),
            (x1, y2),
            (x2, y1),
        ];
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

        let nw = max_x - min_x;
        let nh = max_y - min_y;
        if nw > 2.0 && nh > 2.0 {
            new_labels.push(Label {
                class: l.class,
                cx: (min_x + max_x) / 2.0 / w as f32,
                cy: (min_y + max_y) / 2.0 / h as f32,
                w: nw / w as f32,
                h: nh / h as f32,
            });
        }
    }

    (output, new_labels)
}

/// Mix-up augmentation: blend two images with random alpha.
pub fn mix_up(
    img1: &RgbImage,
    labels1: &[Label],
    img2: &RgbImage,
    labels2: &[Label],
) -> (RgbImage, Vec<Label>) {
    let mut rng = rand::thread_rng();
    // Beta(32, 32) approximation: mean=0.5, concentrated
    let alpha: f32 = rng.gen_range(0.4..0.6);

    let (w, h) = (img1.width().min(img2.width()), img1.height().min(img2.height()));
    let mut output = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p1 = img1.get_pixel(x.min(img1.width() - 1), y.min(img1.height() - 1));
            let p2 = img2.get_pixel(x.min(img2.width() - 1), y.min(img2.height() - 1));
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
