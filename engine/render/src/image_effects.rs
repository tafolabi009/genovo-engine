// engine/render/src/image_effects.rs
//
// Image processing effects on GPU (CPU reference) for the Genovo engine.
//
// Provides a collection of image processing operations:
//
// - **Gaussian blur** — Separable horizontal + vertical passes.
// - **Box blur** — Fast box-filter blur.
// - **Kawase blur** — Iterative dual-filter blur (fast approximation).
// - **Bilateral filter** — Edge-preserving blur.
// - **Sharpen** — Unsharp mask sharpening.
// - **Edge detection** — Sobel and Canny operators.
// - **Image resize** — Bilinear, bicubic, and Lanczos filters.
// - **Histogram computation** — Per-channel histogram.
// - **Auto-levels** — Stretch histogram to full range.
// - **Colour quantization** — Reduce colour palette.
//
// All functions operate on flat pixel buffers (Vec<[f32; 3]> or similar).
// In production, these would run as compute shaders or fragment passes.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Gaussian blur
// ---------------------------------------------------------------------------

/// Compute a 1D Gaussian kernel.
///
/// # Arguments
/// * `radius` — Kernel half-size in pixels (total size = 2*radius + 1).
/// * `sigma` — Standard deviation. If 0, automatically computed from radius.
pub fn gaussian_kernel(radius: u32, sigma: f32) -> Vec<f32> {
    let sigma = if sigma <= 0.0 { radius as f32 / 3.0 + 0.5 } else { sigma };
    let size = (radius * 2 + 1) as usize;
    let mut kernel = vec![0.0f32; size];
    let mut sum = 0.0f32;

    for i in 0..size {
        let x = i as f32 - radius as f32;
        let g = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel[i] = g;
        sum += g;
    }

    // Normalise.
    if sum > 0.0 {
        for k in &mut kernel {
            *k /= sum;
        }
    }

    kernel
}

/// Apply a separable Gaussian blur to an RGB buffer.
///
/// # Arguments
/// * `src` — Source buffer (width * height pixels, RGB float).
/// * `width`, `height` — Dimensions.
/// * `radius` — Blur radius.
/// * `sigma` — Gaussian sigma (0 = auto).
pub fn gaussian_blur(
    src: &[[f32; 3]],
    width: u32,
    height: u32,
    radius: u32,
    sigma: f32,
) -> Vec<[f32; 3]> {
    let kernel = gaussian_kernel(radius, sigma);
    let r = radius as i32;

    // Horizontal pass.
    let mut temp = vec![[0.0f32; 3]; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0f32; 3];
            for k in -r..=r {
                let sx = (x as i32 + k).clamp(0, width as i32 - 1) as u32;
                let pixel = src[(y * width + sx) as usize];
                let w = kernel[(k + r) as usize];
                acc[0] += pixel[0] * w;
                acc[1] += pixel[1] * w;
                acc[2] += pixel[2] * w;
            }
            temp[(y * width + x) as usize] = acc;
        }
    }

    // Vertical pass.
    let mut dst = vec![[0.0f32; 3]; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0f32; 3];
            for k in -r..=r {
                let sy = (y as i32 + k).clamp(0, height as i32 - 1) as u32;
                let pixel = temp[(sy * width + x) as usize];
                let w = kernel[(k + r) as usize];
                acc[0] += pixel[0] * w;
                acc[1] += pixel[1] * w;
                acc[2] += pixel[2] * w;
            }
            dst[(y * width + x) as usize] = acc;
        }
    }

    dst
}

// ---------------------------------------------------------------------------
// Box blur
// ---------------------------------------------------------------------------

/// Apply a box blur to an RGB buffer (separable, two passes).
pub fn box_blur(
    src: &[[f32; 3]],
    width: u32,
    height: u32,
    radius: u32,
) -> Vec<[f32; 3]> {
    let r = radius as i32;
    let kernel_size = (2 * r + 1) as f32;
    let inv_k = 1.0 / kernel_size;

    // Horizontal pass.
    let mut temp = vec![[0.0f32; 3]; (width * height) as usize];
    for y in 0..height {
        // Sliding window sum.
        let mut sum = [0.0f32; 3];
        // Initialise window.
        for k in 0..=r {
            let sx = k.clamp(0, width as i32 - 1) as u32;
            let pixel = src[(y * width + sx) as usize];
            sum[0] += pixel[0];
            sum[1] += pixel[1];
            sum[2] += pixel[2];
            if k > 0 {
                // Mirror for left side.
                sum[0] += pixel[0];
                sum[1] += pixel[1];
                sum[2] += pixel[2];
            }
        }
        temp[(y * width) as usize] = [sum[0] * inv_k, sum[1] * inv_k, sum[2] * inv_k];

        for x in 1..width {
            let add_x = (x as i32 + r).clamp(0, width as i32 - 1) as u32;
            let rem_x = (x as i32 - r - 1).clamp(0, width as i32 - 1) as u32;
            let add_pixel = src[(y * width + add_x) as usize];
            let rem_pixel = src[(y * width + rem_x) as usize];
            sum[0] += add_pixel[0] - rem_pixel[0];
            sum[1] += add_pixel[1] - rem_pixel[1];
            sum[2] += add_pixel[2] - rem_pixel[2];
            temp[(y * width + x) as usize] = [sum[0] * inv_k, sum[1] * inv_k, sum[2] * inv_k];
        }
    }

    // Vertical pass.
    let mut dst = vec![[0.0f32; 3]; (width * height) as usize];
    for x in 0..width {
        let mut sum = [0.0f32; 3];
        for k in 0..=r {
            let sy = k.clamp(0, height as i32 - 1) as u32;
            let pixel = temp[(sy * width + x) as usize];
            sum[0] += pixel[0];
            sum[1] += pixel[1];
            sum[2] += pixel[2];
            if k > 0 {
                sum[0] += pixel[0];
                sum[1] += pixel[1];
                sum[2] += pixel[2];
            }
        }
        dst[x as usize] = [sum[0] * inv_k, sum[1] * inv_k, sum[2] * inv_k];

        for y in 1..height {
            let add_y = (y as i32 + r).clamp(0, height as i32 - 1) as u32;
            let rem_y = (y as i32 - r - 1).clamp(0, height as i32 - 1) as u32;
            let add_pixel = temp[(add_y * width + x) as usize];
            let rem_pixel = temp[(rem_y * width + x) as usize];
            sum[0] += add_pixel[0] - rem_pixel[0];
            sum[1] += add_pixel[1] - rem_pixel[1];
            sum[2] += add_pixel[2] - rem_pixel[2];
            dst[(y * width + x) as usize] = [sum[0] * inv_k, sum[1] * inv_k, sum[2] * inv_k];
        }
    }

    dst
}

// ---------------------------------------------------------------------------
// Kawase blur
// ---------------------------------------------------------------------------

/// Apply a single Kawase blur iteration.
///
/// Samples 4 diagonal neighbours at a given offset.
pub fn kawase_blur_pass(
    src: &[[f32; 3]],
    width: u32,
    height: u32,
    offset: f32,
) -> Vec<[f32; 3]> {
    let mut dst = vec![[0.0f32; 3]; (width * height) as usize];
    let off = offset as i32;

    for y in 0..height {
        for x in 0..width {
            let sample = |dx: i32, dy: i32| -> [f32; 3] {
                let sx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                let sy = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                src[(sy * width + sx) as usize]
            };

            let c = sample(0, 0);
            let tl = sample(-off, -off);
            let tr = sample(off, -off);
            let bl = sample(-off, off);
            let br = sample(off, off);

            dst[(y * width + x) as usize] = [
                (c[0] + tl[0] + tr[0] + bl[0] + br[0]) * 0.2,
                (c[1] + tl[1] + tr[1] + bl[1] + br[1]) * 0.2,
                (c[2] + tl[2] + tr[2] + bl[2] + br[2]) * 0.2,
            ];
        }
    }

    dst
}

/// Apply iterative Kawase blur (multiple passes with increasing offset).
pub fn kawase_blur(
    src: &[[f32; 3]],
    width: u32,
    height: u32,
    iterations: u32,
) -> Vec<[f32; 3]> {
    let mut current = src.to_vec();
    for i in 0..iterations {
        let offset = i as f32 + 0.5;
        current = kawase_blur_pass(&current, width, height, offset);
    }
    current
}

// ---------------------------------------------------------------------------
// Bilateral filter
// ---------------------------------------------------------------------------

/// Apply a bilateral filter (edge-preserving blur).
///
/// # Arguments
/// * `src` — Source buffer.
/// * `width`, `height` — Dimensions.
/// * `spatial_sigma` — Spatial sigma (controls blur radius).
/// * `range_sigma` — Range sigma (controls edge sensitivity).
pub fn bilateral_filter(
    src: &[[f32; 3]],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
) -> Vec<[f32; 3]> {
    let radius = (spatial_sigma * 2.0).ceil() as i32;
    let inv_spatial = -0.5 / (spatial_sigma * spatial_sigma);
    let inv_range = -0.5 / (range_sigma * range_sigma);

    let mut dst = vec![[0.0f32; 3]; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let center = src[(y * width + x) as usize];
            let mut acc = [0.0f32; 3];
            let mut weight_sum = 0.0f32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let sx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let sy = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                    let neighbour = src[(sy * width + sx) as usize];

                    // Spatial weight.
                    let dist_sq = (dx * dx + dy * dy) as f32;
                    let ws = (dist_sq * inv_spatial).exp();

                    // Range weight (colour difference).
                    let dr = center[0] - neighbour[0];
                    let dg = center[1] - neighbour[1];
                    let db = center[2] - neighbour[2];
                    let range_sq = dr * dr + dg * dg + db * db;
                    let wr = (range_sq * inv_range).exp();

                    let w = ws * wr;
                    acc[0] += neighbour[0] * w;
                    acc[1] += neighbour[1] * w;
                    acc[2] += neighbour[2] * w;
                    weight_sum += w;
                }
            }

            if weight_sum > 0.0 {
                let inv = 1.0 / weight_sum;
                dst[(y * width + x) as usize] = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
            } else {
                dst[(y * width + x) as usize] = center;
            }
        }
    }

    dst
}

// ---------------------------------------------------------------------------
// Sharpen (unsharp mask)
// ---------------------------------------------------------------------------

/// Apply unsharp mask sharpening.
///
/// # Arguments
/// * `src` — Source buffer.
/// * `width`, `height` — Dimensions.
/// * `radius` — Blur radius for the unsharp mask.
/// * `amount` — Sharpening strength (1.0 = normal, >1.0 = stronger).
/// * `threshold` — Minimum difference to sharpen (prevents noise amplification).
pub fn unsharp_mask(
    src: &[[f32; 3]],
    width: u32,
    height: u32,
    radius: u32,
    amount: f32,
    threshold: f32,
) -> Vec<[f32; 3]> {
    let blurred = gaussian_blur(src, width, height, radius, 0.0);
    let mut dst = vec![[0.0f32; 3]; (width * height) as usize];

    for i in 0..src.len() {
        let orig = src[i];
        let blur = blurred[i];
        let diff = [
            orig[0] - blur[0],
            orig[1] - blur[1],
            orig[2] - blur[2],
        ];

        // Apply threshold.
        let diff_len = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
        if diff_len < threshold {
            dst[i] = orig;
        } else {
            dst[i] = [
                (orig[0] + diff[0] * amount).max(0.0),
                (orig[1] + diff[1] * amount).max(0.0),
                (orig[2] + diff[2] * amount).max(0.0),
            ];
        }
    }

    dst
}

// ---------------------------------------------------------------------------
// Edge detection
// ---------------------------------------------------------------------------

/// Sobel edge detection.
///
/// Returns the edge magnitude per pixel (greyscale).
pub fn sobel_edge(src: &[[f32; 3]], width: u32, height: u32) -> Vec<f32> {
    let mut dst = vec![0.0f32; (width * height) as usize];

    let lum = |pixel: [f32; 3]| -> f32 {
        0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
    };

    let sample = |x: i32, y: i32| -> f32 {
        let cx = x.clamp(0, width as i32 - 1) as u32;
        let cy = y.clamp(0, height as i32 - 1) as u32;
        lum(src[(cy * width + cx) as usize])
    };

    for y in 0..height {
        for x in 0..width {
            let xi = x as i32;
            let yi = y as i32;

            // Sobel X kernel.
            let gx = -sample(xi - 1, yi - 1) + sample(xi + 1, yi - 1)
                - 2.0 * sample(xi - 1, yi) + 2.0 * sample(xi + 1, yi)
                - sample(xi - 1, yi + 1) + sample(xi + 1, yi + 1);

            // Sobel Y kernel.
            let gy = -sample(xi - 1, yi - 1) - 2.0 * sample(xi, yi - 1) - sample(xi + 1, yi - 1)
                + sample(xi - 1, yi + 1) + 2.0 * sample(xi, yi + 1) + sample(xi + 1, yi + 1);

            dst[(y * width + x) as usize] = (gx * gx + gy * gy).sqrt();
        }
    }

    dst
}

/// Canny edge detection (simplified: Sobel + non-maximum suppression + hysteresis).
///
/// # Arguments
/// * `src` — Source buffer.
/// * `width`, `height` — Dimensions.
/// * `low_threshold` — Lower hysteresis threshold.
/// * `high_threshold` — Upper hysteresis threshold.
pub fn canny_edge(
    src: &[[f32; 3]],
    width: u32,
    height: u32,
    low_threshold: f32,
    high_threshold: f32,
) -> Vec<f32> {
    // Step 1: Blur to reduce noise.
    let blurred = gaussian_blur(src, width, height, 1, 1.0);

    // Step 2: Compute gradient magnitude and direction.
    let lum = |pixel: [f32; 3]| -> f32 {
        0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
    };

    let sample = |x: i32, y: i32| -> f32 {
        let cx = x.clamp(0, width as i32 - 1) as u32;
        let cy = y.clamp(0, height as i32 - 1) as u32;
        lum(blurred[(cy * width + cx) as usize])
    };

    let total = (width * height) as usize;
    let mut magnitude = vec![0.0f32; total];
    let mut direction = vec![0.0f32; total];

    for y in 0..height {
        for x in 0..width {
            let xi = x as i32;
            let yi = y as i32;

            let gx = -sample(xi - 1, yi - 1) + sample(xi + 1, yi - 1)
                - 2.0 * sample(xi - 1, yi) + 2.0 * sample(xi + 1, yi)
                - sample(xi - 1, yi + 1) + sample(xi + 1, yi + 1);

            let gy = -sample(xi - 1, yi - 1) - 2.0 * sample(xi, yi - 1) - sample(xi + 1, yi - 1)
                + sample(xi - 1, yi + 1) + 2.0 * sample(xi, yi + 1) + sample(xi + 1, yi + 1);

            let idx = (y * width + x) as usize;
            magnitude[idx] = (gx * gx + gy * gy).sqrt();
            direction[idx] = gy.atan2(gx);
        }
    }

    // Step 3: Non-maximum suppression.
    let mut suppressed = vec![0.0f32; total];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            let angle = direction[idx];
            let mag = magnitude[idx];

            // Quantise angle to 4 directions.
            let a = ((angle.to_degrees() + 180.0) % 180.0).abs();
            let (dx1, dy1, dx2, dy2) = if a < 22.5 || a >= 157.5 {
                (1i32, 0i32, -1i32, 0i32)
            } else if a < 67.5 {
                (1, 1, -1, -1)
            } else if a < 112.5 {
                (0, 1, 0, -1)
            } else {
                (-1, 1, 1, -1)
            };

            let n1 = magnitude[((y as i32 + dy1) as u32 * width + (x as i32 + dx1) as u32) as usize];
            let n2 = magnitude[((y as i32 + dy2) as u32 * width + (x as i32 + dx2) as u32) as usize];

            if mag >= n1 && mag >= n2 {
                suppressed[idx] = mag;
            }
        }
    }

    // Step 4: Hysteresis thresholding.
    let mut result = vec![0.0f32; total];
    for i in 0..total {
        if suppressed[i] >= high_threshold {
            result[i] = 1.0;
        } else if suppressed[i] >= low_threshold {
            // Check 8-connected neighbours for strong edges.
            let x = (i % width as usize) as i32;
            let y = (i / width as usize) as i32;
            let mut has_strong = false;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = (x + dx).clamp(0, width as i32 - 1) as u32;
                    let ny = (y + dy).clamp(0, height as i32 - 1) as u32;
                    if suppressed[(ny * width + nx) as usize] >= high_threshold {
                        has_strong = true;
                    }
                }
            }
            if has_strong {
                result[i] = 1.0;
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Image resize
// ---------------------------------------------------------------------------

/// Resize mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeFilter {
    /// Nearest neighbour (no filtering).
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
    /// Bicubic interpolation (Mitchell-Netravali).
    Bicubic,
    /// Lanczos-3 filter.
    Lanczos3,
}

/// Resize an RGB buffer.
pub fn resize_image(
    src: &[[f32; 3]],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    filter: ResizeFilter,
) -> Vec<[f32; 3]> {
    let mut dst = vec![[0.0f32; 3]; (dst_w * dst_h) as usize];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (dx as f32 + 0.5) * src_w as f32 / dst_w as f32 - 0.5;
            let sy = (dy as f32 + 0.5) * src_h as f32 / dst_h as f32 - 0.5;

            dst[(dy * dst_w + dx) as usize] = match filter {
                ResizeFilter::Nearest => sample_nearest(src, src_w, src_h, sx, sy),
                ResizeFilter::Bilinear => sample_bilinear(src, src_w, src_h, sx, sy),
                ResizeFilter::Bicubic => sample_bicubic(src, src_w, src_h, sx, sy),
                ResizeFilter::Lanczos3 => sample_lanczos3(src, src_w, src_h, sx, sy),
            };
        }
    }

    dst
}

fn sample_nearest(src: &[[f32; 3]], w: u32, h: u32, x: f32, y: f32) -> [f32; 3] {
    let ix = (x.round() as u32).min(w - 1);
    let iy = (y.round() as u32).min(h - 1);
    src[(iy * w + ix) as usize]
}

fn sample_bilinear(src: &[[f32; 3]], w: u32, h: u32, x: f32, y: f32) -> [f32; 3] {
    let x0 = x.floor().max(0.0) as u32;
    let y0 = y.floor().max(0.0) as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = src[(y0 * w + x0) as usize];
    let p10 = src[(y0 * w + x1) as usize];
    let p01 = src[(y1 * w + x0) as usize];
    let p11 = src[(y1 * w + x1) as usize];

    [
        bilerp(p00[0], p10[0], p01[0], p11[0], fx, fy),
        bilerp(p00[1], p10[1], p01[1], p11[1], fx, fy),
        bilerp(p00[2], p10[2], p01[2], p11[2], fx, fy),
    ]
}

fn sample_bicubic(src: &[[f32; 3]], w: u32, h: u32, x: f32, y: f32) -> [f32; 3] {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let cubic = |t: f32| -> [f32; 4] {
        // Mitchell-Netravali (B=1/3, C=1/3).
        let b = 1.0 / 3.0;
        let c = 1.0 / 3.0;

        let mut w = [0.0f32; 4];
        for i in 0..4 {
            let x = (i as f32 - 1.0 - t).abs();
            if x < 1.0 {
                w[i] = ((12.0 - 9.0 * b - 6.0 * c) * x * x * x
                    + (-18.0 + 12.0 * b + 6.0 * c) * x * x
                    + (6.0 - 2.0 * b)) / 6.0;
            } else if x < 2.0 {
                w[i] = ((-b - 6.0 * c) * x * x * x
                    + (6.0 * b + 30.0 * c) * x * x
                    + (-12.0 * b - 48.0 * c) * x
                    + (8.0 * b + 24.0 * c)) / 6.0;
            }
        }
        w
    };

    let wx = cubic(fx);
    let wy = cubic(fy);

    let mut acc = [0.0f32; 3];
    let mut weight_sum = 0.0f32;

    for j in 0..4 {
        for i in 0..4 {
            let sx = (ix + i as i32 - 1).clamp(0, w as i32 - 1) as u32;
            let sy = (iy + j as i32 - 1).clamp(0, h as i32 - 1) as u32;
            let pixel = src[(sy * w + sx) as usize];
            let weight = wx[i] * wy[j];
            acc[0] += pixel[0] * weight;
            acc[1] += pixel[1] * weight;
            acc[2] += pixel[2] * weight;
            weight_sum += weight;
        }
    }

    if weight_sum.abs() > 1e-10 {
        let inv = 1.0 / weight_sum;
        [acc[0] * inv, acc[1] * inv, acc[2] * inv]
    } else {
        sample_bilinear(src, w, h, x, y)
    }
}

fn sample_lanczos3(src: &[[f32; 3]], w: u32, h: u32, x: f32, y: f32) -> [f32; 3] {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let lanczos = |t: f32, a: f32| -> f32 {
        if t.abs() < 1e-6 {
            1.0
        } else if t.abs() >= a {
            0.0
        } else {
            let pt = PI * t;
            let pta = pt / a;
            (pt.sin() / pt) * (pta.sin() / pta)
        }
    };

    let a = 3.0;
    let mut acc = [0.0f32; 3];
    let mut weight_sum = 0.0f32;

    for j in -2..=3 {
        for i in -2..=3 {
            let sx = (ix + i).clamp(0, w as i32 - 1) as u32;
            let sy = (iy + j).clamp(0, h as i32 - 1) as u32;
            let pixel = src[(sy * w + sx) as usize];
            let weight = lanczos(i as f32 - fx, a) * lanczos(j as f32 - fy, a);
            acc[0] += pixel[0] * weight;
            acc[1] += pixel[1] * weight;
            acc[2] += pixel[2] * weight;
            weight_sum += weight;
        }
    }

    if weight_sum.abs() > 1e-10 {
        let inv = 1.0 / weight_sum;
        [acc[0] * inv, acc[1] * inv, acc[2] * inv]
    } else {
        [0.0; 3]
    }
}

#[inline]
fn bilerp(a: f32, b: f32, c: f32, d: f32, fx: f32, fy: f32) -> f32 {
    let top = a + (b - a) * fx;
    let bot = c + (d - c) * fx;
    top + (bot - top) * fy
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

/// Per-channel histogram (256 bins per channel).
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Red channel histogram.
    pub r: [u32; 256],
    /// Green channel histogram.
    pub g: [u32; 256],
    /// Blue channel histogram.
    pub b: [u32; 256],
    /// Luminance histogram.
    pub lum: [u32; 256],
    /// Total pixel count.
    pub total_pixels: u32,
}

impl Histogram {
    /// Compute a histogram from an RGB buffer (values assumed in [0, 1]).
    pub fn compute(src: &[[f32; 3]], _width: u32, _height: u32) -> Self {
        let mut hist = Self {
            r: [0; 256],
            g: [0; 256],
            b: [0; 256],
            lum: [0; 256],
            total_pixels: src.len() as u32,
        };

        for pixel in src {
            let ri = (pixel[0].clamp(0.0, 1.0) * 255.0) as usize;
            let gi = (pixel[1].clamp(0.0, 1.0) * 255.0) as usize;
            let bi = (pixel[2].clamp(0.0, 1.0) * 255.0) as usize;
            let l = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
            let li = (l.clamp(0.0, 1.0) * 255.0) as usize;

            hist.r[ri.min(255)] += 1;
            hist.g[gi.min(255)] += 1;
            hist.b[bi.min(255)] += 1;
            hist.lum[li.min(255)] += 1;
        }

        hist
    }

    /// Find the percentile value for a channel.
    fn percentile(channel: &[u32; 256], total: u32, p: f32) -> u8 {
        let target = (total as f32 * p) as u32;
        let mut acc = 0u32;
        for (i, &count) in channel.iter().enumerate() {
            acc += count;
            if acc >= target {
                return i as u8;
            }
        }
        255
    }

    /// Get the black point and white point for auto-levels.
    pub fn auto_levels_range(&self, clip_percent: f32) -> (f32, f32) {
        let low = Self::percentile(&self.lum, self.total_pixels, clip_percent);
        let high = Self::percentile(&self.lum, self.total_pixels, 1.0 - clip_percent);
        (low as f32 / 255.0, high as f32 / 255.0)
    }
}

/// Apply auto-levels to an RGB buffer.
///
/// Stretches the histogram so that the darkest `clip_percent` becomes black
/// and the brightest `clip_percent` becomes white.
pub fn auto_levels(
    src: &[[f32; 3]],
    width: u32,
    height: u32,
    clip_percent: f32,
) -> Vec<[f32; 3]> {
    let hist = Histogram::compute(src, width, height);
    let (low, high) = hist.auto_levels_range(clip_percent);

    let range = (high - low).max(0.001);
    let inv_range = 1.0 / range;

    src.iter()
        .map(|pixel| {
            [
                ((pixel[0] - low) * inv_range).clamp(0.0, 1.0),
                ((pixel[1] - low) * inv_range).clamp(0.0, 1.0),
                ((pixel[2] - low) * inv_range).clamp(0.0, 1.0),
            ]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Colour quantization
// ---------------------------------------------------------------------------

/// Quantize an RGB buffer to a reduced number of colours using uniform quantization.
///
/// # Arguments
/// * `src` — Source buffer.
/// * `levels` — Number of levels per channel (e.g. 8 for 512 total colours).
pub fn color_quantize(src: &[[f32; 3]], levels: u32) -> Vec<[f32; 3]> {
    let levels_f = levels as f32;
    let inv = 1.0 / (levels_f - 1.0).max(1.0);

    src.iter()
        .map(|pixel| {
            [
                ((pixel[0] * (levels_f - 1.0)).round() * inv).clamp(0.0, 1.0),
                ((pixel[1] * (levels_f - 1.0)).round() * inv).clamp(0.0, 1.0),
                ((pixel[2] * (levels_f - 1.0)).round() * inv).clamp(0.0, 1.0),
            ]
        })
        .collect()
}

/// Floyd-Steinberg dithered quantization.
pub fn color_quantize_dithered(
    src: &[[f32; 3]],
    width: u32,
    height: u32,
    levels: u32,
) -> Vec<[f32; 3]> {
    let mut buf: Vec<[f32; 3]> = src.to_vec();
    let levels_f = levels as f32;
    let inv = 1.0 / (levels_f - 1.0).max(1.0);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let old = buf[idx];
            let new = [
                ((old[0] * (levels_f - 1.0)).round() * inv).clamp(0.0, 1.0),
                ((old[1] * (levels_f - 1.0)).round() * inv).clamp(0.0, 1.0),
                ((old[2] * (levels_f - 1.0)).round() * inv).clamp(0.0, 1.0),
            ];
            let err = [old[0] - new[0], old[1] - new[1], old[2] - new[2]];
            buf[idx] = new;

            // Distribute error to neighbours.
            let distribute = |bx: u32, by: u32, factor: f32, buf: &mut Vec<[f32; 3]>| {
                let i = (by * width + bx) as usize;
                if i < buf.len() {
                    buf[i][0] += err[0] * factor;
                    buf[i][1] += err[1] * factor;
                    buf[i][2] += err[2] * factor;
                }
            };

            if x + 1 < width {
                distribute(x + 1, y, 7.0 / 16.0, &mut buf);
            }
            if y + 1 < height {
                if x > 0 {
                    distribute(x - 1, y + 1, 3.0 / 16.0, &mut buf);
                }
                distribute(x, y + 1, 5.0 / 16.0, &mut buf);
                if x + 1 < width {
                    distribute(x + 1, y + 1, 1.0 / 16.0, &mut buf);
                }
            }
        }
    }

    buf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel() {
        let kernel = gaussian_kernel(3, 1.0);
        assert_eq!(kernel.len(), 7);
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gaussian_blur() {
        let src = vec![[0.5f32; 3]; 16 * 16];
        let dst = gaussian_blur(&src, 16, 16, 2, 1.0);
        assert_eq!(dst.len(), 16 * 16);
        // Uniform input → uniform output.
        assert!((dst[0][0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_sobel_edge() {
        // Create a simple edge: left half black, right half white.
        let mut src = vec![[0.0f32; 3]; 16 * 16];
        for y in 0..16 {
            for x in 8..16 {
                src[y * 16 + x] = [1.0, 1.0, 1.0];
            }
        }
        let edges = sobel_edge(&src, 16, 16);
        // Edge should be strongest near x=8.
        assert!(edges[8 * 16 + 8] > 0.0);
    }

    #[test]
    fn test_resize() {
        let src = vec![[0.5f32; 3]; 32 * 32];
        let dst = resize_image(&src, 32, 32, 16, 16, ResizeFilter::Bilinear);
        assert_eq!(dst.len(), 16 * 16);
        assert!((dst[0][0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_histogram() {
        let src = vec![[0.5f32; 3]; 100];
        let hist = Histogram::compute(&src, 10, 10);
        assert_eq!(hist.total_pixels, 100);
        assert_eq!(hist.r[128], 100); // 0.5 * 255 ≈ 128.
    }

    #[test]
    fn test_quantize() {
        let src = vec![[0.33f32, 0.66, 0.99]; 4];
        let quantized = color_quantize(&src, 4);
        // Each channel should be quantized to 0.0, 0.33, 0.67, or 1.0.
        for pixel in &quantized {
            for c in pixel {
                assert!((*c - 0.0).abs() < 0.02 || (*c - 0.333).abs() < 0.02
                    || (*c - 0.667).abs() < 0.02 || (*c - 1.0).abs() < 0.02);
            }
        }
    }
}
