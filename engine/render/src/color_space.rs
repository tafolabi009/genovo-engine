// engine/render/src/color_space.rs
//
// Colour space conversions for the Genovo engine.
//
// Provides accurate conversions between a wide range of colour spaces used
// in HDR rendering, colour grading, and display output:
//
// - **sRGB ↔ linear** — The fundamental gamma conversion.
// - **Rec.709** — The colour primaries of sRGB/HDTV.
// - **Rec.2020** — Wide-gamut UHDTV primaries.
// - **DCI-P3** — Digital cinema projection gamut.
// - **ACES AP0 / AP1** — Academy Color Encoding System.
// - **Oklab** — Perceptually uniform colour space (Björn Ottosson).
// - **CIE XYZ** — The reference colorimetric space.
// - **CIE LAB** — Perceptual lightness-chroma model.
// - **LCH** — Cylindrical form of CIE LAB.
// - **Colour temperature** — Planckian locus approximation (CCT to chromaticity).
// - **Chromatic adaptation** — Bradford and Von Kries transforms.
// - **Gamut mapping** — Clamp out-of-gamut colours to a target gamut.
// - **Colour difference** — ΔE*2000 (CIEDE2000) metric.
//
// All conversions use f32 precision. Matrices are stored in row-major order.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// sRGB ↔ linear
// ---------------------------------------------------------------------------

/// Convert a single channel from sRGB gamma to linear.
#[inline]
pub fn srgb_to_linear(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert a single channel from linear to sRGB gamma.
#[inline]
pub fn linear_to_srgb(l: f32) -> f32 {
    if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert RGB from sRGB gamma to linear.
pub fn srgb_to_linear_rgb(srgb: [f32; 3]) -> [f32; 3] {
    [
        srgb_to_linear(srgb[0]),
        srgb_to_linear(srgb[1]),
        srgb_to_linear(srgb[2]),
    ]
}

/// Convert RGB from linear to sRGB gamma.
pub fn linear_to_srgb_rgb(linear: [f32; 3]) -> [f32; 3] {
    [
        linear_to_srgb(linear[0]),
        linear_to_srgb(linear[1]),
        linear_to_srgb(linear[2]),
    ]
}

// ---------------------------------------------------------------------------
// CIE XYZ
// ---------------------------------------------------------------------------

// 3x3 matrix-vector multiply (row-major).
fn mat3_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// sRGB (Rec.709 primaries) linear RGB to CIE XYZ (D65).
pub const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

/// CIE XYZ (D65) to sRGB (Rec.709) linear RGB.
pub const XYZ_TO_SRGB: [[f32; 3]; 3] = [
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
];

/// Convert linear sRGB to CIE XYZ (D65).
pub fn linear_srgb_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    mat3_mul(&SRGB_TO_XYZ, rgb)
}

/// Convert CIE XYZ (D65) to linear sRGB.
pub fn xyz_to_linear_srgb(xyz: [f32; 3]) -> [f32; 3] {
    mat3_mul(&XYZ_TO_SRGB, xyz)
}

// ---------------------------------------------------------------------------
// Rec.2020
// ---------------------------------------------------------------------------

/// Linear Rec.2020 to CIE XYZ (D65).
pub const REC2020_TO_XYZ: [[f32; 3]; 3] = [
    [0.6369580, 0.1446169, 0.1688810],
    [0.2627002, 0.6779981, 0.0593017],
    [0.0000000, 0.0280727, 1.0609851],
];

/// CIE XYZ (D65) to linear Rec.2020.
pub const XYZ_TO_REC2020: [[f32; 3]; 3] = [
    [ 1.7166512, -0.3556708, -0.2533663],
    [-0.6666844,  1.6164812,  0.0157685],
    [ 0.0176399, -0.0427706,  0.9421031],
];

/// Convert linear sRGB to linear Rec.2020.
pub fn srgb_to_rec2020(rgb: [f32; 3]) -> [f32; 3] {
    let xyz = linear_srgb_to_xyz(rgb);
    mat3_mul(&XYZ_TO_REC2020, xyz)
}

/// Convert linear Rec.2020 to linear sRGB.
pub fn rec2020_to_srgb(rec2020: [f32; 3]) -> [f32; 3] {
    let xyz = mat3_mul(&REC2020_TO_XYZ, rec2020);
    xyz_to_linear_srgb(xyz)
}

// ---------------------------------------------------------------------------
// DCI-P3
// ---------------------------------------------------------------------------

/// Linear DCI-P3 (D65) to CIE XYZ.
pub const P3_TO_XYZ: [[f32; 3]; 3] = [
    [0.4865709, 0.2656677, 0.1982173],
    [0.2289746, 0.6917385, 0.0792869],
    [0.0000000, 0.0451134, 1.0439444],
];

/// CIE XYZ to linear DCI-P3 (D65).
pub const XYZ_TO_P3: [[f32; 3]; 3] = [
    [ 2.4934969, -0.9313836, -0.4027108],
    [-0.8294890,  1.7626641,  0.0236247],
    [ 0.0358458, -0.0761724,  0.9568845],
];

/// Convert linear sRGB to linear DCI-P3.
pub fn srgb_to_p3(rgb: [f32; 3]) -> [f32; 3] {
    let xyz = linear_srgb_to_xyz(rgb);
    mat3_mul(&XYZ_TO_P3, xyz)
}

/// Convert linear DCI-P3 to linear sRGB.
pub fn p3_to_srgb(p3: [f32; 3]) -> [f32; 3] {
    let xyz = mat3_mul(&P3_TO_XYZ, p3);
    xyz_to_linear_srgb(xyz)
}

// ---------------------------------------------------------------------------
// ACES AP0 / AP1
// ---------------------------------------------------------------------------

/// sRGB to ACES AP0 (ACEScg uses AP1; this is the full ACES interchange).
pub const SRGB_TO_AP0: [[f32; 3]; 3] = [
    [0.4397010, 0.3829780, 0.1773350],
    [0.0897923, 0.8134230, 0.0967616],
    [0.0175440, 0.1115440, 0.8707040],
];

/// ACES AP0 to sRGB.
pub const AP0_TO_SRGB: [[f32; 3]; 3] = [
    [ 2.5216830, -1.1340060, -0.3876770],
    [-0.2764590,  1.3727280, -0.0962693],
    [-0.0153780, -0.1529960,  1.1683740],
];

/// sRGB to ACES AP1 (ACEScg working space).
pub const SRGB_TO_AP1: [[f32; 3]; 3] = [
    [0.6131324, 0.3395380, 0.0474166],
    [0.0701934, 0.9163540, 0.0134526],
    [0.0206155, 0.1095698, 0.8697794],
];

/// ACES AP1 to sRGB.
pub const AP1_TO_SRGB: [[f32; 3]; 3] = [
    [ 1.7050510, -0.6217921, -0.0832589],
    [-0.1302564,  1.1408052, -0.0105488],
    [-0.0240034, -0.1289690,  1.1529720],
];

/// Convert linear sRGB to ACES AP0.
pub fn srgb_to_ap0(rgb: [f32; 3]) -> [f32; 3] {
    mat3_mul(&SRGB_TO_AP0, rgb)
}

/// Convert ACES AP0 to linear sRGB.
pub fn ap0_to_srgb(ap0: [f32; 3]) -> [f32; 3] {
    mat3_mul(&AP0_TO_SRGB, ap0)
}

/// Convert linear sRGB to ACES AP1 (ACEScg).
pub fn srgb_to_ap1(rgb: [f32; 3]) -> [f32; 3] {
    mat3_mul(&SRGB_TO_AP1, rgb)
}

/// Convert ACES AP1 (ACEScg) to linear sRGB.
pub fn ap1_to_srgb(ap1: [f32; 3]) -> [f32; 3] {
    mat3_mul(&AP1_TO_SRGB, ap1)
}

// ---------------------------------------------------------------------------
// Oklab
// ---------------------------------------------------------------------------

/// Convert linear sRGB to Oklab.
///
/// Oklab is a perceptually uniform colour space designed by Björn Ottosson.
pub fn linear_srgb_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let l = 0.4122214708 * rgb[0] + 0.5363325363 * rgb[1] + 0.0514459929 * rgb[2];
    let m = 0.2119034982 * rgb[0] + 0.6806995451 * rgb[1] + 0.1073969566 * rgb[2];
    let s = 0.0883024619 * rgb[0] + 0.2817188376 * rgb[1] + 0.6299787005 * rgb[2];

    let l_ = l.max(0.0).cbrt();
    let m_ = m.max(0.0).cbrt();
    let s_ = s.max(0.0).cbrt();

    [
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ]
}

/// Convert Oklab to linear sRGB.
pub fn oklab_to_linear_srgb(lab: [f32; 3]) -> [f32; 3] {
    let l_ = lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2];
    let m_ = lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2];
    let s_ = lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2];

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    [
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ]
}

// ---------------------------------------------------------------------------
// CIE LAB
// ---------------------------------------------------------------------------

/// D65 white point in XYZ.
pub const D65_WHITE: [f32; 3] = [0.95047, 1.00000, 1.08883];
/// D50 white point in XYZ.
pub const D50_WHITE: [f32; 3] = [0.96422, 1.00000, 0.82521];

/// CIE LAB forward transform helper.
#[inline]
fn lab_f(t: f32) -> f32 {
    let delta = 6.0 / 29.0;
    let delta_sq = delta * delta;
    let delta_cb = delta_sq * delta;

    if t > delta_cb {
        t.cbrt()
    } else {
        t / (3.0 * delta_sq) + 4.0 / 29.0
    }
}

/// CIE LAB inverse transform helper.
#[inline]
fn lab_f_inv(t: f32) -> f32 {
    let delta = 6.0 / 29.0;
    let delta_sq = delta * delta;

    if t > delta {
        t * t * t
    } else {
        3.0 * delta_sq * (t - 4.0 / 29.0)
    }
}

/// Convert CIE XYZ to CIE LAB (using a given white point).
pub fn xyz_to_lab(xyz: [f32; 3], white: [f32; 3]) -> [f32; 3] {
    let fx = lab_f(xyz[0] / white[0]);
    let fy = lab_f(xyz[1] / white[1]);
    let fz = lab_f(xyz[2] / white[2]);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);

    [l, a, b]
}

/// Convert CIE LAB to CIE XYZ (using a given white point).
pub fn lab_to_xyz(lab: [f32; 3], white: [f32; 3]) -> [f32; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = lab[1] / 500.0 + fy;
    let fz = fy - lab[2] / 200.0;

    [
        white[0] * lab_f_inv(fx),
        white[1] * lab_f_inv(fy),
        white[2] * lab_f_inv(fz),
    ]
}

/// Convert linear sRGB to CIE LAB (D65).
pub fn srgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let xyz = linear_srgb_to_xyz(rgb);
    xyz_to_lab(xyz, D65_WHITE)
}

/// Convert CIE LAB (D65) to linear sRGB.
pub fn lab_to_srgb(lab: [f32; 3]) -> [f32; 3] {
    let xyz = lab_to_xyz(lab, D65_WHITE);
    xyz_to_linear_srgb(xyz)
}

// ---------------------------------------------------------------------------
// LCH (cylindrical LAB)
// ---------------------------------------------------------------------------

/// Convert CIE LAB to LCH.
pub fn lab_to_lch(lab: [f32; 3]) -> [f32; 3] {
    let l = lab[0];
    let c = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
    let h = lab[2].atan2(lab[1]).to_degrees();
    let h = if h < 0.0 { h + 360.0 } else { h };
    [l, c, h]
}

/// Convert LCH to CIE LAB.
pub fn lch_to_lab(lch: [f32; 3]) -> [f32; 3] {
    let h_rad = lch[2].to_radians();
    [
        lch[0],
        lch[1] * h_rad.cos(),
        lch[1] * h_rad.sin(),
    ]
}

/// Convert linear sRGB to LCH.
pub fn srgb_to_lch(rgb: [f32; 3]) -> [f32; 3] {
    lab_to_lch(srgb_to_lab(rgb))
}

/// Convert LCH to linear sRGB.
pub fn lch_to_srgb(lch: [f32; 3]) -> [f32; 3] {
    lab_to_srgb(lch_to_lab(lch))
}

// ---------------------------------------------------------------------------
// Colour temperature (Planckian locus)
// ---------------------------------------------------------------------------

/// Convert correlated colour temperature (CCT) to CIE xy chromaticity.
///
/// Uses Kang et al. (2002) approximation for the Planckian locus.
///
/// # Arguments
/// * `kelvin` — Colour temperature in Kelvin (1667 - 25000).
pub fn cct_to_xy(kelvin: f32) -> [f32; 2] {
    let t = kelvin.clamp(1667.0, 25000.0);
    let t2 = t * t;
    let t3 = t2 * t;

    let x = if t <= 4000.0 {
        -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    };

    let x2 = x * x;
    let x3 = x2 * x;

    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };

    [x, y]
}

/// Convert CIE xy chromaticity to CIE XYZ (Y=1).
pub fn xy_to_xyz(xy: [f32; 2]) -> [f32; 3] {
    if xy[1] <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]]
}

/// Convert colour temperature to linear sRGB.
pub fn cct_to_srgb(kelvin: f32) -> [f32; 3] {
    let xy = cct_to_xy(kelvin);
    let xyz = xy_to_xyz(xy);
    xyz_to_linear_srgb(xyz)
}

/// Tint a colour towards a target colour temperature.
///
/// # Arguments
/// * `color` — Source linear RGB.
/// * `kelvin` — Target colour temperature.
/// * `strength` — Tint strength [0, 1].
pub fn apply_color_temperature(color: [f32; 3], kelvin: f32, strength: f32) -> [f32; 3] {
    let tint = cct_to_srgb(kelvin);
    let tint_lum = 0.2126 * tint[0] + 0.7152 * tint[1] + 0.0722 * tint[2];
    if tint_lum <= 0.0 {
        return color;
    }
    let inv_lum = 1.0 / tint_lum;
    let normalised = [tint[0] * inv_lum, tint[1] * inv_lum, tint[2] * inv_lum];

    [
        color[0] * lerp(1.0, normalised[0], strength),
        color[1] * lerp(1.0, normalised[1], strength),
        color[2] * lerp(1.0, normalised[2], strength),
    ]
}

// ---------------------------------------------------------------------------
// Chromatic adaptation
// ---------------------------------------------------------------------------

/// Bradford chromatic adaptation matrix (D65 → D50).
pub const BRADFORD_D65_TO_D50: [[f32; 3]; 3] = [
    [ 1.0478112, 0.0228866, -0.0501270],
    [ 0.0295424, 0.9904844, -0.0170491],
    [-0.0092345, 0.0150436,  0.7521316],
];

/// Bradford chromatic adaptation matrix (D50 → D65).
pub const BRADFORD_D50_TO_D65: [[f32; 3]; 3] = [
    [ 0.9555766, -0.0230393,  0.0631636],
    [-0.0282895,  1.0099416,  0.0210077],
    [ 0.0122982, -0.0204830,  1.3299098],
];

/// Bradford cone response matrix.
pub const BRADFORD_CONE: [[f32; 3]; 3] = [
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296],
];

/// Bradford inverse cone response matrix.
pub const BRADFORD_CONE_INV: [[f32; 3]; 3] = [
    [ 0.9869929, -0.1470543,  0.1599627],
    [ 0.4323053,  0.5183603,  0.0492912],
    [-0.0085287,  0.0400428,  0.9684867],
];

/// Von Kries cone response matrix (Hunt-Pointer-Estevez).
pub const VON_KRIES_CONE: [[f32; 3]; 3] = [
    [ 0.4002400,  0.7076000, -0.0808100],
    [-0.2263000,  1.1653200,  0.0457000],
    [ 0.0000000,  0.0000000,  0.9182200],
];

/// Von Kries inverse cone response matrix.
pub const VON_KRIES_CONE_INV: [[f32; 3]; 3] = [
    [1.8599364, -1.1293816,  0.2198974],
    [0.3611914,  0.6388125, -0.0000064],
    [0.0000000,  0.0000000,  1.0890636],
];

/// Adapt a CIE XYZ colour from one white point to another using Bradford.
pub fn bradford_adapt(xyz: [f32; 3], src_white: [f32; 3], dst_white: [f32; 3]) -> [f32; 3] {
    let src_cone = mat3_mul(&BRADFORD_CONE, src_white);
    let dst_cone = mat3_mul(&BRADFORD_CONE, dst_white);

    // Diagonal adaptation matrix.
    let scale = [
        if src_cone[0].abs() > 1e-10 { dst_cone[0] / src_cone[0] } else { 1.0 },
        if src_cone[1].abs() > 1e-10 { dst_cone[1] / src_cone[1] } else { 1.0 },
        if src_cone[2].abs() > 1e-10 { dst_cone[2] / src_cone[2] } else { 1.0 },
    ];

    let cone_response = mat3_mul(&BRADFORD_CONE, xyz);
    let adapted = [
        cone_response[0] * scale[0],
        cone_response[1] * scale[1],
        cone_response[2] * scale[2],
    ];
    mat3_mul(&BRADFORD_CONE_INV, adapted)
}

/// Adapt a CIE XYZ colour from one white point to another using Von Kries.
pub fn von_kries_adapt(xyz: [f32; 3], src_white: [f32; 3], dst_white: [f32; 3]) -> [f32; 3] {
    let src_cone = mat3_mul(&VON_KRIES_CONE, src_white);
    let dst_cone = mat3_mul(&VON_KRIES_CONE, dst_white);

    let scale = [
        if src_cone[0].abs() > 1e-10 { dst_cone[0] / src_cone[0] } else { 1.0 },
        if src_cone[1].abs() > 1e-10 { dst_cone[1] / src_cone[1] } else { 1.0 },
        if src_cone[2].abs() > 1e-10 { dst_cone[2] / src_cone[2] } else { 1.0 },
    ];

    let cone_response = mat3_mul(&VON_KRIES_CONE, xyz);
    let adapted = [
        cone_response[0] * scale[0],
        cone_response[1] * scale[1],
        cone_response[2] * scale[2],
    ];
    mat3_mul(&VON_KRIES_CONE_INV, adapted)
}

// ---------------------------------------------------------------------------
// Gamut mapping
// ---------------------------------------------------------------------------

/// Clamp an sRGB colour to the valid [0, 1] range per channel.
pub fn clamp_srgb(rgb: [f32; 3]) -> [f32; 3] {
    [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ]
}

/// Desaturate out-of-gamut colours towards the neutral axis.
///
/// Preserves luminance while bringing the colour into gamut.
pub fn gamut_map_desaturate(rgb: [f32; 3]) -> [f32; 3] {
    let lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    let min_c = rgb[0].min(rgb[1]).min(rgb[2]);
    let max_c = rgb[0].max(rgb[1]).max(rgb[2]);

    if min_c >= 0.0 && max_c <= 1.0 {
        return rgb; // Already in gamut.
    }

    // Binary search for the maximum saturation that keeps the colour in gamut.
    let mut lo = 0.0_f32;
    let mut hi = 1.0_f32;

    for _ in 0..16 {
        let mid = (lo + hi) * 0.5;
        let test = [
            lum + (rgb[0] - lum) * mid,
            lum + (rgb[1] - lum) * mid,
            lum + (rgb[2] - lum) * mid,
        ];
        let test_min = test[0].min(test[1]).min(test[2]);
        let test_max = test[0].max(test[1]).max(test[2]);

        if test_min >= 0.0 && test_max <= 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    [
        lum + (rgb[0] - lum) * lo,
        lum + (rgb[1] - lum) * lo,
        lum + (rgb[2] - lum) * lo,
    ]
}

/// Check if a colour is within the sRGB gamut.
pub fn is_in_srgb_gamut(rgb: [f32; 3]) -> bool {
    rgb[0] >= -1e-6 && rgb[0] <= 1.0 + 1e-6
        && rgb[1] >= -1e-6 && rgb[1] <= 1.0 + 1e-6
        && rgb[2] >= -1e-6 && rgb[2] <= 1.0 + 1e-6
}

// ---------------------------------------------------------------------------
// Colour difference (ΔE*2000)
// ---------------------------------------------------------------------------

/// Compute the CIEDE2000 colour difference between two CIE LAB colours.
///
/// Returns a perceptual difference value. Values below 1.0 are generally
/// imperceptible; values below 2.0 are nearly indistinguishable.
pub fn delta_e_2000(lab1: [f32; 3], lab2: [f32; 3]) -> f32 {
    let l1 = lab1[0];
    let a1 = lab1[1];
    let b1 = lab1[2];
    let l2 = lab2[0];
    let a2 = lab2[1];
    let b2 = lab2[2];

    // Step 1: Calculate Cab, hab.
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_avg = (c1 + c2) * 0.5;

    let c7 = c_avg.powi(7);
    let g = 0.5 * (1.0 - (c7 / (c7 + 25.0_f32.powi(7))).sqrt());

    let a1p = a1 * (1.0 + g);
    let a2p = a2 * (1.0 + g);

    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();

    let h1p = if a1p.abs() < 1e-10 && b1.abs() < 1e-10 {
        0.0
    } else {
        let h = b1.atan2(a1p).to_degrees();
        if h < 0.0 { h + 360.0 } else { h }
    };

    let h2p = if a2p.abs() < 1e-10 && b2.abs() < 1e-10 {
        0.0
    } else {
        let h = b2.atan2(a2p).to_degrees();
        if h < 0.0 { h + 360.0 } else { h }
    };

    // Step 2: Calculate dL', dC', dH'.
    let dl = l2 - l1;
    let dc = c2p - c1p;

    let dh_deg = if c1p * c2p < 1e-10 {
        0.0
    } else {
        let diff = h2p - h1p;
        if diff.abs() <= 180.0 {
            diff
        } else if diff > 180.0 {
            diff - 360.0
        } else {
            diff + 360.0
        }
    };

    let dh = 2.0 * (c1p * c2p).sqrt() * (dh_deg * 0.5 * PI / 180.0).sin();

    // Step 3: Calculate CIEDE2000.
    let l_avg = (l1 + l2) * 0.5;
    let c_avg_p = (c1p + c2p) * 0.5;

    let h_avg = if c1p * c2p < 1e-10 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) * 0.5
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) * 0.5
    } else {
        (h1p + h2p - 360.0) * 0.5
    };

    let t = 1.0
        - 0.17 * ((h_avg - 30.0).to_radians()).cos()
        + 0.24 * ((2.0 * h_avg).to_radians()).cos()
        + 0.32 * ((3.0 * h_avg + 6.0).to_radians()).cos()
        - 0.20 * ((4.0 * h_avg - 63.0).to_radians()).cos();

    let sl = 1.0 + 0.015 * (l_avg - 50.0).powi(2) / (20.0 + (l_avg - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * c_avg_p;
    let sh = 1.0 + 0.015 * c_avg_p * t;

    let c_avg_p7 = c_avg_p.powi(7);
    let rt = -2.0 * (c_avg_p7 / (c_avg_p7 + 25.0_f32.powi(7))).sqrt()
        * (60.0 * (-((h_avg - 275.0) / 25.0).powi(2)).exp()).to_radians().sin();

    let kl = 1.0_f32;
    let kc = 1.0_f32;
    let kh = 1.0_f32;

    let term_l = dl / (kl * sl);
    let term_c = dc / (kc * sc);
    let term_h = dh / (kh * sh);

    (term_l * term_l + term_c * term_c + term_h * term_h + rt * term_c * term_h).sqrt()
}

// ---------------------------------------------------------------------------
// HSV / HSL conversions
// ---------------------------------------------------------------------------

/// Convert linear RGB to HSV.
pub fn rgb_to_hsv(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max > 0.0 { delta / max } else { 0.0 };

    let h = if delta < 1e-10 {
        0.0
    } else if (max - r).abs() < 1e-10 {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max - g).abs() < 1e-10 {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };
    [h, s, v]
}

/// Convert HSV to linear RGB.
pub fn hsv_to_rgb(hsv: [f32; 3]) -> [f32; 3] {
    let h = hsv[0];
    let s = hsv[1];
    let v = hsv[2];

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

    [r + m, g + m, b + m]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_roundtrip() {
        let original = [0.5, 0.3, 0.7];
        let linear = srgb_to_linear_rgb(original);
        let back = linear_to_srgb_rgb(linear);
        for i in 0..3 {
            assert!((original[i] - back[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_xyz_roundtrip() {
        let rgb = [0.5, 0.3, 0.7];
        let xyz = linear_srgb_to_xyz(rgb);
        let back = xyz_to_linear_srgb(xyz);
        for i in 0..3 {
            assert!((rgb[i] - back[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_oklab_roundtrip() {
        let rgb = [0.5, 0.3, 0.7];
        let lab = linear_srgb_to_oklab(rgb);
        let back = oklab_to_linear_srgb(lab);
        for i in 0..3 {
            assert!((rgb[i] - back[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_lab_roundtrip() {
        let rgb = [0.5, 0.3, 0.7];
        let lab = srgb_to_lab(rgb);
        let back = lab_to_srgb(lab);
        for i in 0..3 {
            assert!((rgb[i] - back[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_lch_roundtrip() {
        let lab = [50.0, 20.0, -30.0];
        let lch = lab_to_lch(lab);
        let back = lch_to_lab(lch);
        for i in 0..3 {
            assert!((lab[i] - back[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_color_temperature() {
        // 6500K should be roughly white.
        let rgb = cct_to_srgb(6500.0);
        assert!(rgb[0] > 0.5 && rgb[1] > 0.5 && rgb[2] > 0.5);

        // 2700K should be warm (reddish).
        let rgb_warm = cct_to_srgb(2700.0);
        assert!(rgb_warm[0] > rgb_warm[2]); // Red > Blue.
    }

    #[test]
    fn test_delta_e_same_color() {
        let lab = [50.0, 20.0, -10.0];
        let de = delta_e_2000(lab, lab);
        assert!(de < 0.001);
    }

    #[test]
    fn test_delta_e_different_colors() {
        let lab1 = [50.0, 0.0, 0.0];
        let lab2 = [80.0, 0.0, 0.0];
        let de = delta_e_2000(lab1, lab2);
        assert!(de > 10.0);
    }

    #[test]
    fn test_gamut_map() {
        let out_of_gamut = [1.5, -0.2, 0.8];
        let mapped = gamut_map_desaturate(out_of_gamut);
        assert!(mapped[0] >= 0.0 && mapped[0] <= 1.0);
        assert!(mapped[1] >= 0.0 && mapped[1] <= 1.0);
        assert!(mapped[2] >= 0.0 && mapped[2] <= 1.0);
    }

    #[test]
    fn test_hsv_roundtrip() {
        let rgb = [0.8, 0.3, 0.5];
        let hsv = rgb_to_hsv(rgb);
        let back = hsv_to_rgb(hsv);
        for i in 0..3 {
            assert!((rgb[i] - back[i]).abs() < 1e-4);
        }
    }
}
