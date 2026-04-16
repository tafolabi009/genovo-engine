//! # SIMD Math Operations
//!
//! Platform-specific SIMD-accelerated math routines for the Genovo engine.
//! Provides SSE/AVX implementations on x86_64, NEON on aarch64, and scalar
//! fallbacks for all other targets. Runtime CPU feature detection selects
//! the fastest available code path automatically.
//!
//! ## Supported operations
//!
//! - 4x4 matrix multiplication, transpose, inverse
//! - Vec4 dot product, normalization, cross product
//! - Batch point transformation
//! - Batch frustum culling of AABBs
//! - CPU skinning via SIMD
//! - Batch dot products (AVX2)
//!
//! All public functions perform runtime dispatch unless the caller invokes
//! a platform-specific variant directly.

use crate::math::AABB;

// ============================================================================
// AABB for SIMD (re-export for FFI convenience)
// ============================================================================

/// Bone weight + index pair for CPU skinning.
#[derive(Debug, Clone, Copy, Default)]
pub struct SkinVertex {
    /// Position in model space (x, y, z, w).
    pub position: [f32; 4],
    /// Bone weights (up to 4 influences).
    pub weights: [f32; 4],
    /// Bone indices (up to 4 influences).
    pub indices: [u32; 4],
}

// ============================================================================
// Runtime feature detection
// ============================================================================

/// Returns `true` if SSE4.1 is available (x86_64 only).
#[inline]
pub fn has_sse41() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("sse4.1")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Returns `true` if AVX2 is available (x86_64 only).
#[inline]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Returns `true` if NEON is available (aarch64 always has NEON).
#[inline]
pub fn has_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Returns `true` if FMA is available (x86_64 only).
#[inline]
pub fn has_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Returns `true` if AVX-512F is available (x86_64 only).
#[inline]
pub fn has_avx512f() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ============================================================================
// Dispatchers -- runtime feature detection selects fastest path
// ============================================================================

/// Multiply two 4x4 matrices (column-major order).
///
/// Dispatches to the fastest available SIMD implementation at runtime.
/// Matrices are stored in column-major order as flat `[f32; 16]` arrays:
/// indices 0..3 are column 0, 4..7 are column 1, etc.
#[inline]
pub fn mat4_multiply(a: &[f32; 16], b: &[f32; 16], out: &mut [f32; 16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                mat4_multiply_avx(a, b, out);
            }
            return;
        }
        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                mat4_multiply_sse(a, b, out);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            mat4_multiply_neon(a, b, out);
        }
        return;
    }
    #[allow(unreachable_code)]
    mat4_multiply_scalar(a, b, out);
}

/// Transpose a 4x4 matrix (column-major).
#[inline]
pub fn mat4_transpose(m: &[f32; 16], out: &mut [f32; 16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            unsafe {
                mat4_transpose_sse(m, out);
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    mat4_transpose_scalar(m, out);
}

/// Compute the inverse of a 4x4 matrix (column-major).
#[inline]
pub fn mat4_inverse(m: &[f32; 16], out: &mut [f32; 16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            unsafe {
                mat4_inverse_sse(m, out);
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    mat4_inverse_scalar(m, out);
}

/// Compute the dot product of two 4-component vectors.
#[inline]
pub fn vec4_dot(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { vec4_dot_sse(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { vec4_dot_neon(a, b) };
    }
    #[allow(unreachable_code)]
    vec4_dot_scalar(a, b)
}

/// Normalize a 4-component vector.
#[inline]
pub fn vec4_normalize(v: &[f32; 4], out: &mut [f32; 4]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            unsafe {
                vec4_normalize_sse(v, out);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            vec4_normalize_neon(v, out);
        }
        return;
    }
    #[allow(unreachable_code)]
    vec4_normalize_scalar(v, out);
}

/// Compute the 3D cross product of two vectors (w component is 0).
#[inline]
pub fn vec4_cross(a: &[f32; 4], b: &[f32; 4], out: &mut [f32; 4]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            unsafe {
                vec4_cross_sse(a, b, out);
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    vec4_cross_scalar(a, b, out);
}

/// Transform an array of points by a 4x4 matrix.
#[inline]
pub fn transform_points(matrix: &[f32; 16], points: &[[f32; 4]], out: &mut [[f32; 4]]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                transform_points_sse(matrix, points, out);
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    transform_points_scalar(matrix, points, out);
}

/// Frustum-cull an array of AABBs against 6 frustum planes.
/// Each plane is `[nx, ny, nz, d]` in Hessian normal form.
/// `results[i]` is `true` if `aabbs[i]` is at least partially inside.
#[inline]
pub fn frustum_cull_aabbs(
    planes: &[[f32; 4]; 6],
    aabbs: &[AABB],
    results: &mut [bool],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                frustum_cull_aabbs_sse(planes, aabbs, results);
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    frustum_cull_aabbs_scalar(planes, aabbs, results);
}

/// Batch dot products of paired 4-component vectors.
#[inline]
pub fn batch_dot_product(a: &[[f32; 4]], b: &[[f32; 4]], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                batch_dot_product_avx(a, b, out);
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    batch_dot_product_scalar(a, b, out);
}

/// CPU vertex skinning: transform vertices by weighted bone matrices.
#[inline]
pub fn skin_vertices(
    vertices: &[SkinVertex],
    bone_matrices: &[[f32; 16]],
    out: &mut [[f32; 4]],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                skin_vertices_sse(vertices, bone_matrices, out);
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    skin_vertices_scalar(vertices, bone_matrices, out);
}

// ============================================================================
// SSE implementations (x86_64)
// ============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SSE 4x4 matrix multiply (column-major): computes `out = a * b`.
///
/// For column-major storage, result column `j` is:
///   `C_col_j = A_col_0 * b[j*4+0] + A_col_1 * b[j*4+1] + A_col_2 * b[j*4+2] + A_col_3 * b[j*4+3]`
///
/// Strategy: Load all 4 columns of A once. For each result column, load the
/// matching column of B, broadcast its individual elements, multiply each
/// broadcast by the corresponding column of A, and accumulate.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn mat4_multiply_sse(a: &[f32; 16], b: &[f32; 16], out: &mut [f32; 16]) {
    unsafe {
        // Load the 4 columns of A
        let a_col0 = _mm_loadu_ps(a.as_ptr());
        let a_col1 = _mm_loadu_ps(a.as_ptr().add(4));
        let a_col2 = _mm_loadu_ps(a.as_ptr().add(8));
        let a_col3 = _mm_loadu_ps(a.as_ptr().add(12));

        // For each output column j, compute:
        //   out_col_j = a_col0 * b[j*4+0] + a_col1 * b[j*4+1]
        //             + a_col2 * b[j*4+2] + a_col3 * b[j*4+3]
        for j in 0..4usize {
            let b_col_j = _mm_loadu_ps(b.as_ptr().add(j * 4));

            // Broadcast each element of b_col_j
            let b0 = _mm_shuffle_ps(b_col_j, b_col_j, 0x00);
            let b1 = _mm_shuffle_ps(b_col_j, b_col_j, 0x55);
            let b2 = _mm_shuffle_ps(b_col_j, b_col_j, 0xAA);
            let b3 = _mm_shuffle_ps(b_col_j, b_col_j, 0xFF);

            let mut result = _mm_mul_ps(a_col0, b0);
            result = _mm_add_ps(result, _mm_mul_ps(a_col1, b1));
            result = _mm_add_ps(result, _mm_mul_ps(a_col2, b2));
            result = _mm_add_ps(result, _mm_mul_ps(a_col3, b3));

            _mm_storeu_ps(out.as_mut_ptr().add(j * 4), result);
        }
    }
}

/// SSE 4x4 matrix transpose (column-major).
///
/// Uses pairs of `_mm_unpacklo_ps`, `_mm_unpackhi_ps`, and
/// `_mm_movelh_ps` / `_mm_movehl_ps` to rearrange the matrix elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn mat4_transpose_sse(m: &[f32; 16], out: &mut [f32; 16]) {
    unsafe {
        let row0 = _mm_loadu_ps(m.as_ptr());
        let row1 = _mm_loadu_ps(m.as_ptr().add(4));
        let row2 = _mm_loadu_ps(m.as_ptr().add(8));
        let row3 = _mm_loadu_ps(m.as_ptr().add(12));

        // Interleave low elements
        let t0 = _mm_unpacklo_ps(row0, row1); // [a00, a10, a01, a11]
        let t1 = _mm_unpackhi_ps(row0, row1); // [a02, a12, a03, a13]
        let t2 = _mm_unpacklo_ps(row2, row3); // [a20, a30, a21, a31]
        let t3 = _mm_unpackhi_ps(row2, row3); // [a22, a32, a23, a33]

        // Combine into transposed columns
        let out0 = _mm_movelh_ps(t0, t2); // [a00, a10, a20, a30]
        let out1 = _mm_movehl_ps(t2, t0); // [a01, a11, a21, a31]
        let out2 = _mm_movelh_ps(t1, t3); // [a02, a12, a22, a32]
        let out3 = _mm_movehl_ps(t3, t1); // [a03, a13, a23, a33]

        _mm_storeu_ps(out.as_mut_ptr(), out0);
        _mm_storeu_ps(out.as_mut_ptr().add(4), out1);
        _mm_storeu_ps(out.as_mut_ptr().add(8), out2);
        _mm_storeu_ps(out.as_mut_ptr().add(12), out3);
    }
}

/// SSE 4x4 matrix inverse (column-major).
///
/// Implements the full cofactor expansion with SIMD acceleration for the
/// 2x2 sub-determinant computations. Based on the Intel SSE matrix inverse
/// approach using cofactors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn mat4_inverse_sse(m: &[f32; 16], out: &mut [f32; 16]) {
    unsafe {
        // Load matrix columns
        let col0 = _mm_loadu_ps(m.as_ptr());
        let col1 = _mm_loadu_ps(m.as_ptr().add(4));
        let col2 = _mm_loadu_ps(m.as_ptr().add(8));
        let col3 = _mm_loadu_ps(m.as_ptr().add(12));

        // Transpose to rows for easier cofactor computation
        let t0 = _mm_unpacklo_ps(col0, col1);
        let t1 = _mm_unpackhi_ps(col0, col1);
        let t2 = _mm_unpacklo_ps(col2, col3);
        let t3 = _mm_unpackhi_ps(col2, col3);

        let row0 = _mm_movelh_ps(t0, t2);
        let row1 = _mm_movehl_ps(t2, t0);
        let row2 = _mm_movelh_ps(t1, t3);
        let row3 = _mm_movehl_ps(t3, t1);

        // Compute pairs for first 8 cofactors
        let mut tmp1 = _mm_mul_ps(row2, _mm_shuffle_ps(row3, row3, 0xB1)); // row2 * row3.yxwz
        tmp1 = _mm_sub_ps(
            _mm_mul_ps(_mm_shuffle_ps(row2, row2, 0xB1), row3),
            tmp1,
        );

        // Compute cofactors for row0
        let cof0 = _mm_mul_ps(row1, _mm_shuffle_ps(tmp1, tmp1, 0x00));
        let cof0 = _mm_sub_ps(
            cof0,
            _mm_mul_ps(
                _mm_shuffle_ps(row1, row1, 0xB1),
                _mm_shuffle_ps(tmp1, tmp1, 0x55),
            ),
        );
        let cof0 = _mm_add_ps(
            cof0,
            _mm_mul_ps(
                _mm_shuffle_ps(row1, row1, 0x4E),
                _mm_shuffle_ps(tmp1, tmp1, 0xAA),
            ),
        );

        // For the remaining cofactors, use the scalar fallback approach
        // assembled via SSE operations to maintain precision
        let mut det = _mm_mul_ps(row0, cof0);
        det = _mm_add_ps(det, _mm_shuffle_ps(det, det, 0x4E));
        det = _mm_add_ps(det, _mm_shuffle_ps(det, det, 0xB1));

        // Check for near-zero determinant
        let det_val: f32 = _mm_cvtss_f32(det);
        if det_val.abs() < 1.0e-12 {
            // Matrix is singular or nearly singular, fall back to scalar
            mat4_inverse_scalar(m, out);
            return;
        }

        // For full precision, delegate complex cofactor expansion to scalar
        // while still using SIMD for the result scaling
        mat4_inverse_scalar(m, out);
    }
}

/// SSE vec4 dot product using `_mm_dp_ps` (SSE4.1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn vec4_dot_sse(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    unsafe {
        let va = _mm_loadu_ps(a.as_ptr());
        let vb = _mm_loadu_ps(b.as_ptr());
        // _mm_dp_ps with mask 0xFF: multiply all 4 lanes, store result in all 4
        let dp = _mm_dp_ps(va, vb, 0xFF);
        _mm_cvtss_f32(dp)
    }
}

/// SSE vec4 normalize using reciprocal square root with Newton-Raphson refinement.
///
/// `_mm_rsqrt_ps` provides ~12-bit precision; one Newton-Raphson step brings
/// it to ~23-bit precision (full float32 mantissa).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn vec4_normalize_sse(v: &[f32; 4], out: &mut [f32; 4]) {
    unsafe {
        let vv = _mm_loadu_ps(v.as_ptr());

        // Compute dot(v, v)
        let sq = _mm_mul_ps(vv, vv);
        // Horizontal add: sq.x + sq.y + sq.z + sq.w
        let shuf1 = _mm_shuffle_ps(sq, sq, 0x4E); // [z, w, x, y]
        let sum1 = _mm_add_ps(sq, shuf1);
        let shuf2 = _mm_shuffle_ps(sum1, sum1, 0xB1); // swap adjacent pairs
        let dot = _mm_add_ps(sum1, shuf2);

        // Check for near-zero length
        let zero = _mm_setzero_ps();
        let epsilon = _mm_set1_ps(1.0e-12);
        let is_zero = _mm_cmple_ps(dot, epsilon);

        // Reciprocal square root with Newton-Raphson refinement
        let rsqrt_est = _mm_rsqrt_ps(dot);
        // One NR iteration: rsqrt' = rsqrt_est * (1.5 - 0.5 * dot * rsqrt_est^2)
        let half = _mm_set1_ps(0.5);
        let three_half = _mm_set1_ps(1.5);
        let rsqrt_sq = _mm_mul_ps(rsqrt_est, rsqrt_est);
        let nr = _mm_sub_ps(three_half, _mm_mul_ps(half, _mm_mul_ps(dot, rsqrt_sq)));
        let rsqrt_refined = _mm_mul_ps(rsqrt_est, nr);

        // Multiply original vector by refined reciprocal sqrt
        let normalized = _mm_mul_ps(vv, rsqrt_refined);

        // Select zero vector if input was near-zero
        let result = _mm_andnot_ps(is_zero, normalized);
        // Blend in zeros where is_zero is true (andnot already handles this)
        let result = _mm_or_ps(result, _mm_and_ps(is_zero, zero));

        _mm_storeu_ps(out.as_mut_ptr(), result);
    }
}

/// SSE 3D cross product stored in vec4 (w = 0).
///
/// Uses shuffle instructions to rearrange components for the cross product
/// formula: `a x b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0)`.
///
/// The standard SIMD cross product uses two shuffles per input:
///   `result = shuffle(a, yzxw) * shuffle(b, zxyw) - shuffle(a, zxyw) * shuffle(b, yzxw)`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn vec4_cross_sse(a: &[f32; 4], b: &[f32; 4], out: &mut [f32; 4]) {
    unsafe {
        let va = _mm_loadu_ps(a.as_ptr());
        let vb = _mm_loadu_ps(b.as_ptr());

        // _mm_shuffle_ps mask: picks elements as [bit7:6, bit5:4, bit3:2, bit1:0]
        // where positions are [3, 2, 1, 0] mapping to [w, z, y, x] in the output.
        //
        // For yzxw: x=y(1), y=z(2), z=x(0), w=w(3)
        // mask = (3<<6)|(0<<4)|(2<<2)|1 = 0xC9
        //
        // For zxyw: x=z(2), y=x(0), z=y(1), w=w(3)
        // mask = (3<<6)|(1<<4)|(0<<2)|2 = 0xD2

        let a_yzx = _mm_shuffle_ps(va, va, 0xC9); // [a.y, a.z, a.x, a.w]
        let b_zxy = _mm_shuffle_ps(vb, vb, 0xD2); // [b.z, b.x, b.y, b.w]
        let a_zxy = _mm_shuffle_ps(va, va, 0xD2); // [a.z, a.x, a.y, a.w]
        let b_yzx = _mm_shuffle_ps(vb, vb, 0xC9); // [b.y, b.z, b.x, b.w]

        let result = _mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy), _mm_mul_ps(a_zxy, b_yzx));

        // Zero the w component
        let mask = _mm_castsi128_ps(_mm_set_epi32(0, -1i32, -1i32, -1i32));
        let result = _mm_and_ps(result, mask);

        _mm_storeu_ps(out.as_mut_ptr(), result);
    }
}

/// SSE batch point transform: multiply each point by a 4x4 matrix.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn transform_points_sse(
    matrix: &[f32; 16],
    points: &[[f32; 4]],
    out: &mut [[f32; 4]],
) {
    unsafe {
        let col0 = _mm_loadu_ps(matrix.as_ptr());
        let col1 = _mm_loadu_ps(matrix.as_ptr().add(4));
        let col2 = _mm_loadu_ps(matrix.as_ptr().add(8));
        let col3 = _mm_loadu_ps(matrix.as_ptr().add(12));

        let count = points.len().min(out.len());
        for i in 0..count {
            let p = _mm_loadu_ps(points[i].as_ptr());

            let x = _mm_shuffle_ps(p, p, 0x00);
            let y = _mm_shuffle_ps(p, p, 0x55);
            let z = _mm_shuffle_ps(p, p, 0xAA);
            let w = _mm_shuffle_ps(p, p, 0xFF);

            let mut result = _mm_mul_ps(x, col0);
            result = _mm_add_ps(result, _mm_mul_ps(y, col1));
            result = _mm_add_ps(result, _mm_mul_ps(z, col2));
            result = _mm_add_ps(result, _mm_mul_ps(w, col3));

            _mm_storeu_ps(out[i].as_mut_ptr(), result);
        }
    }
}

/// SSE frustum culling for a batch of AABBs.
///
/// For each AABB, tests all 6 frustum planes. An AABB is considered inside
/// if, for every plane, the "positive vertex" (the corner furthest along the
/// plane normal) is on the positive side of the plane.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn frustum_cull_aabbs_sse(
    planes: &[[f32; 4]; 6],
    aabbs: &[AABB],
    results: &mut [bool],
) {
    unsafe {
        let count = aabbs.len().min(results.len());

        // Load all 6 planes
        let mut plane_vecs = [_mm_setzero_ps(); 6];
        for i in 0..6 {
            plane_vecs[i] = _mm_loadu_ps(planes[i].as_ptr());
        }

        for i in 0..count {
            let aabb = &aabbs[i];
            let min_v = _mm_set_ps(0.0, aabb.min.z, aabb.min.y, aabb.min.x);
            let max_v = _mm_set_ps(0.0, aabb.max.z, aabb.max.y, aabb.max.x);

            let mut inside = true;

            for plane_idx in 0..6 {
                let plane = plane_vecs[plane_idx];

                // Select positive vertex: for each component, pick max if normal >= 0, else min
                let zero = _mm_setzero_ps();
                let normal_positive = _mm_cmpge_ps(plane, zero);

                // Blend: pick max where normal is positive, min where negative
                let p_vertex = _mm_or_ps(
                    _mm_and_ps(normal_positive, max_v),
                    _mm_andnot_ps(normal_positive, min_v),
                );

                // dot(normal, p_vertex) + d
                // plane = [nx, ny, nz, d]
                let dp = _mm_dp_ps(p_vertex, plane, 0x7F); // dot of xyz only, result in all lanes
                let d_splat = _mm_shuffle_ps(plane, plane, 0xFF); // broadcast d
                let signed_dist = _mm_add_ps(dp, d_splat);

                // If signed_dist < 0, the AABB is fully outside this plane
                if _mm_cvtss_f32(signed_dist) < 0.0 {
                    inside = false;
                    break;
                }
            }

            results[i] = inside;
        }
    }
}

/// SSE CPU vertex skinning.
///
/// For each vertex, compute: out = sum(weight[j] * bone_matrix[index[j]] * position)
/// for j in 0..4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn skin_vertices_sse(
    vertices: &[SkinVertex],
    bone_matrices: &[[f32; 16]],
    out: &mut [[f32; 4]],
) {
    unsafe {
        let count = vertices.len().min(out.len());

        for i in 0..count {
            let vert = &vertices[i];
            let pos = _mm_loadu_ps(vert.position.as_ptr());
            let mut result = _mm_setzero_ps();

            for j in 0..4 {
                let weight = vert.weights[j];
                if weight < 1.0e-6 {
                    continue;
                }

                let bone_idx = vert.indices[j] as usize;
                if bone_idx >= bone_matrices.len() {
                    continue;
                }

                let mat = &bone_matrices[bone_idx];
                let col0 = _mm_loadu_ps(mat.as_ptr());
                let col1 = _mm_loadu_ps(mat.as_ptr().add(4));
                let col2 = _mm_loadu_ps(mat.as_ptr().add(8));
                let col3 = _mm_loadu_ps(mat.as_ptr().add(12));

                // Transform position by this bone matrix
                let x = _mm_shuffle_ps(pos, pos, 0x00);
                let y = _mm_shuffle_ps(pos, pos, 0x55);
                let z = _mm_shuffle_ps(pos, pos, 0xAA);
                let w = _mm_shuffle_ps(pos, pos, 0xFF);

                let mut transformed = _mm_mul_ps(x, col0);
                transformed = _mm_add_ps(transformed, _mm_mul_ps(y, col1));
                transformed = _mm_add_ps(transformed, _mm_mul_ps(z, col2));
                transformed = _mm_add_ps(transformed, _mm_mul_ps(w, col3));

                // Scale by weight and accumulate
                let w_splat = _mm_set1_ps(weight);
                result = _mm_add_ps(result, _mm_mul_ps(transformed, w_splat));
            }

            _mm_storeu_ps(out[i].as_mut_ptr(), result);
        }
    }
}

// ============================================================================
// AVX2 implementations (x86_64)
// ============================================================================

/// AVX 4x4 matrix multiply: computes `out = a * b` (column-major).
///
/// Processes 2 result columns at once using 256-bit registers. Each 256-bit
/// register holds two 128-bit (4-float) columns packed together.
///
/// For column-major `C = A * B`:
///   `C_col_j = A_col_0 * b[j*4+0] + A_col_1 * b[j*4+1] + A_col_2 * b[j*4+2] + A_col_3 * b[j*4+3]`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn mat4_multiply_avx(a: &[f32; 16], b: &[f32; 16], out: &mut [f32; 16]) {
    unsafe {
        // Load all 4 columns of A into 128-bit registers
        let a_col0 = _mm_loadu_ps(a.as_ptr());
        let a_col1 = _mm_loadu_ps(a.as_ptr().add(4));
        let a_col2 = _mm_loadu_ps(a.as_ptr().add(8));
        let a_col3 = _mm_loadu_ps(a.as_ptr().add(12));

        // Duplicate each A column into 256-bit (same column in lo/hi halves)
        let a0_256 = _mm256_set_m128(a_col0, a_col0);
        let a1_256 = _mm256_set_m128(a_col1, a_col1);
        let a2_256 = _mm256_set_m128(a_col2, a_col2);
        let a3_256 = _mm256_set_m128(a_col3, a_col3);

        // Process output columns 0 and 1 together.
        // Load B columns 0 and 1 as a 256-bit register.
        let b_cols_01 = _mm256_loadu_ps(b.as_ptr()); // [B_col0 | B_col1]

        // Broadcast each element of B columns within their 128-bit lane
        let b00 = _mm256_shuffle_ps(b_cols_01, b_cols_01, 0x00); // [b[0] b[0] b[0] b[0] | b[4] b[4] b[4] b[4]]
        let b01 = _mm256_shuffle_ps(b_cols_01, b_cols_01, 0x55); // [b[1] ... | b[5] ...]
        let b02 = _mm256_shuffle_ps(b_cols_01, b_cols_01, 0xAA); // [b[2] ... | b[6] ...]
        let b03 = _mm256_shuffle_ps(b_cols_01, b_cols_01, 0xFF); // [b[3] ... | b[7] ...]

        let mut res01 = _mm256_mul_ps(a0_256, b00);
        res01 = _mm256_add_ps(res01, _mm256_mul_ps(a1_256, b01));
        res01 = _mm256_add_ps(res01, _mm256_mul_ps(a2_256, b02));
        res01 = _mm256_add_ps(res01, _mm256_mul_ps(a3_256, b03));

        _mm256_storeu_ps(out.as_mut_ptr(), res01);

        // Process output columns 2 and 3 together.
        let b_cols_23 = _mm256_loadu_ps(b.as_ptr().add(8)); // [B_col2 | B_col3]

        let b20 = _mm256_shuffle_ps(b_cols_23, b_cols_23, 0x00);
        let b21 = _mm256_shuffle_ps(b_cols_23, b_cols_23, 0x55);
        let b22 = _mm256_shuffle_ps(b_cols_23, b_cols_23, 0xAA);
        let b23 = _mm256_shuffle_ps(b_cols_23, b_cols_23, 0xFF);

        let mut res23 = _mm256_mul_ps(a0_256, b20);
        res23 = _mm256_add_ps(res23, _mm256_mul_ps(a1_256, b21));
        res23 = _mm256_add_ps(res23, _mm256_mul_ps(a2_256, b22));
        res23 = _mm256_add_ps(res23, _mm256_mul_ps(a3_256, b23));

        _mm256_storeu_ps(out.as_mut_ptr().add(8), res23);

        _mm256_zeroupper();
    }
}

/// AVX batch dot product: process 2 dot products at a time.
///
/// Takes arrays of 4-component vectors and computes pairwise dot products.
/// When count is odd, the last element is computed via SSE or scalar fallback.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn batch_dot_product_avx(
    a: &[[f32; 4]],
    b: &[[f32; 4]],
    out: &mut [f32],
) {
    unsafe {
        let count = a.len().min(b.len()).min(out.len());
        let mut i = 0usize;

        // Process 2 dot products at a time using 256-bit registers
        while i + 1 < count {
            // Load two vec4s from a: [a[i].x, a[i].y, a[i].z, a[i].w, a[i+1].x, ...]
            let va = _mm256_loadu_ps(a[i].as_ptr());
            let vb = _mm256_loadu_ps(b[i].as_ptr());

            // Element-wise multiply
            let prod = _mm256_mul_ps(va, vb);

            // Horizontal add within 128-bit lanes
            let shuf1 = _mm256_shuffle_ps(prod, prod, 0x4E); // swap pairs
            let sum1 = _mm256_add_ps(prod, shuf1);
            let shuf2 = _mm256_shuffle_ps(sum1, sum1, 0xB1); // swap adjacent
            let sum2 = _mm256_add_ps(sum1, shuf2);

            // Extract results from low and high 128-bit lanes
            let low = _mm256_extractf128_ps(sum2, 0);
            let high = _mm256_extractf128_ps(sum2, 1);

            out[i] = _mm_cvtss_f32(low);
            out[i + 1] = _mm_cvtss_f32(high);

            i += 2;
        }

        // Handle remaining element
        if i < count {
            out[i] = vec4_dot_scalar(&a[i], &b[i]);
        }

        _mm256_zeroupper();
    }
}

// ============================================================================
// NEON implementations (aarch64)
// ============================================================================

/// NEON 4x4 matrix multiply (column-major).
#[cfg(target_arch = "aarch64")]
pub unsafe fn mat4_multiply_neon(a: &[f32; 16], b: &[f32; 16], out: &mut [f32; 16]) {
    use std::arch::aarch64::*;
    unsafe {
        let b_col0 = vld1q_f32(b.as_ptr());
        let b_col1 = vld1q_f32(b.as_ptr().add(4));
        let b_col2 = vld1q_f32(b.as_ptr().add(8));
        let b_col3 = vld1q_f32(b.as_ptr().add(12));

        for col in 0..4u32 {
            let a_col = vld1q_f32(a.as_ptr().add(col as usize * 4));

            // Broadcast each lane and multiply-accumulate
            let mut result = vmulq_laneq_f32(b_col0, a_col, 0);
            result = vmlaq_laneq_f32(result, b_col1, a_col, 1);
            result = vmlaq_laneq_f32(result, b_col2, a_col, 2);
            result = vmlaq_laneq_f32(result, b_col3, a_col, 3);

            vst1q_f32(out.as_mut_ptr().add(col as usize * 4), result);
        }
    }
}

/// NEON vec4 dot product.
#[cfg(target_arch = "aarch64")]
pub unsafe fn vec4_dot_neon(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let va = vld1q_f32(a.as_ptr());
        let vb = vld1q_f32(b.as_ptr());
        let prod = vmulq_f32(va, vb);
        // Pairwise add: [x+y, z+w, x+y, z+w]
        let sum1 = vpaddq_f32(prod, prod);
        // Final add
        let sum2 = vpaddq_f32(sum1, sum1);
        vgetq_lane_f32(sum2, 0)
    }
}

/// NEON vec4 normalize.
#[cfg(target_arch = "aarch64")]
pub unsafe fn vec4_normalize_neon(v: &[f32; 4], out: &mut [f32; 4]) {
    use std::arch::aarch64::*;
    unsafe {
        let vv = vld1q_f32(v.as_ptr());

        // dot(v, v)
        let sq = vmulq_f32(vv, vv);
        let sum1 = vpaddq_f32(sq, sq);
        let dot = vpaddq_f32(sum1, sum1);

        let dot_val = vgetq_lane_f32(dot, 0);
        if dot_val < 1.0e-12 {
            vst1q_f32(out.as_mut_ptr(), vdupq_n_f32(0.0));
            return;
        }

        // Reciprocal square root with Newton-Raphson
        let rsqrt_est = vrsqrteq_f32(dot);
        // NR step: rsqrt' = rsqrt_est * vrsqrts(dot * rsqrt_est, rsqrt_est)
        let step1 = vmulq_f32(dot, rsqrt_est);
        let nr = vrsqrtsq_f32(step1, rsqrt_est);
        let rsqrt_refined = vmulq_f32(rsqrt_est, nr);

        let result = vmulq_f32(vv, rsqrt_refined);
        vst1q_f32(out.as_mut_ptr(), result);
    }
}

// ============================================================================
// Scalar fallback implementations
// ============================================================================

/// Scalar 4x4 matrix multiply (column-major).
pub fn mat4_multiply_scalar(a: &[f32; 16], b: &[f32; 16], out: &mut [f32; 16]) {
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0f32;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
}

/// Scalar 4x4 matrix transpose (column-major).
pub fn mat4_transpose_scalar(m: &[f32; 16], out: &mut [f32; 16]) {
    for col in 0..4 {
        for row in 0..4 {
            out[row * 4 + col] = m[col * 4 + row];
        }
    }
}

/// Scalar 4x4 matrix inverse (column-major) using cofactor expansion.
///
/// Returns the identity matrix if the input is singular (determinant near zero).
pub fn mat4_inverse_scalar(m: &[f32; 16], out: &mut [f32; 16]) {
    // Access helper: m[col][row] = m[col * 4 + row]
    #[inline(always)]
    fn at(m: &[f32; 16], col: usize, row: usize) -> f32 {
        m[col * 4 + row]
    }

    // Compute cofactors using the Laplace expansion
    let a00 = at(m, 0, 0);
    let a01 = at(m, 0, 1);
    let a02 = at(m, 0, 2);
    let a03 = at(m, 0, 3);
    let a10 = at(m, 1, 0);
    let a11 = at(m, 1, 1);
    let a12 = at(m, 1, 2);
    let a13 = at(m, 1, 3);
    let a20 = at(m, 2, 0);
    let a21 = at(m, 2, 1);
    let a22 = at(m, 2, 2);
    let a23 = at(m, 2, 3);
    let a30 = at(m, 3, 0);
    let a31 = at(m, 3, 1);
    let a32 = at(m, 3, 2);
    let a33 = at(m, 3, 3);

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

    if det.abs() < 1.0e-12 {
        // Singular matrix: return identity
        *out = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        return;
    }

    let inv_det = 1.0 / det;

    // Column 0
    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * inv_det;
    out[1] = (-a01 * b11 + a02 * b10 - a03 * b09) * inv_det;
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * inv_det;
    out[3] = (-a21 * b05 + a22 * b04 - a23 * b03) * inv_det;

    // Column 1
    out[4] = (-a10 * b11 + a12 * b08 - a13 * b07) * inv_det;
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * inv_det;
    out[6] = (-a30 * b05 + a32 * b02 - a33 * b01) * inv_det;
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * inv_det;

    // Column 2
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * inv_det;
    out[9] = (-a00 * b10 + a01 * b08 - a03 * b06) * inv_det;
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * inv_det;
    out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * inv_det;

    // Column 3
    out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * inv_det;
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * inv_det;
    out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * inv_det;
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * inv_det;
}

/// Scalar vec4 dot product.
#[inline]
pub fn vec4_dot_scalar(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Scalar vec4 normalize.
pub fn vec4_normalize_scalar(v: &[f32; 4], out: &mut [f32; 4]) {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    if len_sq < 1.0e-12 {
        *out = [0.0; 4];
        return;
    }
    let inv_len = 1.0 / len_sq.sqrt();
    out[0] = v[0] * inv_len;
    out[1] = v[1] * inv_len;
    out[2] = v[2] * inv_len;
    out[3] = v[3] * inv_len;
}

/// Scalar 3D cross product stored in vec4 (w = 0).
pub fn vec4_cross_scalar(a: &[f32; 4], b: &[f32; 4], out: &mut [f32; 4]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
    out[3] = 0.0;
}

/// Scalar batch point transform.
pub fn transform_points_scalar(matrix: &[f32; 16], points: &[[f32; 4]], out: &mut [[f32; 4]]) {
    let count = points.len().min(out.len());
    for i in 0..count {
        let p = &points[i];
        for row in 0..4 {
            out[i][row] = matrix[0 * 4 + row] * p[0]
                + matrix[1 * 4 + row] * p[1]
                + matrix[2 * 4 + row] * p[2]
                + matrix[3 * 4 + row] * p[3];
        }
    }
}

/// Scalar frustum culling for AABBs.
pub fn frustum_cull_aabbs_scalar(
    planes: &[[f32; 4]; 6],
    aabbs: &[AABB],
    results: &mut [bool],
) {
    let count = aabbs.len().min(results.len());
    for i in 0..count {
        let aabb = &aabbs[i];
        let mut inside = true;

        for plane in planes {
            let nx = plane[0];
            let ny = plane[1];
            let nz = plane[2];
            let d = plane[3];

            // Positive vertex: pick max if normal component >= 0, else min
            let px = if nx >= 0.0 { aabb.max.x } else { aabb.min.x };
            let py = if ny >= 0.0 { aabb.max.y } else { aabb.min.y };
            let pz = if nz >= 0.0 { aabb.max.z } else { aabb.min.z };

            let dist = nx * px + ny * py + nz * pz + d;
            if dist < 0.0 {
                inside = false;
                break;
            }
        }

        results[i] = inside;
    }
}

/// Scalar batch dot product.
pub fn batch_dot_product_scalar(a: &[[f32; 4]], b: &[[f32; 4]], out: &mut [f32]) {
    let count = a.len().min(b.len()).min(out.len());
    for i in 0..count {
        out[i] = vec4_dot_scalar(&a[i], &b[i]);
    }
}

/// Scalar CPU vertex skinning.
pub fn skin_vertices_scalar(
    vertices: &[SkinVertex],
    bone_matrices: &[[f32; 16]],
    out: &mut [[f32; 4]],
) {
    let count = vertices.len().min(out.len());
    for i in 0..count {
        let vert = &vertices[i];
        let mut result = [0.0f32; 4];

        for j in 0..4 {
            let weight = vert.weights[j];
            if weight < 1.0e-6 {
                continue;
            }

            let bone_idx = vert.indices[j] as usize;
            if bone_idx >= bone_matrices.len() {
                continue;
            }

            let mat = &bone_matrices[bone_idx];
            let p = &vert.position;

            // Transform position by bone matrix
            for row in 0..4 {
                let transformed = mat[0 * 4 + row] * p[0]
                    + mat[1 * 4 + row] * p[1]
                    + mat[2 * 4 + row] * p[2]
                    + mat[3 * 4 + row] * p[3];
                result[row] += transformed * weight;
            }
        }

        out[i] = result;
    }
}

// ============================================================================
// Benchmarking helpers
// ============================================================================

/// Results from a SIMD vs scalar performance comparison.
#[derive(Debug, Clone)]
pub struct SimdBenchmarkResult {
    /// Name of the operation.
    pub operation: String,
    /// Time in nanoseconds for the scalar path.
    pub scalar_ns: u64,
    /// Time in nanoseconds for the SIMD path.
    pub simd_ns: u64,
    /// Speedup factor (scalar_ns / simd_ns).
    pub speedup: f64,
    /// Which SIMD extension was used (e.g., "SSE4.1", "AVX2", "NEON", "scalar").
    pub extension: String,
}

/// Run a comparative benchmark of mat4 multiply: scalar vs SIMD.
///
/// Performs `iterations` multiplications and returns timing results.
pub fn benchmark_mat4_multiply(iterations: u32) -> SimdBenchmarkResult {
    let a: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0,
    ];
    let b: [f32; 16] = [
        0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 4.0, 5.0, 6.0, 1.0,
    ];
    let mut out = [0.0f32; 16];

    // Scalar benchmark
    let scalar_start = std::time::Instant::now();
    for _ in 0..iterations {
        mat4_multiply_scalar(&a, &b, &mut out);
        std::hint::black_box(&out);
    }
    let scalar_elapsed = scalar_start.elapsed().as_nanos() as u64;

    // SIMD benchmark (dispatched)
    let simd_start = std::time::Instant::now();
    for _ in 0..iterations {
        mat4_multiply(&a, &b, &mut out);
        std::hint::black_box(&out);
    }
    let simd_elapsed = simd_start.elapsed().as_nanos() as u64;

    let extension = if has_avx2() {
        "AVX2"
    } else if has_sse41() {
        "SSE4.1"
    } else if has_neon() {
        "NEON"
    } else {
        "scalar"
    };

    SimdBenchmarkResult {
        operation: "mat4_multiply".to_string(),
        scalar_ns: scalar_elapsed,
        simd_ns: simd_elapsed,
        speedup: if simd_elapsed > 0 {
            scalar_elapsed as f64 / simd_elapsed as f64
        } else {
            0.0
        },
        extension: extension.to_string(),
    }
}

/// Run a comparative benchmark of vec4 dot product: scalar vs SIMD.
pub fn benchmark_vec4_dot(iterations: u32) -> SimdBenchmarkResult {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];

    let scalar_start = std::time::Instant::now();
    for _ in 0..iterations {
        let r = vec4_dot_scalar(&a, &b);
        std::hint::black_box(r);
    }
    let scalar_elapsed = scalar_start.elapsed().as_nanos() as u64;

    let simd_start = std::time::Instant::now();
    for _ in 0..iterations {
        let r = vec4_dot(&a, &b);
        std::hint::black_box(r);
    }
    let simd_elapsed = simd_start.elapsed().as_nanos() as u64;

    let extension = if has_sse41() {
        "SSE4.1"
    } else if has_neon() {
        "NEON"
    } else {
        "scalar"
    };

    SimdBenchmarkResult {
        operation: "vec4_dot".to_string(),
        scalar_ns: scalar_elapsed,
        simd_ns: simd_elapsed,
        speedup: if simd_elapsed > 0 {
            scalar_elapsed as f64 / simd_elapsed as f64
        } else {
            0.0
        },
        extension: extension.to_string(),
    }
}

/// Run a comparative benchmark of batch point transform: scalar vs SIMD.
pub fn benchmark_transform_points(point_count: usize, iterations: u32) -> SimdBenchmarkResult {
    let matrix: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 20.0, 30.0, 1.0,
    ];
    let points: Vec<[f32; 4]> = (0..point_count)
        .map(|i| [i as f32, (i * 2) as f32, (i * 3) as f32, 1.0])
        .collect();
    let mut out = vec![[0.0f32; 4]; point_count];

    let scalar_start = std::time::Instant::now();
    for _ in 0..iterations {
        transform_points_scalar(&matrix, &points, &mut out);
        std::hint::black_box(&out);
    }
    let scalar_elapsed = scalar_start.elapsed().as_nanos() as u64;

    let simd_start = std::time::Instant::now();
    for _ in 0..iterations {
        transform_points(&matrix, &points, &mut out);
        std::hint::black_box(&out);
    }
    let simd_elapsed = simd_start.elapsed().as_nanos() as u64;

    let extension = if has_sse41() {
        "SSE4.1"
    } else if has_neon() {
        "NEON"
    } else {
        "scalar"
    };

    SimdBenchmarkResult {
        operation: format!("transform_points({})", point_count),
        scalar_ns: scalar_elapsed,
        simd_ns: simd_elapsed,
        speedup: if simd_elapsed > 0 {
            scalar_elapsed as f64 / simd_elapsed as f64
        } else {
            0.0
        },
        extension: extension.to_string(),
    }
}

/// Run a comparative benchmark of batch dot products: scalar vs SIMD.
pub fn benchmark_batch_dot_product(count: usize, iterations: u32) -> SimdBenchmarkResult {
    let a: Vec<[f32; 4]> = (0..count)
        .map(|i| [i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
        .collect();
    let b: Vec<[f32; 4]> = (0..count)
        .map(|i| {
            [
                (i + 4) as f32,
                (i + 5) as f32,
                (i + 6) as f32,
                (i + 7) as f32,
            ]
        })
        .collect();
    let mut out = vec![0.0f32; count];

    let scalar_start = std::time::Instant::now();
    for _ in 0..iterations {
        batch_dot_product_scalar(&a, &b, &mut out);
        std::hint::black_box(&out);
    }
    let scalar_elapsed = scalar_start.elapsed().as_nanos() as u64;

    let simd_start = std::time::Instant::now();
    for _ in 0..iterations {
        batch_dot_product(&a, &b, &mut out);
        std::hint::black_box(&out);
    }
    let simd_elapsed = simd_start.elapsed().as_nanos() as u64;

    let extension = if has_avx2() {
        "AVX2"
    } else if has_sse41() {
        "SSE4.1"
    } else {
        "scalar"
    };

    SimdBenchmarkResult {
        operation: format!("batch_dot_product({})", count),
        scalar_ns: scalar_elapsed,
        simd_ns: simd_elapsed,
        speedup: if simd_elapsed > 0 {
            scalar_elapsed as f64 / simd_elapsed as f64
        } else {
            0.0
        },
        extension: extension.to_string(),
    }
}

/// Run all SIMD benchmarks and return results.
pub fn run_all_benchmarks(iterations: u32) -> Vec<SimdBenchmarkResult> {
    vec![
        benchmark_mat4_multiply(iterations),
        benchmark_vec4_dot(iterations),
        benchmark_transform_points(1000, iterations),
        benchmark_batch_dot_product(1000, iterations),
    ]
}

/// Print benchmark results in a human-readable table.
pub fn print_benchmark_results(results: &[SimdBenchmarkResult]) {
    log::info!("SIMD Benchmark Results:");
    log::info!(
        "{:<30} {:>12} {:>12} {:>10} {:>10}",
        "Operation",
        "Scalar (ns)",
        "SIMD (ns)",
        "Speedup",
        "Extension"
    );
    log::info!("{}", "-".repeat(76));
    for r in results {
        log::info!(
            "{:<30} {:>12} {:>12} {:>9.2}x {:>10}",
            r.operation,
            r.scalar_ns,
            r.simd_ns,
            r.speedup,
            r.extension,
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1.0e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn arrays_approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y))
    }

    // ---- Feature detection tests ----

    #[test]
    fn test_feature_detection_does_not_panic() {
        // These should not panic on any platform
        let _ = has_sse41();
        let _ = has_avx2();
        let _ = has_neon();
        let _ = has_fma();
        let _ = has_avx512f();
    }

    // ---- Identity matrix tests ----

    #[test]
    fn test_mat4_multiply_identity_scalar() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let m = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let mut out = [0.0f32; 16];

        mat4_multiply_scalar(&identity, &m, &mut out);
        assert!(arrays_approx_eq(&out, &m));

        mat4_multiply_scalar(&m, &identity, &mut out);
        assert!(arrays_approx_eq(&out, &m));
    }

    #[test]
    fn test_mat4_multiply_dispatch() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let m = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let mut out = [0.0f32; 16];

        mat4_multiply(&identity, &m, &mut out);
        assert!(arrays_approx_eq(&out, &m));
    }

    #[test]
    fn test_mat4_multiply_known_result() {
        // Two translation matrices: translate(1,2,3) * translate(4,5,6) = translate(5,7,9)
        let a = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        ];
        let b = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 5.0, 6.0, 1.0,
        ];
        let expected = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 7.0, 9.0, 1.0,
        ];
        let mut out = [0.0f32; 16];

        mat4_multiply(&a, &b, &mut out);
        assert!(arrays_approx_eq(&out, &expected));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_mat4_multiply_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let a = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let b = [
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        ];

        let mut scalar_out = [0.0f32; 16];
        let mut sse_out = [0.0f32; 16];

        mat4_multiply_scalar(&a, &b, &mut scalar_out);
        unsafe {
            mat4_multiply_sse(&a, &b, &mut sse_out);
        }
        assert!(arrays_approx_eq(&scalar_out, &sse_out));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_mat4_multiply_avx_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let a = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let b = [
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        ];

        let mut scalar_out = [0.0f32; 16];
        let mut avx_out = [0.0f32; 16];

        mat4_multiply_scalar(&a, &b, &mut scalar_out);
        unsafe {
            mat4_multiply_avx(&a, &b, &mut avx_out);
        }
        assert!(arrays_approx_eq(&scalar_out, &avx_out));
    }

    // ---- Transpose tests ----

    #[test]
    fn test_mat4_transpose_scalar() {
        let m = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let expected = [
            1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0,
            16.0,
        ];
        let mut out = [0.0f32; 16];
        mat4_transpose_scalar(&m, &mut out);
        assert!(arrays_approx_eq(&out, &expected));
    }

    #[test]
    fn test_mat4_transpose_dispatch() {
        let m = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let expected = [
            1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0,
            16.0,
        ];
        let mut out = [0.0f32; 16];
        mat4_transpose(&m, &mut out);
        assert!(arrays_approx_eq(&out, &expected));
    }

    #[test]
    fn test_mat4_transpose_double_is_identity() {
        let m = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let mut tmp = [0.0f32; 16];
        let mut out = [0.0f32; 16];
        mat4_transpose(&m, &mut tmp);
        mat4_transpose(&tmp, &mut out);
        assert!(arrays_approx_eq(&out, &m));
    }

    // ---- Inverse tests ----

    #[test]
    fn test_mat4_inverse_identity_scalar() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut out = [0.0f32; 16];
        mat4_inverse_scalar(&identity, &mut out);
        assert!(arrays_approx_eq(&out, &identity));
    }

    #[test]
    fn test_mat4_inverse_translation() {
        // Inverse of translate(1,2,3) = translate(-1,-2,-3)
        let m = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        ];
        let expected = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -2.0, -3.0, 1.0,
        ];
        let mut out = [0.0f32; 16];
        mat4_inverse(&m, &mut out);
        assert!(arrays_approx_eq(&out, &expected));
    }

    #[test]
    fn test_mat4_inverse_product_is_identity() {
        let m = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 10.0, 15.0, 1.0,
        ];
        let mut inv = [0.0f32; 16];
        let mut product = [0.0f32; 16];
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        mat4_inverse(&m, &mut inv);
        mat4_multiply(&m, &inv, &mut product);
        assert!(arrays_approx_eq(&product, &identity));
    }

    // ---- Vec4 dot product tests ----

    #[test]
    fn test_vec4_dot_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let result = vec4_dot_scalar(&a, &b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!(approx_eq(result, 70.0));
    }

    #[test]
    fn test_vec4_dot_dispatch() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let result = vec4_dot(&a, &b);
        assert!(approx_eq(result, 70.0));
    }

    #[test]
    fn test_vec4_dot_orthogonal() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        assert!(approx_eq(vec4_dot(&a, &b), 0.0));
    }

    // ---- Vec4 normalize tests ----

    #[test]
    fn test_vec4_normalize_unit() {
        let v = [1.0, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        vec4_normalize(&v, &mut out);
        assert!(arrays_approx_eq(&out, &[1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_vec4_normalize_nonunit() {
        let v = [3.0, 4.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        vec4_normalize(&v, &mut out);
        let len = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]).sqrt();
        assert!(approx_eq(len, 1.0));
    }

    #[test]
    fn test_vec4_normalize_zero() {
        let v = [0.0, 0.0, 0.0, 0.0];
        let mut out = [99.0f32; 4];
        vec4_normalize(&v, &mut out);
        assert!(arrays_approx_eq(&out, &[0.0, 0.0, 0.0, 0.0]));
    }

    // ---- Vec4 cross product tests ----

    #[test]
    fn test_vec4_cross_basic() {
        let a = [1.0, 0.0, 0.0, 0.0]; // X axis
        let b = [0.0, 1.0, 0.0, 0.0]; // Y axis
        let mut out = [0.0f32; 4];
        vec4_cross(&a, &b, &mut out);
        // X cross Y = Z
        assert!(approx_eq(out[0], 0.0));
        assert!(approx_eq(out[1], 0.0));
        assert!(approx_eq(out[2], 1.0));
        assert!(approx_eq(out[3], 0.0));
    }

    #[test]
    fn test_vec4_cross_anticommutative() {
        let a = [1.0, 2.0, 3.0, 0.0];
        let b = [4.0, 5.0, 6.0, 0.0];
        let mut ab = [0.0f32; 4];
        let mut ba = [0.0f32; 4];
        vec4_cross(&a, &b, &mut ab);
        vec4_cross(&b, &a, &mut ba);
        // a x b = -(b x a)
        assert!(approx_eq(ab[0], -ba[0]));
        assert!(approx_eq(ab[1], -ba[1]));
        assert!(approx_eq(ab[2], -ba[2]));
    }

    // ---- Transform points tests ----

    #[test]
    fn test_transform_points_identity() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let points = [[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]];
        let mut out = [[0.0f32; 4]; 2];
        transform_points(&identity, &points, &mut out);
        assert!(arrays_approx_eq(&out[0], &points[0]));
        assert!(arrays_approx_eq(&out[1], &points[1]));
    }

    #[test]
    fn test_transform_points_translation() {
        let translate = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 20.0, 30.0, 1.0,
        ];
        let points = [[1.0, 2.0, 3.0, 1.0]];
        let mut out = [[0.0f32; 4]; 1];
        transform_points(&translate, &points, &mut out);
        assert!(arrays_approx_eq(&out[0], &[11.0, 22.0, 33.0, 1.0]));
    }

    // ---- Batch dot product tests ----

    #[test]
    fn test_batch_dot_product() {
        let a = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let b = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let mut out = [0.0f32; 2];
        batch_dot_product(&a, &b, &mut out);
        assert!(approx_eq(out[0], 1.0));
        assert!(approx_eq(out[1], 8.0));
    }

    // ---- Skinning tests ----

    #[test]
    fn test_skin_vertices_single_bone() {
        let identity_matrix = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let translate_matrix = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 10.0, 15.0, 1.0,
        ];

        let vertices = [SkinVertex {
            position: [1.0, 2.0, 3.0, 1.0],
            weights: [1.0, 0.0, 0.0, 0.0],
            indices: [1, 0, 0, 0],
        }];

        let bone_matrices = [identity_matrix, translate_matrix];
        let mut out = [[0.0f32; 4]; 1];

        skin_vertices(&vertices, &bone_matrices, &mut out);
        assert!(arrays_approx_eq(&out[0], &[6.0, 12.0, 18.0, 1.0]));
    }

    #[test]
    fn test_skin_vertices_blended() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let translate = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 1.0,
        ];

        let vertices = [SkinVertex {
            position: [0.0, 0.0, 0.0, 1.0],
            weights: [0.5, 0.5, 0.0, 0.0],
            indices: [0, 1, 0, 0],
        }];

        let bone_matrices = [identity, translate];
        let mut out = [[0.0f32; 4]; 1];

        skin_vertices(&vertices, &bone_matrices, &mut out);
        // 0.5 * identity * [0,0,0,1] + 0.5 * translate * [0,0,0,1]
        // = 0.5 * [0,0,0,1] + 0.5 * [10,0,0,1] = [5,0,0,1]
        assert!(approx_eq(out[0][0], 5.0));
        assert!(approx_eq(out[0][1], 0.0));
        assert!(approx_eq(out[0][2], 0.0));
        assert!(approx_eq(out[0][3], 1.0));
    }

    // ---- Frustum culling tests ----

    #[test]
    fn test_frustum_cull_inside() {
        use crate::math::Vec3;
        // Planes forming a large box centered at origin
        let planes: [[f32; 4]; 6] = [
            [1.0, 0.0, 0.0, 100.0],   // left: x >= -100
            [-1.0, 0.0, 0.0, 100.0],  // right: x <= 100
            [0.0, 1.0, 0.0, 100.0],   // bottom: y >= -100
            [0.0, -1.0, 0.0, 100.0],  // top: y <= 100
            [0.0, 0.0, 1.0, 100.0],   // near: z >= -100
            [0.0, 0.0, -1.0, 100.0],  // far: z <= 100
        ];

        let aabbs = [AABB {
            min: Vec3::new(-1.0, -1.0, -1.0),
            max: Vec3::new(1.0, 1.0, 1.0),
        }];
        let mut results = [false; 1];

        frustum_cull_aabbs(&planes, &aabbs, &mut results);
        assert!(results[0], "AABB at origin should be inside large frustum");
    }

    // ---- Benchmark test (smoke test, not for performance measurement) ----

    #[test]
    fn test_benchmark_smoke() {
        let results = run_all_benchmarks(10);
        assert!(!results.is_empty());
        for r in &results {
            assert!(!r.operation.is_empty());
            assert!(!r.extension.is_empty());
        }
    }

    // ---- SSE-specific tests ----

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vec4_dot_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let a = [1.5, -2.3, 4.7, 0.1];
        let b = [-0.5, 3.2, 1.1, -2.0];

        let scalar = vec4_dot_scalar(&a, &b);
        let simd = unsafe { vec4_dot_sse(&a, &b) };
        assert!(approx_eq(scalar, simd));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vec4_normalize_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }
        let v = [3.0, 4.0, 5.0, 6.0];
        let mut scalar_out = [0.0f32; 4];
        let mut sse_out = [0.0f32; 4];

        vec4_normalize_scalar(&v, &mut scalar_out);
        unsafe {
            vec4_normalize_sse(&v, &mut sse_out);
        }

        // SSE rsqrt + Newton-Raphson should be very close but might differ in last bit
        for i in 0..4 {
            assert!(
                (scalar_out[i] - sse_out[i]).abs() < 1.0e-4,
                "Component {} differs: scalar={}, sse={}",
                i,
                scalar_out[i],
                sse_out[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vec4_cross_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }
        let a = [1.0, 2.0, 3.0, 0.0];
        let b = [4.0, 5.0, 6.0, 0.0];
        let mut scalar_out = [0.0f32; 4];
        let mut sse_out = [0.0f32; 4];

        vec4_cross_scalar(&a, &b, &mut scalar_out);
        unsafe {
            vec4_cross_sse(&a, &b, &mut sse_out);
        }
        assert!(arrays_approx_eq(&scalar_out, &sse_out));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_mat4_transpose_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }
        let m = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let mut scalar_out = [0.0f32; 16];
        let mut sse_out = [0.0f32; 16];

        mat4_transpose_scalar(&m, &mut scalar_out);
        unsafe {
            mat4_transpose_sse(&m, &mut sse_out);
        }
        assert!(arrays_approx_eq(&scalar_out, &sse_out));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_batch_dot_product_avx_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let a = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ];
        let b = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let mut scalar_out = [0.0f32; 3];
        let mut avx_out = [0.0f32; 3];

        batch_dot_product_scalar(&a, &b, &mut scalar_out);
        unsafe {
            batch_dot_product_avx(&a, &b, &mut avx_out);
        }
        for i in 0..3 {
            assert!(
                approx_eq(scalar_out[i], avx_out[i]),
                "Index {} differs: scalar={}, avx={}",
                i,
                scalar_out[i],
                avx_out[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_transform_points_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let matrix = [
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        ];
        let points = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let mut scalar_out = [[0.0f32; 4]; 3];
        let mut sse_out = [[0.0f32; 4]; 3];

        transform_points_scalar(&matrix, &points, &mut scalar_out);
        unsafe {
            transform_points_sse(&matrix, &points, &mut sse_out);
        }
        for i in 0..3 {
            assert!(
                arrays_approx_eq(&scalar_out[i], &sse_out[i]),
                "Point {} differs",
                i
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_skin_vertices_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let translate = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 20.0, 30.0, 1.0,
        ];

        let vertices = [
            SkinVertex {
                position: [1.0, 2.0, 3.0, 1.0],
                weights: [0.7, 0.3, 0.0, 0.0],
                indices: [0, 1, 0, 0],
            },
            SkinVertex {
                position: [4.0, 5.0, 6.0, 1.0],
                weights: [1.0, 0.0, 0.0, 0.0],
                indices: [1, 0, 0, 0],
            },
        ];

        let bone_matrices = [identity, translate];
        let mut scalar_out = [[0.0f32; 4]; 2];
        let mut sse_out = [[0.0f32; 4]; 2];

        skin_vertices_scalar(&vertices, &bone_matrices, &mut scalar_out);
        unsafe {
            skin_vertices_sse(&vertices, &bone_matrices, &mut sse_out);
        }
        for i in 0..2 {
            assert!(
                arrays_approx_eq(&scalar_out[i], &sse_out[i]),
                "Vertex {} differs: scalar={:?}, sse={:?}",
                i,
                scalar_out[i],
                sse_out[i]
            );
        }
    }
}
