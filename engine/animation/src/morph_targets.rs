//! Morph target (blend shape) animation system.
//!
//! Morph targets allow deforming a mesh by blending between a base shape
//! and one or more target shapes. Each target stores per-vertex deltas
//! (position and normal offsets) that are added to the base mesh weighted
//! by a blend factor.
//!
//! Common uses:
//! - Facial animation (smile, blink, eyebrow raise, phonemes)
//! - Damage/destruction states
//! - Muscle flexing
//! - Corrective shapes for skeleton deformation
//!
//! # Pipeline
//!
//! 1. Load base mesh vertices and morph targets from asset (glTF, FBX)
//! 2. Set target weights via gameplay code or animation curves
//! 3. Call `apply_morph_targets` to compute final vertex positions
//! 4. Upload to GPU vertex buffer
//!
//! # Sparse Morph Targets
//!
//! For efficiency, morph targets can be stored in sparse format where only
//! vertices with non-zero deltas are stored, significantly reducing memory
//! for targets that affect small regions of a mesh (e.g., lip movements
//! on a full-body mesh).

use glam::Vec3;

// ===========================================================================
// Vertex
// ===========================================================================

/// A mesh vertex with position and normal.
///
/// This is the minimal vertex data needed for morph target blending.
/// In a production engine, this would reference the full vertex format
/// including UVs, tangents, bone weights, etc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorphVertex {
    /// World-space or model-space position.
    pub position: Vec3,
    /// Vertex normal (should be unit length).
    pub normal: Vec3,
}

impl MorphVertex {
    /// Create a new vertex.
    pub fn new(position: Vec3, normal: Vec3) -> Self {
        Self { position, normal }
    }

    /// Create a vertex at the origin with up-pointing normal.
    pub fn origin() -> Self {
        Self {
            position: Vec3::ZERO,
            normal: Vec3::Y,
        }
    }
}

impl Default for MorphVertex {
    fn default() -> Self {
        Self::origin()
    }
}

// ===========================================================================
// VertexDelta
// ===========================================================================

/// A per-vertex delta for a morph target.
///
/// Stores the position and normal offset to apply to a specific vertex
/// when the morph target is active.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexDelta {
    /// Position offset to add to the base vertex position.
    pub position_delta: Vec3,
    /// Normal offset to add to the base vertex normal.
    pub normal_delta: Vec3,
}

impl VertexDelta {
    /// Create a new delta with position and normal offsets.
    pub fn new(position_delta: Vec3, normal_delta: Vec3) -> Self {
        Self {
            position_delta,
            normal_delta,
        }
    }

    /// Create a delta with only a position offset (no normal change).
    pub fn position_only(delta: Vec3) -> Self {
        Self {
            position_delta: delta,
            normal_delta: Vec3::ZERO,
        }
    }

    /// Create a zero delta (no effect).
    pub fn zero() -> Self {
        Self {
            position_delta: Vec3::ZERO,
            normal_delta: Vec3::ZERO,
        }
    }

    /// Whether this delta is effectively zero.
    pub fn is_zero(&self) -> bool {
        self.position_delta.length_squared() < f32::EPSILON
            && self.normal_delta.length_squared() < f32::EPSILON
    }

    /// Scale this delta by a weight factor.
    pub fn scaled(&self, weight: f32) -> Self {
        Self {
            position_delta: self.position_delta * weight,
            normal_delta: self.normal_delta * weight,
        }
    }
}

impl Default for VertexDelta {
    fn default() -> Self {
        Self::zero()
    }
}

// ===========================================================================
// SparseVertexDelta
// ===========================================================================

/// A sparse vertex delta that stores the vertex index alongside the delta.
///
/// Used for morph targets that only affect a subset of the mesh's vertices,
/// avoiding the memory cost of storing zeros for unaffected vertices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseVertexDelta {
    /// Index of the vertex this delta applies to.
    pub vertex_index: u32,
    /// The position and normal delta.
    pub delta: VertexDelta,
}

impl SparseVertexDelta {
    /// Create a new sparse delta.
    pub fn new(vertex_index: u32, delta: VertexDelta) -> Self {
        Self {
            vertex_index,
            delta,
        }
    }
}

// ===========================================================================
// MorphTarget
// ===========================================================================

/// A single morph target (blend shape).
///
/// Stores the vertex deltas that define how the mesh deforms when this
/// target is active. Can be either dense (one delta per vertex) or sparse
/// (only non-zero deltas stored).
#[derive(Debug, Clone)]
pub struct MorphTarget {
    /// Human-readable name (e.g., "Smile", "BlinkLeft", "Phoneme_A").
    pub name: String,

    /// Dense vertex deltas (one per vertex in the base mesh).
    /// Empty if using sparse representation.
    pub vertex_deltas: Vec<VertexDelta>,

    /// Sparse vertex deltas (only non-zero entries).
    /// Empty if using dense representation.
    pub sparse_deltas: Vec<SparseVertexDelta>,

    /// Whether this target uses sparse storage.
    pub is_sparse: bool,

    /// Default weight for this target (typically 0.0).
    pub default_weight: f32,

    /// Minimum allowed weight (typically 0.0, but can be negative for
    /// inverse/corrective targets).
    pub min_weight: f32,

    /// Maximum allowed weight (typically 1.0, but can exceed 1.0 for
    /// exaggeration).
    pub max_weight: f32,
}

impl MorphTarget {
    /// Create a new dense morph target.
    pub fn new(name: impl Into<String>, vertex_deltas: Vec<VertexDelta>) -> Self {
        Self {
            name: name.into(),
            vertex_deltas,
            sparse_deltas: Vec::new(),
            is_sparse: false,
            default_weight: 0.0,
            min_weight: 0.0,
            max_weight: 1.0,
        }
    }

    /// Create a new sparse morph target.
    pub fn sparse(name: impl Into<String>, sparse_deltas: Vec<SparseVertexDelta>) -> Self {
        Self {
            name: name.into(),
            vertex_deltas: Vec::new(),
            sparse_deltas,
            is_sparse: true,
            default_weight: 0.0,
            min_weight: 0.0,
            max_weight: 1.0,
        }
    }

    /// Create a morph target from two mesh snapshots (base and target).
    ///
    /// Computes the deltas by subtracting base positions/normals from
    /// the target positions/normals.
    pub fn from_meshes(
        name: impl Into<String>,
        base_vertices: &[MorphVertex],
        target_vertices: &[MorphVertex],
    ) -> Self {
        assert_eq!(
            base_vertices.len(),
            target_vertices.len(),
            "Base and target vertex counts must match"
        );

        let deltas: Vec<VertexDelta> = base_vertices
            .iter()
            .zip(target_vertices.iter())
            .map(|(base, target)| VertexDelta {
                position_delta: target.position - base.position,
                normal_delta: target.normal - base.normal,
            })
            .collect();

        Self::new(name, deltas)
    }

    /// Convert a dense morph target to sparse, removing zero deltas.
    ///
    /// This can significantly reduce memory for targets that only affect
    /// a small portion of the mesh.
    pub fn to_sparse(&self) -> Self {
        if self.is_sparse {
            return self.clone();
        }

        let sparse: Vec<SparseVertexDelta> = self
            .vertex_deltas
            .iter()
            .enumerate()
            .filter(|(_, d)| !d.is_zero())
            .map(|(i, d)| SparseVertexDelta::new(i as u32, *d))
            .collect();

        Self {
            name: self.name.clone(),
            vertex_deltas: Vec::new(),
            sparse_deltas: sparse,
            is_sparse: true,
            default_weight: self.default_weight,
            min_weight: self.min_weight,
            max_weight: self.max_weight,
        }
    }

    /// Convert a sparse morph target to dense.
    pub fn to_dense(&self, vertex_count: usize) -> Self {
        if !self.is_sparse {
            return self.clone();
        }

        let mut deltas = vec![VertexDelta::zero(); vertex_count];
        for sd in &self.sparse_deltas {
            let idx = sd.vertex_index as usize;
            if idx < deltas.len() {
                deltas[idx] = sd.delta;
            }
        }

        Self {
            name: self.name.clone(),
            vertex_deltas: deltas,
            sparse_deltas: Vec::new(),
            is_sparse: false,
            default_weight: self.default_weight,
            min_weight: self.min_weight,
            max_weight: self.max_weight,
        }
    }

    /// Get the delta for a specific vertex.
    pub fn delta_for_vertex(&self, vertex_index: usize) -> VertexDelta {
        if self.is_sparse {
            self.sparse_deltas
                .iter()
                .find(|sd| sd.vertex_index as usize == vertex_index)
                .map(|sd| sd.delta)
                .unwrap_or_default()
        } else {
            self.vertex_deltas
                .get(vertex_index)
                .copied()
                .unwrap_or_default()
        }
    }

    /// Number of non-zero deltas in this target.
    pub fn active_delta_count(&self) -> usize {
        if self.is_sparse {
            self.sparse_deltas.len()
        } else {
            self.vertex_deltas.iter().filter(|d| !d.is_zero()).count()
        }
    }

    /// Memory usage estimate in bytes.
    pub fn memory_bytes(&self) -> usize {
        if self.is_sparse {
            self.sparse_deltas.len() * std::mem::size_of::<SparseVertexDelta>()
        } else {
            self.vertex_deltas.len() * std::mem::size_of::<VertexDelta>()
        }
    }
}

// ===========================================================================
// MorphTargetWeights
// ===========================================================================

/// Per-target blend weights for a morph target set.
///
/// Manages the current weight for each target and provides methods for
/// setting, clamping, and animating weights.
#[derive(Debug, Clone)]
pub struct MorphTargetWeights {
    /// Current weight for each target, indexed by target index.
    pub weights: Vec<f32>,
    /// Previous frame's weights (for delta computation in animations).
    pub previous_weights: Vec<f32>,
}

impl MorphTargetWeights {
    /// Create weights for the given number of targets, all set to 0.
    pub fn new(target_count: usize) -> Self {
        Self {
            weights: vec![0.0; target_count],
            previous_weights: vec![0.0; target_count],
        }
    }

    /// Create weights with the given default values.
    pub fn from_defaults(defaults: &[f32]) -> Self {
        Self {
            weights: defaults.to_vec(),
            previous_weights: defaults.to_vec(),
        }
    }

    /// Set the weight for a specific target.
    pub fn set_weight(&mut self, target_index: usize, weight: f32) {
        if target_index < self.weights.len() {
            self.weights[target_index] = weight;
        }
    }

    /// Get the weight for a specific target.
    pub fn weight(&self, target_index: usize) -> f32 {
        self.weights.get(target_index).copied().unwrap_or(0.0)
    }

    /// Set all weights to zero.
    pub fn reset(&mut self) {
        for w in &mut self.weights {
            *w = 0.0;
        }
    }

    /// Clamp all weights to [0, 1].
    pub fn clamp_all(&mut self) {
        for w in &mut self.weights {
            *w = w.clamp(0.0, 1.0);
        }
    }

    /// Clamp weights using per-target min/max from the morph target set.
    pub fn clamp_with_targets(&mut self, targets: &[MorphTarget]) {
        for (i, w) in self.weights.iter_mut().enumerate() {
            if let Some(target) = targets.get(i) {
                *w = w.clamp(target.min_weight, target.max_weight);
            }
        }
    }

    /// Smoothly interpolate weights toward target values.
    ///
    /// `target_weights` are the desired weights, `speed` controls how
    /// fast the interpolation occurs (higher = faster).
    pub fn lerp_toward(&mut self, target_weights: &[f32], speed: f32, dt: f32) {
        let t = (speed * dt).clamp(0.0, 1.0);
        for (i, w) in self.weights.iter_mut().enumerate() {
            if let Some(&target) = target_weights.get(i) {
                *w = *w + (target - *w) * t;
            }
        }
    }

    /// Store current weights as previous (call at the start of each frame).
    pub fn save_previous(&mut self) {
        self.previous_weights.clone_from(&self.weights);
    }

    /// Check if any weight is non-zero.
    pub fn has_active_targets(&self) -> bool {
        self.weights.iter().any(|&w| w.abs() > f32::EPSILON)
    }

    /// Number of targets.
    pub fn target_count(&self) -> usize {
        self.weights.len()
    }
}

// ===========================================================================
// MorphTargetSet
// ===========================================================================

/// A complete morph target set for a mesh.
///
/// Contains the base mesh, all morph targets, and the current blend weights.
/// Provides the main `apply` method that computes the final deformed vertices.
#[derive(Debug, Clone)]
pub struct MorphTargetSet {
    /// Human-readable name (e.g., "FacialBlendShapes").
    pub name: String,

    /// Base mesh vertices (the neutral/rest pose).
    pub base_vertices: Vec<MorphVertex>,

    /// All morph targets in this set.
    pub targets: Vec<MorphTarget>,

    /// Current blend weights.
    pub weights: MorphTargetWeights,

    /// Cached result from the last `apply` call.
    cached_result: Vec<MorphVertex>,

    /// Whether the cache is dirty and needs recomputation.
    cache_dirty: bool,
}

impl MorphTargetSet {
    /// Create a new morph target set with the given base vertices.
    pub fn new(name: impl Into<String>, base_vertices: Vec<MorphVertex>) -> Self {
        Self {
            name: name.into(),
            base_vertices: base_vertices.clone(),
            targets: Vec::new(),
            weights: MorphTargetWeights::new(0),
            cached_result: base_vertices,
            cache_dirty: false,
        }
    }

    /// Add a morph target to the set.
    pub fn add_target(&mut self, target: MorphTarget) {
        self.targets.push(target);
        self.weights.weights.push(0.0);
        self.weights.previous_weights.push(0.0);
        self.cache_dirty = true;
    }

    /// Find a target by name and return its index.
    pub fn find_target(&self, name: &str) -> Option<usize> {
        self.targets.iter().position(|t| t.name == name)
    }

    /// Set the weight for a target by name.
    pub fn set_weight_by_name(&mut self, name: &str, weight: f32) {
        if let Some(idx) = self.find_target(name) {
            self.weights.set_weight(idx, weight);
            self.cache_dirty = true;
        }
    }

    /// Set the weight for a target by index.
    pub fn set_weight(&mut self, target_index: usize, weight: f32) {
        self.weights.set_weight(target_index, weight);
        self.cache_dirty = true;
    }

    /// Number of morph targets.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Number of base vertices.
    pub fn vertex_count(&self) -> usize {
        self.base_vertices.len()
    }

    /// Apply all active morph targets and return the deformed vertices.
    ///
    /// This is the main evaluation function. For each vertex, it computes:
    ///
    /// ```text
    /// final_pos = base_pos + sum(weight_i * delta_pos_i)
    /// final_normal = normalize(base_normal + sum(weight_i * delta_normal_i))
    /// ```
    ///
    /// Only targets with non-zero weights are processed.
    pub fn apply(&mut self) -> &[MorphVertex] {
        if !self.cache_dirty && !self.weights.has_active_targets() {
            return &self.cached_result;
        }

        self.cached_result = apply_morph_targets(
            &self.base_vertices,
            &self.targets,
            &self.weights.weights,
        );
        self.cache_dirty = false;

        &self.cached_result
    }

    /// Force recomputation on the next `apply` call.
    pub fn invalidate_cache(&mut self) {
        self.cache_dirty = true;
    }

    /// Get the cached result without recomputing.
    pub fn cached_vertices(&self) -> &[MorphVertex] {
        &self.cached_result
    }

    /// Total memory usage estimate in bytes.
    pub fn memory_bytes(&self) -> usize {
        let base = self.base_vertices.len() * std::mem::size_of::<MorphVertex>();
        let targets: usize = self.targets.iter().map(|t| t.memory_bytes()).sum();
        let cache = self.cached_result.len() * std::mem::size_of::<MorphVertex>();
        base + targets + cache
    }

    /// Reset all weights to their default values.
    pub fn reset_weights(&mut self) {
        for (i, target) in self.targets.iter().enumerate() {
            if i < self.weights.weights.len() {
                self.weights.weights[i] = target.default_weight;
            }
        }
        self.cache_dirty = true;
    }

    /// Convert all dense targets to sparse representation.
    pub fn sparsify_all(&mut self) {
        for i in 0..self.targets.len() {
            if !self.targets[i].is_sparse {
                self.targets[i] = self.targets[i].to_sparse();
            }
        }
    }

    /// Get a summary of the morph target set.
    pub fn summary(&self) -> MorphTargetSetSummary {
        MorphTargetSetSummary {
            name: self.name.clone(),
            vertex_count: self.base_vertices.len(),
            target_count: self.targets.len(),
            sparse_count: self.targets.iter().filter(|t| t.is_sparse).count(),
            dense_count: self.targets.iter().filter(|t| !t.is_sparse).count(),
            total_memory_bytes: self.memory_bytes(),
            active_targets: self.weights.weights.iter().filter(|&&w| w.abs() > f32::EPSILON).count(),
        }
    }
}

/// Summary information about a morph target set.
#[derive(Debug, Clone)]
pub struct MorphTargetSetSummary {
    pub name: String,
    pub vertex_count: usize,
    pub target_count: usize,
    pub sparse_count: usize,
    pub dense_count: usize,
    pub total_memory_bytes: usize,
    pub active_targets: usize,
}

// ===========================================================================
// Core application function
// ===========================================================================

/// Apply morph targets to a base mesh with the given weights.
///
/// This is the core morph target blending function. For each vertex:
///
/// ```text
/// final_pos = base_pos + sum(weight_i * delta_pos_i)
/// final_normal = normalize(base_normal + sum(weight_i * delta_normal_i))
/// ```
///
/// Handles both dense and sparse morph targets efficiently.
pub fn apply_morph_targets(
    base_vertices: &[MorphVertex],
    targets: &[MorphTarget],
    weights: &[f32],
) -> Vec<MorphVertex> {
    let mut result: Vec<MorphVertex> = base_vertices.to_vec();

    for (target_idx, target) in targets.iter().enumerate() {
        let weight = weights.get(target_idx).copied().unwrap_or(0.0);
        if weight.abs() < f32::EPSILON {
            continue;
        }

        if target.is_sparse {
            // Sparse path: only process stored deltas.
            for sd in &target.sparse_deltas {
                let vi = sd.vertex_index as usize;
                if vi < result.len() {
                    result[vi].position += sd.delta.position_delta * weight;
                    result[vi].normal += sd.delta.normal_delta * weight;
                }
            }
        } else {
            // Dense path: process all vertices.
            for (vi, delta) in target.vertex_deltas.iter().enumerate() {
                if vi < result.len() {
                    result[vi].position += delta.position_delta * weight;
                    result[vi].normal += delta.normal_delta * weight;
                }
            }
        }
    }

    // Renormalize normals after blending.
    for v in &mut result {
        let len = v.normal.length();
        if len > f32::EPSILON {
            v.normal /= len;
        }
    }

    result
}

// ===========================================================================
// MorphTargetAnimation
// ===========================================================================

/// Animates morph target weights over time.
///
/// Each keyframe specifies the weight values for all targets at a given
/// time. Weights are linearly interpolated between keyframes.
#[derive(Debug, Clone)]
pub struct MorphTargetAnimation {
    /// Name of this animation (e.g., "Smile", "Talking").
    pub name: String,

    /// Keyframes sorted by time.
    pub keyframes: Vec<MorphWeightKeyframe>,

    /// Total duration in seconds.
    pub duration: f32,

    /// Whether to loop.
    pub looping: bool,
}

/// A keyframe containing weights for all targets at a specific time.
#[derive(Debug, Clone)]
pub struct MorphWeightKeyframe {
    /// Time in seconds.
    pub time: f32,
    /// Weight values for each target (indexed by target index).
    pub weights: Vec<f32>,
}

impl MorphWeightKeyframe {
    /// Create a new keyframe.
    pub fn new(time: f32, weights: Vec<f32>) -> Self {
        Self { time, weights }
    }
}

impl MorphTargetAnimation {
    /// Create a new morph target animation.
    pub fn new(name: impl Into<String>, duration: f32) -> Self {
        Self {
            name: name.into(),
            keyframes: Vec::new(),
            duration,
            looping: false,
        }
    }

    /// Add a keyframe.
    pub fn add_keyframe(&mut self, keyframe: MorphWeightKeyframe) {
        self.keyframes.push(keyframe);
        self.keyframes.sort_by(|a, b| {
            a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sample the animation at the given time, returning interpolated weights.
    pub fn sample(&self, time: f32) -> Vec<f32> {
        if self.keyframes.is_empty() {
            return Vec::new();
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].weights.clone();
        }

        let t = if self.looping && self.duration > 0.0 {
            time.rem_euclid(self.duration)
        } else {
            time.clamp(0.0, self.duration)
        };

        // Before first keyframe.
        if t <= self.keyframes[0].time {
            return self.keyframes[0].weights.clone();
        }

        // After last keyframe.
        let last = self.keyframes.len() - 1;
        if t >= self.keyframes[last].time {
            return self.keyframes[last].weights.clone();
        }

        // Find bracketing keyframes.
        for i in 0..last {
            let k0 = &self.keyframes[i];
            let k1 = &self.keyframes[i + 1];
            if t >= k0.time && t <= k1.time {
                let dt = k1.time - k0.time;
                let blend = if dt > f32::EPSILON {
                    (t - k0.time) / dt
                } else {
                    0.0
                };

                let target_count = k0.weights.len().max(k1.weights.len());
                let mut result = Vec::with_capacity(target_count);
                for j in 0..target_count {
                    let w0 = k0.weights.get(j).copied().unwrap_or(0.0);
                    let w1 = k1.weights.get(j).copied().unwrap_or(0.0);
                    result.push(w0 + (w1 - w0) * blend);
                }
                return result;
            }
        }

        self.keyframes[last].weights.clone()
    }

    /// Apply this animation to a morph target set at the given time.
    pub fn apply_to(&self, target_set: &mut MorphTargetSet, time: f32) {
        let weights = self.sample(time);
        for (i, &w) in weights.iter().enumerate() {
            target_set.set_weight(i, w);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_base_mesh() -> Vec<MorphVertex> {
        vec![
            MorphVertex::new(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
            MorphVertex::new(Vec3::new(1.0, 0.0, 0.0), Vec3::Y),
            MorphVertex::new(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
            MorphVertex::new(Vec3::new(1.0, 1.0, 0.0), Vec3::Y),
        ]
    }

    fn make_smile_target() -> MorphTarget {
        MorphTarget::new(
            "Smile",
            vec![
                VertexDelta::position_only(Vec3::new(0.0, 0.1, 0.0)),
                VertexDelta::position_only(Vec3::new(0.0, 0.1, 0.0)),
                VertexDelta::position_only(Vec3::new(0.0, -0.05, 0.0)),
                VertexDelta::position_only(Vec3::new(0.0, -0.05, 0.0)),
            ],
        )
    }

    #[test]
    fn apply_zero_weight() {
        let base = make_base_mesh();
        let target = make_smile_target();
        let result = apply_morph_targets(&base, &[target], &[0.0]);
        for (i, v) in result.iter().enumerate() {
            assert!(
                (v.position - base[i].position).length() < f32::EPSILON,
                "Zero weight should not change vertices"
            );
        }
    }

    #[test]
    fn apply_full_weight() {
        let base = make_base_mesh();
        let target = make_smile_target();
        let result = apply_morph_targets(&base, &[target], &[1.0]);
        // Vertex 0 should move up by 0.1.
        assert!((result[0].position.y - 0.1).abs() < 0.001);
        // Vertex 2 should move down by 0.05.
        assert!((result[2].position.y - 0.95).abs() < 0.001);
    }

    #[test]
    fn apply_half_weight() {
        let base = make_base_mesh();
        let target = make_smile_target();
        let result = apply_morph_targets(&base, &[target], &[0.5]);
        assert!((result[0].position.y - 0.05).abs() < 0.001);
    }

    #[test]
    fn apply_multiple_targets() {
        let base = make_base_mesh();
        let smile = make_smile_target();
        let blink = MorphTarget::new(
            "Blink",
            vec![
                VertexDelta::position_only(Vec3::ZERO),
                VertexDelta::position_only(Vec3::ZERO),
                VertexDelta::position_only(Vec3::new(0.0, -0.5, 0.0)),
                VertexDelta::position_only(Vec3::new(0.0, -0.5, 0.0)),
            ],
        );
        let result = apply_morph_targets(&base, &[smile, blink], &[1.0, 1.0]);
        // Vertex 2: base.y=1.0, smile=-0.05, blink=-0.5 -> 0.45
        assert!((result[2].position.y - 0.45).abs() < 0.001);
    }

    #[test]
    fn sparse_target() {
        let base = make_base_mesh();
        let sparse = MorphTarget::sparse(
            "SparseSmile",
            vec![
                SparseVertexDelta::new(0, VertexDelta::position_only(Vec3::new(0.0, 0.2, 0.0))),
            ],
        );
        let result = apply_morph_targets(&base, &[sparse], &[1.0]);
        // Only vertex 0 should be affected.
        assert!((result[0].position.y - 0.2).abs() < 0.001);
        assert!((result[1].position - base[1].position).length() < f32::EPSILON);
    }

    #[test]
    fn dense_to_sparse_conversion() {
        let target = make_smile_target();
        let sparse = target.to_sparse();
        assert!(sparse.is_sparse);
        assert_eq!(sparse.sparse_deltas.len(), target.active_delta_count());
    }

    #[test]
    fn sparse_to_dense_conversion() {
        let sparse = MorphTarget::sparse(
            "Test",
            vec![
                SparseVertexDelta::new(1, VertexDelta::position_only(Vec3::X)),
            ],
        );
        let dense = sparse.to_dense(4);
        assert!(!dense.is_sparse);
        assert_eq!(dense.vertex_deltas.len(), 4);
        assert!(dense.vertex_deltas[0].is_zero());
        assert!(!dense.vertex_deltas[1].is_zero());
    }

    #[test]
    fn from_meshes() {
        let base = make_base_mesh();
        let target_mesh: Vec<MorphVertex> = base
            .iter()
            .map(|v| MorphVertex::new(v.position + Vec3::new(0.0, 0.5, 0.0), v.normal))
            .collect();

        let target = MorphTarget::from_meshes("MoveUp", &base, &target_mesh);
        assert_eq!(target.vertex_deltas.len(), 4);
        for d in &target.vertex_deltas {
            assert!((d.position_delta.y - 0.5).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn morph_target_weights() {
        let mut weights = MorphTargetWeights::new(3);
        assert_eq!(weights.target_count(), 3);
        assert!(!weights.has_active_targets());

        weights.set_weight(1, 0.5);
        assert!(weights.has_active_targets());
        assert!((weights.weight(1) - 0.5).abs() < f32::EPSILON);

        weights.reset();
        assert!(!weights.has_active_targets());
    }

    #[test]
    fn morph_target_weights_lerp() {
        let mut weights = MorphTargetWeights::new(2);
        let target = [1.0, 0.5];
        weights.lerp_toward(&target, 10.0, 0.1);
        // After one step with high speed, should be close to target.
        assert!(weights.weight(0) > 0.5);
        assert!(weights.weight(1) > 0.2);
    }

    #[test]
    fn morph_target_set_basic() {
        let base = make_base_mesh();
        let mut set = MorphTargetSet::new("FaceShapes", base);
        set.add_target(make_smile_target());
        assert_eq!(set.target_count(), 1);
        assert_eq!(set.vertex_count(), 4);

        set.set_weight(0, 1.0);
        let result = set.apply();
        assert!((result[0].position.y - 0.1).abs() < 0.001);
    }

    #[test]
    fn morph_target_set_find() {
        let base = make_base_mesh();
        let mut set = MorphTargetSet::new("Test", base);
        set.add_target(make_smile_target());
        assert_eq!(set.find_target("Smile"), Some(0));
        assert_eq!(set.find_target("Missing"), None);
    }

    #[test]
    fn morph_target_set_by_name() {
        let base = make_base_mesh();
        let mut set = MorphTargetSet::new("Test", base);
        set.add_target(make_smile_target());
        set.set_weight_by_name("Smile", 0.7);
        assert!((set.weights.weight(0) - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn morph_target_animation() {
        let mut anim = MorphTargetAnimation::new("SmileAnim", 1.0);
        anim.add_keyframe(MorphWeightKeyframe::new(0.0, vec![0.0]));
        anim.add_keyframe(MorphWeightKeyframe::new(0.5, vec![1.0]));
        anim.add_keyframe(MorphWeightKeyframe::new(1.0, vec![0.0]));

        let w0 = anim.sample(0.0);
        assert!((w0[0] - 0.0).abs() < f32::EPSILON);

        let w_mid = anim.sample(0.5);
        assert!((w_mid[0] - 1.0).abs() < f32::EPSILON);

        let w_quarter = anim.sample(0.25);
        assert!((w_quarter[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn morph_target_animation_looping() {
        let mut anim = MorphTargetAnimation::new("Loop", 1.0);
        anim.looping = true;
        anim.add_keyframe(MorphWeightKeyframe::new(0.0, vec![0.0]));
        anim.add_keyframe(MorphWeightKeyframe::new(1.0, vec![1.0]));

        // At t=1.5 (looped), should be 0.5.
        let w = anim.sample(1.5);
        assert!((w[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn morph_target_set_summary() {
        let base = make_base_mesh();
        let mut set = MorphTargetSet::new("Face", base);
        set.add_target(make_smile_target());
        let summary = set.summary();
        assert_eq!(summary.vertex_count, 4);
        assert_eq!(summary.target_count, 1);
        assert_eq!(summary.dense_count, 1);
        assert_eq!(summary.sparse_count, 0);
    }

    #[test]
    fn normals_renormalized() {
        let base = vec![MorphVertex::new(Vec3::ZERO, Vec3::Y)];
        let target = MorphTarget::new(
            "Test",
            vec![VertexDelta::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0))],
        );
        let result = apply_morph_targets(&base, &[target], &[1.0]);
        let normal_len = result[0].normal.length();
        assert!((normal_len - 1.0).abs() < 0.001, "Normal should be unit length, got {normal_len}");
    }
}
