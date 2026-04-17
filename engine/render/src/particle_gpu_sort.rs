// engine/render/src/particle_gpu_sort.rs
//
// GPU particle sorting system for the Genovo engine.
//
// Implements bitonic merge sort on the GPU for sorting particles by depth
// (or any other key) enabling correct alpha-blended particle rendering.
//
// Features:
// - Bitonic sort algorithm adapted for GPU compute dispatches.
// - Depth-based sort key generation from camera position.
// - Indirect dispatch for variable particle counts.
// - Sort stability via secondary key (particle ID).
// - Multi-pass sorting for large particle counts.
// - Key-value pair sorting (sort keys, particle indices separately).

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Workgroup size for GPU compute dispatches.
const WORKGROUP_SIZE: u32 = 256;

/// Maximum particles that can be sorted in a single dispatch.
const MAX_SORT_COUNT: u32 = 1 << 22; // ~4 million

/// Number of elements processed per thread.
const ELEMENTS_PER_THREAD: u32 = 2;

/// Bits used for depth encoding in the sort key.
const DEPTH_KEY_BITS: u32 = 24;

/// Bits used for stability index in the sort key.
const STABILITY_KEY_BITS: u32 = 8;

// ---------------------------------------------------------------------------
// Sort Key
// ---------------------------------------------------------------------------

/// A sort key combining depth and stability information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SortKey {
    /// The combined key value: upper bits = depth, lower bits = stability index.
    pub value: u32,
}

impl SortKey {
    /// Create a sort key from a depth value (0.0 = near, 1.0 = far) and stability index.
    pub fn from_depth(depth: f32, stability_index: u32) -> Self {
        let depth_clamped = depth.clamp(0.0, 1.0);
        let depth_bits = (depth_clamped * ((1u32 << DEPTH_KEY_BITS) - 1) as f32) as u32;
        let stability_bits = stability_index & ((1u32 << STABILITY_KEY_BITS) - 1);
        Self {
            value: (depth_bits << STABILITY_KEY_BITS) | stability_bits,
        }
    }

    /// Create a sort key from a raw distance and particle ID.
    pub fn from_distance(distance: f32, particle_id: u32) -> Self {
        // Flip bits so that farther particles sort first (for back-to-front).
        let max_depth_val = (1u32 << DEPTH_KEY_BITS) - 1;
        let depth_bits = max_depth_val.saturating_sub(
            (distance.clamp(0.0, 1000.0) / 1000.0 * max_depth_val as f32) as u32,
        );
        let stability_bits = particle_id & ((1u32 << STABILITY_KEY_BITS) - 1);
        Self {
            value: (depth_bits << STABILITY_KEY_BITS) | stability_bits,
        }
    }

    /// Extract the depth portion of the key.
    pub fn depth_value(&self) -> f32 {
        let depth_bits = self.value >> STABILITY_KEY_BITS;
        depth_bits as f32 / ((1u32 << DEPTH_KEY_BITS) - 1) as f32
    }

    /// Extract the stability index portion.
    pub fn stability_index(&self) -> u32 {
        self.value & ((1u32 << STABILITY_KEY_BITS) - 1)
    }
}

// ---------------------------------------------------------------------------
// Sort Key-Value Pair
// ---------------------------------------------------------------------------

/// A key-value pair for sorting, where key is the sort criterion and
/// value is the index into the particle buffer.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SortPair {
    /// Sort key (e.g., encoded depth).
    pub key: u32,
    /// Value (particle buffer index).
    pub value: u32,
}

impl SortPair {
    /// Create a new sort pair.
    pub fn new(key: u32, value: u32) -> Self {
        Self { key, value }
    }

    /// Compare two pairs for bitonic sort (ascending order).
    pub fn compare_ascending(a: &Self, b: &Self) -> bool {
        a.key <= b.key
    }

    /// Compare two pairs for bitonic sort (descending order).
    pub fn compare_descending(a: &Self, b: &Self) -> bool {
        a.key >= b.key
    }

    /// Size of a pair in bytes.
    pub fn stride() -> usize {
        std::mem::size_of::<Self>()
    }
}

// ---------------------------------------------------------------------------
// Bitonic Sort Passes
// ---------------------------------------------------------------------------

/// Describes a single bitonic sort pass (compare-and-swap at a given stride).
#[derive(Debug, Clone, Copy)]
pub struct BitonicSortPass {
    /// The block size for this pass (power of 2).
    pub block_size: u32,
    /// The compare stride for this pass (power of 2, <= block_size).
    pub compare_stride: u32,
    /// Sort direction for this pass (true = ascending sub-sequences).
    pub ascending: bool,
}

/// Generate the sequence of bitonic sort passes for N elements.
pub fn generate_bitonic_passes(count: u32) -> Vec<BitonicSortPass> {
    let n = count.next_power_of_two();
    let mut passes = Vec::new();

    let mut block_size = 2u32;
    while block_size <= n {
        let mut stride = block_size / 2;
        while stride >= 1 {
            passes.push(BitonicSortPass {
                block_size,
                compare_stride: stride,
                ascending: true,
            });
            stride /= 2;
        }
        block_size *= 2;
    }

    passes
}

/// Compute the number of passes needed for a given element count.
pub fn pass_count(count: u32) -> u32 {
    let n = count.next_power_of_two();
    let log_n = (n as f32).log2() as u32;
    log_n * (log_n + 1) / 2
}

// ---------------------------------------------------------------------------
// CPU Bitonic Sort (Reference / Fallback)
// ---------------------------------------------------------------------------

/// Perform a CPU-side bitonic sort on key-value pairs (for testing and fallback).
pub fn bitonic_sort_cpu(pairs: &mut [SortPair], ascending: bool) {
    let n = pairs.len();
    if n <= 1 {
        return;
    }

    // Pad to power of 2 if needed.
    let padded_n = n.next_power_of_two();
    let mut padded = Vec::with_capacity(padded_n);
    padded.extend_from_slice(pairs);
    while padded.len() < padded_n {
        padded.push(SortPair::new(if ascending { u32::MAX } else { 0 }, u32::MAX));
    }

    // Bitonic sort.
    let mut block_size = 2usize;
    while block_size <= padded_n {
        let mut stride = block_size / 2;
        while stride >= 1 {
            for i in 0..padded_n {
                let partner = i ^ stride;
                if partner > i && partner < padded_n {
                    let dir = ((i / block_size) % 2 == 0) == ascending;
                    let should_swap = if dir {
                        padded[i].key > padded[partner].key
                    } else {
                        padded[i].key < padded[partner].key
                    };
                    if should_swap {
                        padded.swap(i, partner);
                    }
                }
            }
            stride /= 2;
        }
        block_size *= 2;
    }

    // Copy back.
    pairs.copy_from_slice(&padded[..n]);
}

/// Verify that an array of sort pairs is sorted in ascending order by key.
pub fn verify_sorted_ascending(pairs: &[SortPair]) -> bool {
    for i in 1..pairs.len() {
        if pairs[i].key < pairs[i - 1].key {
            return false;
        }
    }
    true
}

/// Verify that an array of sort pairs is sorted in descending order by key.
pub fn verify_sorted_descending(pairs: &[SortPair]) -> bool {
    for i in 1..pairs.len() {
        if pairs[i].key > pairs[i - 1].key {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Depth Key Generation
// ---------------------------------------------------------------------------

/// Generate depth sort keys for a set of particle positions.
pub fn generate_depth_keys(
    positions: &[[f32; 3]],
    camera_position: [f32; 3],
    camera_forward: [f32; 3],
    near: f32,
    far: f32,
) -> Vec<SortPair> {
    let range = far - near;
    let inv_range = if range > 1e-8 { 1.0 / range } else { 1.0 };

    positions
        .iter()
        .enumerate()
        .map(|(i, pos)| {
            let dx = pos[0] - camera_position[0];
            let dy = pos[1] - camera_position[1];
            let dz = pos[2] - camera_position[2];

            // Project onto camera forward to get view-space depth.
            let depth = dx * camera_forward[0] + dy * camera_forward[1] + dz * camera_forward[2];
            let normalized = ((depth - near) * inv_range).clamp(0.0, 1.0);

            let key = SortKey::from_depth(normalized, i as u32);
            SortPair::new(key.value, i as u32)
        })
        .collect()
}

/// Generate squared distance sort keys (faster, no sqrt needed).
pub fn generate_distance_keys(
    positions: &[[f32; 3]],
    camera_position: [f32; 3],
) -> Vec<SortPair> {
    let max_dist_sq = 1_000_000.0f32; // 1000 units max
    let inv_max = 1.0 / max_dist_sq;

    positions
        .iter()
        .enumerate()
        .map(|(i, pos)| {
            let dx = pos[0] - camera_position[0];
            let dy = pos[1] - camera_position[1];
            let dz = pos[2] - camera_position[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            // Far particles first (back-to-front) for alpha blending.
            let normalized = (dist_sq * inv_max).clamp(0.0, 1.0);
            let max_key = (1u32 << DEPTH_KEY_BITS) - 1;
            let key = max_key - (normalized * max_key as f32) as u32;

            SortPair::new(key, i as u32)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GPU Sort Dispatch Plan
// ---------------------------------------------------------------------------

/// Describes a full GPU sort dispatch plan.
#[derive(Debug, Clone)]
pub struct GpuSortPlan {
    /// Number of elements to sort.
    pub element_count: u32,
    /// Padded element count (next power of 2).
    pub padded_count: u32,
    /// Sequence of sort passes.
    pub passes: Vec<BitonicSortPass>,
    /// Number of workgroups per dispatch.
    pub workgroups: u32,
    /// Total dispatches needed.
    pub total_dispatches: u32,
    /// Whether to sort ascending (front-to-back) or descending (back-to-front).
    pub ascending: bool,
}

impl GpuSortPlan {
    /// Create a sort plan for the given element count.
    pub fn new(element_count: u32, ascending: bool) -> Self {
        let padded_count = element_count.next_power_of_two().min(MAX_SORT_COUNT);
        let passes = generate_bitonic_passes(padded_count);
        let workgroups = (padded_count + WORKGROUP_SIZE * ELEMENTS_PER_THREAD - 1)
            / (WORKGROUP_SIZE * ELEMENTS_PER_THREAD);

        Self {
            element_count,
            padded_count,
            total_dispatches: passes.len() as u32,
            passes,
            workgroups,
            ascending,
        }
    }

    /// Estimate the total number of compare-and-swap operations.
    pub fn total_comparisons(&self) -> u64 {
        self.passes.len() as u64 * (self.padded_count as u64 / 2)
    }

    /// Estimate the GPU buffer size needed (in bytes).
    pub fn buffer_size_bytes(&self) -> usize {
        self.padded_count as usize * SortPair::stride()
    }

    /// Whether the sort can fit in a single dispatch.
    pub fn is_single_dispatch(&self) -> bool {
        self.padded_count <= WORKGROUP_SIZE * ELEMENTS_PER_THREAD
    }
}

// ---------------------------------------------------------------------------
// Indirect Dispatch Arguments
// ---------------------------------------------------------------------------

/// Indirect dispatch arguments for compute shaders.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct IndirectDispatchArgs {
    /// Number of workgroups in X dimension.
    pub x: u32,
    /// Number of workgroups in Y dimension (usually 1).
    pub y: u32,
    /// Number of workgroups in Z dimension (usually 1).
    pub z: u32,
}

impl IndirectDispatchArgs {
    /// Create dispatch args for a given number of elements.
    pub fn for_elements(element_count: u32) -> Self {
        let workgroups = (element_count + WORKGROUP_SIZE * ELEMENTS_PER_THREAD - 1)
            / (WORKGROUP_SIZE * ELEMENTS_PER_THREAD);
        Self {
            x: workgroups,
            y: 1,
            z: 1,
        }
    }

    /// Size of this struct in bytes.
    pub fn stride() -> usize {
        std::mem::size_of::<Self>()
    }
}

// ---------------------------------------------------------------------------
// Sort Statistics
// ---------------------------------------------------------------------------

/// Statistics from a GPU sort operation.
#[derive(Debug, Clone, Default)]
pub struct GpuSortStats {
    /// Number of elements sorted.
    pub element_count: u32,
    /// Number of sort passes executed.
    pub pass_count: u32,
    /// Number of GPU dispatches issued.
    pub dispatch_count: u32,
    /// Total GPU time in microseconds (if profiled).
    pub gpu_time_us: f64,
    /// Sort buffer size in bytes.
    pub buffer_size_bytes: usize,
    /// Whether the sort was verified correct (debug only).
    pub verified: bool,
}

// ---------------------------------------------------------------------------
// GPU Particle Sort Manager
// ---------------------------------------------------------------------------

/// Manages GPU particle sorting across multiple particle systems.
#[derive(Debug)]
pub struct GpuParticleSortManager {
    /// Active sort plans keyed by particle system ID.
    pub plans: Vec<(u64, GpuSortPlan)>,
    /// Reusable key-value buffer (CPU side, for fallback).
    pub cpu_buffer: Vec<SortPair>,
    /// Whether to use CPU fallback instead of GPU sort.
    pub use_cpu_fallback: bool,
    /// Maximum particle count across all systems.
    pub max_particles: u32,
    /// Statistics from the last frame.
    pub stats: GpuSortStats,
    /// Whether to verify sort results in debug mode.
    pub debug_verify: bool,
}

impl GpuParticleSortManager {
    /// Create a new sort manager.
    pub fn new() -> Self {
        Self {
            plans: Vec::new(),
            cpu_buffer: Vec::new(),
            use_cpu_fallback: false,
            max_particles: 0,
            stats: GpuSortStats::default(),
            debug_verify: false,
        }
    }

    /// Create a sort plan for a particle system.
    pub fn create_plan(&mut self, system_id: u64, particle_count: u32, ascending: bool) {
        let plan = GpuSortPlan::new(particle_count, ascending);
        self.max_particles = self.max_particles.max(plan.padded_count);
        self.plans.push((system_id, plan));
    }

    /// Clear all plans (called at the start of each frame).
    pub fn clear_plans(&mut self) {
        self.plans.clear();
        self.stats = GpuSortStats::default();
    }

    /// Perform CPU fallback sort for a set of particles.
    pub fn sort_cpu_fallback(
        &mut self,
        positions: &[[f32; 3]],
        camera_pos: [f32; 3],
    ) -> Vec<u32> {
        let mut pairs = generate_distance_keys(positions, camera_pos);
        bitonic_sort_cpu(&mut pairs, true);

        if self.debug_verify {
            self.stats.verified = verify_sorted_ascending(&pairs);
        }

        self.stats.element_count = pairs.len() as u32;
        self.stats.pass_count = pass_count(pairs.len() as u32);

        pairs.iter().map(|p| p.value).collect()
    }

    /// Returns the total buffer memory needed for all active plans.
    pub fn total_buffer_size(&self) -> usize {
        self.plans.iter().map(|(_, p)| p.buffer_size_bytes()).sum()
    }

    /// Returns the total dispatch count across all plans.
    pub fn total_dispatches(&self) -> u32 {
        self.plans.iter().map(|(_, p)| p.total_dispatches).sum()
    }
}

// ---------------------------------------------------------------------------
// Sort Mode
// ---------------------------------------------------------------------------

/// Sorting mode for particles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParticleSortMode {
    /// No sorting (fastest, may have artifacts with alpha blending).
    None,
    /// Sort by view-space depth (back-to-front).
    ByDepth,
    /// Sort by distance from camera (back-to-front).
    ByDistance,
    /// Sort by age (oldest first).
    ByAge,
    /// Sort by a custom attribute.
    ByCustom,
    /// Reverse depth sort (front-to-back, for opaque particles).
    ByDepthReverse,
}

impl ParticleSortMode {
    /// Whether this mode requires a sort pass.
    pub fn requires_sort(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Whether this mode sorts back-to-front.
    pub fn is_back_to_front(self) -> bool {
        matches!(self, Self::ByDepth | Self::ByDistance | Self::ByAge)
    }
}
