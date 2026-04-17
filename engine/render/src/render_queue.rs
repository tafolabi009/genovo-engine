// engine/render/src/render_queue_v2.rs
//
// Advanced render queue with multi-criteria sorting for optimal GPU throughput.
//
// Objects are categorized into render buckets (opaque, transparent, shadow casters,
// sky, overlay) and sorted within each bucket using criteria appropriate for that
// bucket type. The queue is rebuilt each frame from scratch to reflect the
// current scene state.
//
// Sorting strategies:
//   - Opaque: front-to-back by distance (minimize overdraw), then by material
//     (minimize state changes), then by mesh (maximize instancing).
//   - Transparent: back-to-front by distance (correct blending).
//   - Shadow casters: by cascade index, then front-to-back.
//   - Sky: rendered last, no depth write.
//   - Overlay: rendered on top, sorted by priority.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Sort key construction
// ---------------------------------------------------------------------------

/// Constructs a 64-bit sort key by packing multiple fields.
///
/// Layout (MSB to LSB):
///   - Bits 63-60: Bucket (4 bits, 0-15)
///   - Bits 59-56: Priority override (4 bits, 0-15)
///   - Bits 55-40: Depth (16 bits, quantized)
///   - Bits 39-24: Material ID (16 bits)
///   - Bits 23-8:  Mesh ID (16 bits)
///   - Bits 7-0:   Sub-sort (8 bits)
pub fn build_sort_key(
    bucket: RenderBucket,
    priority: u8,
    depth_normalized: f32,
    material_id: u16,
    mesh_id: u16,
    sub_sort: u8,
) -> u64 {
    let bucket_bits = (bucket as u64 & 0xF) << 60;
    let priority_bits = (priority as u64 & 0xF) << 56;
    let depth_quantized = (depth_normalized.clamp(0.0, 1.0) * 65535.0) as u64;
    let depth_bits = (depth_quantized & 0xFFFF) << 40;
    let material_bits = (material_id as u64) << 24;
    let mesh_bits = (mesh_id as u64) << 8;
    let sub_bits = sub_sort as u64;

    bucket_bits | priority_bits | depth_bits | material_bits | mesh_bits | sub_bits
}

/// Extract the bucket from a sort key.
pub fn sort_key_bucket(key: u64) -> u8 {
    ((key >> 60) & 0xF) as u8
}

/// Extract the depth from a sort key (as normalized float).
pub fn sort_key_depth(key: u64) -> f32 {
    ((key >> 40) & 0xFFFF) as f32 / 65535.0
}

/// Extract the material ID from a sort key.
pub fn sort_key_material(key: u64) -> u16 {
    ((key >> 24) & 0xFFFF) as u16
}

// ---------------------------------------------------------------------------
// Render bucket
// ---------------------------------------------------------------------------

/// Categorization of renderable objects into rendering passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum RenderBucket {
    /// Shadow map rendering (first pass).
    ShadowCaster = 0,
    /// Depth prepass (if enabled).
    DepthPrepass = 1,
    /// Opaque geometry (main pass, front-to-back).
    Opaque = 2,
    /// Alpha-tested geometry (masked).
    AlphaTest = 3,
    /// Transparent geometry (back-to-front).
    Transparent = 4,
    /// Additive effects (particles, glow).
    Additive = 5,
    /// Sky/environment (rendered at max depth).
    Sky = 6,
    /// Post-process (screen-space effects).
    PostProcess = 7,
    /// Debug overlay (on top of everything).
    DebugOverlay = 8,
    /// UI overlay (final layer).
    UIOverlay = 9,
}

impl RenderBucket {
    /// Whether objects in this bucket should be sorted front-to-back.
    pub fn is_front_to_back(&self) -> bool {
        matches!(self, RenderBucket::Opaque | RenderBucket::AlphaTest | RenderBucket::ShadowCaster | RenderBucket::DepthPrepass)
    }

    /// Whether objects in this bucket should be sorted back-to-front.
    pub fn is_back_to_front(&self) -> bool {
        matches!(self, RenderBucket::Transparent | RenderBucket::Additive)
    }

    /// Whether depth writing is enabled for this bucket.
    pub fn writes_depth(&self) -> bool {
        matches!(self, RenderBucket::Opaque | RenderBucket::AlphaTest | RenderBucket::DepthPrepass | RenderBucket::ShadowCaster)
    }

    /// Whether blending is enabled for this bucket.
    pub fn uses_blending(&self) -> bool {
        matches!(self, RenderBucket::Transparent | RenderBucket::Additive | RenderBucket::UIOverlay)
    }

    pub fn all() -> &'static [RenderBucket] {
        &[
            RenderBucket::ShadowCaster,
            RenderBucket::DepthPrepass,
            RenderBucket::Opaque,
            RenderBucket::AlphaTest,
            RenderBucket::Transparent,
            RenderBucket::Additive,
            RenderBucket::Sky,
            RenderBucket::PostProcess,
            RenderBucket::DebugOverlay,
            RenderBucket::UIOverlay,
        ]
    }
}

// ---------------------------------------------------------------------------
// Draw command
// ---------------------------------------------------------------------------

/// Unique handle for a mesh resource.
pub type MeshHandle = u64;

/// Unique handle for a material resource.
pub type MaterialHandle = u64;

/// A single draw command in the render queue.
#[derive(Debug, Clone)]
pub struct DrawCommand {
    /// Sort key for ordering within the queue.
    pub sort_key: u64,
    /// Which bucket this draw belongs to.
    pub bucket: RenderBucket,
    /// Mesh to draw.
    pub mesh: MeshHandle,
    /// Material to use.
    pub material: MaterialHandle,
    /// Model transform (4x4 matrix, column-major).
    pub model_matrix: [f32; 16],
    /// Instance count (1 for non-instanced).
    pub instance_count: u32,
    /// First instance index (for instanced drawing).
    pub first_instance: u32,
    /// Index count (0 = use vertex count).
    pub index_count: u32,
    /// First index.
    pub first_index: u32,
    /// Vertex offset.
    pub vertex_offset: i32,
    /// Bounding sphere center (world space).
    pub bounds_center: [f32; 3],
    /// Bounding sphere radius.
    pub bounds_radius: f32,
    /// Distance from camera (computed during sorting).
    pub camera_distance: f32,
    /// Priority override (0 = default, higher = rendered first within bucket).
    pub priority: u8,
    /// Layer mask for visibility culling.
    pub layer_mask: u32,
    /// Entity ID (for debug/selection).
    pub entity_id: u64,
    /// Shadow cascade index (for shadow casters).
    pub shadow_cascade: u8,
    /// Material LOD level.
    pub material_lod: u8,
    /// Custom sort bias (added to camera distance).
    pub sort_bias: f32,
    /// Custom user data.
    pub user_data: u64,
}

impl DrawCommand {
    pub fn new(mesh: MeshHandle, material: MaterialHandle, bucket: RenderBucket) -> Self {
        Self {
            sort_key: 0,
            bucket,
            mesh,
            material,
            model_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            instance_count: 1,
            first_instance: 0,
            index_count: 0,
            first_index: 0,
            vertex_offset: 0,
            bounds_center: [0.0; 3],
            bounds_radius: 1.0,
            camera_distance: 0.0,
            priority: 0,
            layer_mask: u32::MAX,
            entity_id: 0,
            shadow_cascade: 0,
            material_lod: 0,
            sort_bias: 0.0,
            user_data: 0,
        }
    }

    /// Set the model matrix from position, scale (uniform).
    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.model_matrix[12] = x;
        self.model_matrix[13] = y;
        self.model_matrix[14] = z;
        self.bounds_center = [x, y, z];
        self
    }

    /// Get the world position from the model matrix.
    pub fn world_position(&self) -> [f32; 3] {
        [self.model_matrix[12], self.model_matrix[13], self.model_matrix[14]]
    }

    /// Compute the distance from a camera position.
    pub fn compute_distance(&mut self, camera_pos: [f32; 3]) {
        let dx = self.bounds_center[0] - camera_pos[0];
        let dy = self.bounds_center[1] - camera_pos[1];
        let dz = self.bounds_center[2] - camera_pos[2];
        self.camera_distance = (dx * dx + dy * dy + dz * dz).sqrt() + self.sort_bias;
    }

    /// Build the sort key based on current state.
    pub fn compute_sort_key(&mut self) {
        let depth_norm = match self.bucket {
            b if b.is_front_to_back() => {
                // Front-to-back: small distance = low key = sorted first.
                (self.camera_distance / 1000.0).clamp(0.0, 1.0)
            }
            b if b.is_back_to_front() => {
                // Back-to-front: large distance = low key = sorted first.
                1.0 - (self.camera_distance / 1000.0).clamp(0.0, 1.0)
            }
            _ => 0.0,
        };
        self.sort_key = build_sort_key(
            self.bucket,
            self.priority,
            depth_norm,
            (self.material & 0xFFFF) as u16,
            (self.mesh & 0xFFFF) as u16,
            self.material_lod,
        );
    }
}

// ---------------------------------------------------------------------------
// Render queue
// ---------------------------------------------------------------------------

/// Configuration for the render queue.
#[derive(Debug, Clone)]
pub struct RenderQueueConfig {
    /// Initial capacity for the draw command list.
    pub initial_capacity: usize,
    /// Whether to enable material-based batching within opaque bucket.
    pub material_batching: bool,
    /// Whether to enable per-cascade sorting for shadow casters.
    pub cascade_sorting: bool,
    /// Maximum number of shadow cascades.
    pub max_shadow_cascades: u8,
    /// Whether to track per-bucket statistics.
    pub track_stats: bool,
}

impl Default for RenderQueueConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 4096,
            material_batching: true,
            cascade_sorting: true,
            max_shadow_cascades: 4,
            track_stats: true,
        }
    }
}

/// Statistics for the render queue.
#[derive(Debug, Clone, Default)]
pub struct RenderQueueStats {
    /// Total draw commands submitted.
    pub total_commands: u32,
    /// Commands per bucket.
    pub commands_per_bucket: HashMap<u8, u32>,
    /// Number of unique materials used.
    pub unique_materials: u32,
    /// Number of unique meshes used.
    pub unique_meshes: u32,
    /// Number of material state changes (after sorting).
    pub material_state_changes: u32,
    /// Number of mesh state changes (after sorting).
    pub mesh_state_changes: u32,
    /// Total instance count.
    pub total_instances: u32,
    /// Time spent sorting in microseconds.
    pub sort_time_us: u64,
    /// Total triangles (estimated from index counts).
    pub estimated_triangles: u64,
}

/// The main render queue.
///
/// Collects draw commands, sorts them optimally, and provides iteration in
/// render order.
pub struct RenderQueueV2 {
    config: RenderQueueConfig,
    /// All draw commands for the current frame.
    commands: Vec<DrawCommand>,
    /// Whether the queue has been sorted this frame.
    sorted: bool,
    /// Per-frame statistics.
    pub stats: RenderQueueStats,
    /// Bucket start/end indices after sorting.
    bucket_ranges: HashMap<u8, (usize, usize)>,
}

impl RenderQueueV2 {
    pub fn new(config: RenderQueueConfig) -> Self {
        let initial_cap = config.initial_capacity;
        Self {
            config,
            commands: Vec::with_capacity(initial_cap),
            sorted: false,
            stats: RenderQueueStats::default(),
            bucket_ranges: HashMap::new(),
        }
    }

    /// Clear the queue for a new frame.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.sorted = false;
        self.stats = RenderQueueStats::default();
        self.bucket_ranges.clear();
    }

    /// Submit a draw command to the queue.
    pub fn submit(&mut self, cmd: DrawCommand) {
        self.sorted = false;
        self.commands.push(cmd);
    }

    /// Submit multiple draw commands.
    pub fn submit_batch(&mut self, cmds: impl IntoIterator<Item = DrawCommand>) {
        self.sorted = false;
        self.commands.extend(cmds);
    }

    /// Get the number of commands in the queue.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Sort all commands and compute bucket ranges.
    pub fn sort(&mut self, camera_pos: [f32; 3]) {
        let start = std::time::Instant::now();

        // Compute distances and sort keys.
        for cmd in &mut self.commands {
            cmd.compute_distance(camera_pos);
            cmd.compute_sort_key();
        }

        // Sort by sort key.
        self.commands.sort_by(|a, b| a.sort_key.cmp(&b.sort_key));

        // Compute bucket ranges.
        self.bucket_ranges.clear();
        if !self.commands.is_empty() {
            let mut current_bucket = sort_key_bucket(self.commands[0].sort_key);
            let mut start_idx = 0usize;
            for i in 1..self.commands.len() {
                let bucket = sort_key_bucket(self.commands[i].sort_key);
                if bucket != current_bucket {
                    self.bucket_ranges.insert(current_bucket, (start_idx, i));
                    current_bucket = bucket;
                    start_idx = i;
                }
            }
            self.bucket_ranges.insert(current_bucket, (start_idx, self.commands.len()));
        }

        // Compute stats.
        if self.config.track_stats {
            self.compute_stats();
        }

        self.stats.sort_time_us = start.elapsed().as_micros() as u64;
        self.sorted = true;
    }

    /// Get all commands for a specific bucket (sorted).
    pub fn bucket_commands(&self, bucket: RenderBucket) -> &[DrawCommand] {
        if let Some(&(start, end)) = self.bucket_ranges.get(&(bucket as u8)) {
            &self.commands[start..end]
        } else {
            &[]
        }
    }

    /// Iterate over all commands in render order.
    pub fn iter(&self) -> impl Iterator<Item = &DrawCommand> {
        self.commands.iter()
    }

    /// Iterate over commands in a specific bucket.
    pub fn iter_bucket(&self, bucket: RenderBucket) -> impl Iterator<Item = &DrawCommand> {
        self.bucket_commands(bucket).iter()
    }

    /// Get the number of commands in a specific bucket.
    pub fn bucket_count(&self, bucket: RenderBucket) -> usize {
        self.bucket_commands(bucket).len()
    }

    /// Iterate over shadow caster commands grouped by cascade.
    pub fn shadow_cascades(&self) -> Vec<Vec<&DrawCommand>> {
        let shadow_cmds = self.bucket_commands(RenderBucket::ShadowCaster);
        let max_cascade = self.config.max_shadow_cascades as usize;
        let mut cascades: Vec<Vec<&DrawCommand>> = vec![Vec::new(); max_cascade];
        for cmd in shadow_cmds {
            let cascade = (cmd.shadow_cascade as usize).min(max_cascade - 1);
            cascades[cascade].push(cmd);
        }
        cascades
    }

    /// Filter commands by visibility layer mask.
    pub fn filter_by_layer(&self, layer_mask: u32) -> Vec<&DrawCommand> {
        self.commands.iter().filter(|c| c.layer_mask & layer_mask != 0).collect()
    }

    /// Rebuild the queue: clear and submit new commands.
    pub fn rebuild(&mut self, commands: Vec<DrawCommand>, camera_pos: [f32; 3]) {
        self.clear();
        self.commands = commands;
        self.sort(camera_pos);
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn compute_stats(&mut self) {
        let mut unique_materials = std::collections::HashSet::new();
        let mut unique_meshes = std::collections::HashSet::new();
        let mut total_instances = 0u32;
        let mut estimated_tris = 0u64;

        for cmd in &self.commands {
            unique_materials.insert(cmd.material);
            unique_meshes.insert(cmd.mesh);
            total_instances += cmd.instance_count;
            estimated_tris += (cmd.index_count as u64 / 3) * cmd.instance_count as u64;

            let bucket_key = cmd.bucket as u8;
            *self.stats.commands_per_bucket.entry(bucket_key).or_insert(0) += 1;
        }

        // Count material state changes.
        let mut material_changes = 0u32;
        let mut mesh_changes = 0u32;
        if self.commands.len() > 1 {
            for i in 1..self.commands.len() {
                if self.commands[i].material != self.commands[i - 1].material {
                    material_changes += 1;
                }
                if self.commands[i].mesh != self.commands[i - 1].mesh {
                    mesh_changes += 1;
                }
            }
        }

        self.stats.total_commands = self.commands.len() as u32;
        self.stats.unique_materials = unique_materials.len() as u32;
        self.stats.unique_meshes = unique_meshes.len() as u32;
        self.stats.material_state_changes = material_changes;
        self.stats.mesh_state_changes = mesh_changes;
        self.stats.total_instances = total_instances;
        self.stats.estimated_triangles = estimated_tris;
    }
}

// ---------------------------------------------------------------------------
// Instancing helper
// ---------------------------------------------------------------------------

/// Batch builder that merges compatible draw commands into instanced draws.
pub struct InstanceBatcher {
    /// Minimum instance count to justify batching.
    pub min_instance_count: u32,
}

impl InstanceBatcher {
    pub fn new(min_instance_count: u32) -> Self {
        Self { min_instance_count }
    }

    /// Group commands by (mesh, material) and merge into instanced draw calls.
    pub fn batch(&self, commands: &[DrawCommand]) -> Vec<DrawCommand> {
        let mut groups: HashMap<(MeshHandle, MaterialHandle), Vec<usize>> = HashMap::new();
        for (i, cmd) in commands.iter().enumerate() {
            groups.entry((cmd.mesh, cmd.material)).or_default().push(i);
        }

        let mut result = Vec::with_capacity(commands.len());
        for ((mesh, material), indices) in &groups {
            if indices.len() as u32 >= self.min_instance_count && indices.len() > 1 {
                // Merge into a single instanced draw.
                let first = &commands[indices[0]];
                let mut batched = first.clone();
                batched.instance_count = indices.len() as u32;
                batched.first_instance = 0;
                result.push(batched);
            } else {
                // Keep individual draws.
                for &i in indices {
                    result.push(commands[i].clone());
                }
            }
        }
        let _ = (mesh, material); // Suppress unused warning in doc context.
        result
    }
}

// ---------------------------------------------------------------------------
// Material priority system
// ---------------------------------------------------------------------------

/// Per-material render priority override.
#[derive(Debug, Clone)]
pub struct MaterialPriority {
    /// Material handle.
    pub material: MaterialHandle,
    /// Priority override (higher = rendered first within bucket).
    pub priority: u8,
    /// Force a specific bucket (overrides default).
    pub force_bucket: Option<RenderBucket>,
    /// Custom sort bias.
    pub sort_bias: f32,
}

/// Registry of material priority overrides.
pub struct MaterialPriorityRegistry {
    priorities: HashMap<MaterialHandle, MaterialPriority>,
}

impl MaterialPriorityRegistry {
    pub fn new() -> Self {
        Self { priorities: HashMap::new() }
    }

    pub fn set(&mut self, priority: MaterialPriority) {
        self.priorities.insert(priority.material, priority);
    }

    pub fn get(&self, material: MaterialHandle) -> Option<&MaterialPriority> {
        self.priorities.get(&material)
    }

    pub fn remove(&mut self, material: MaterialHandle) -> bool {
        self.priorities.remove(&material).is_some()
    }

    /// Apply priorities to a draw command, modifying its bucket and priority.
    pub fn apply(&self, cmd: &mut DrawCommand) {
        if let Some(pri) = self.priorities.get(&cmd.material) {
            cmd.priority = pri.priority;
            cmd.sort_bias = pri.sort_bias;
            if let Some(bucket) = pri.force_bucket {
                cmd.bucket = bucket;
            }
        }
    }

    /// Apply priorities to all commands in a queue.
    pub fn apply_to_queue(&self, queue: &mut RenderQueueV2) {
        for cmd in &mut queue.commands {
            self.apply(cmd);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_key_construction() {
        let key = build_sort_key(RenderBucket::Opaque, 5, 0.5, 100, 200, 0);
        assert_eq!(sort_key_bucket(key), RenderBucket::Opaque as u8);
        let depth = sort_key_depth(key);
        assert!((depth - 0.5).abs() < 0.001);
        assert_eq!(sort_key_material(key), 100);
    }

    #[test]
    fn test_queue_sort_opaque_front_to_back() {
        let mut queue = RenderQueueV2::new(RenderQueueConfig::default());
        queue.submit(DrawCommand::new(1, 1, RenderBucket::Opaque).with_position(0.0, 0.0, 10.0));
        queue.submit(DrawCommand::new(2, 1, RenderBucket::Opaque).with_position(0.0, 0.0, 5.0));
        queue.submit(DrawCommand::new(3, 1, RenderBucket::Opaque).with_position(0.0, 0.0, 20.0));
        queue.sort([0.0, 0.0, 0.0]);

        let opaque = queue.bucket_commands(RenderBucket::Opaque);
        assert_eq!(opaque.len(), 3);
        // Front-to-back: closest first.
        assert!(opaque[0].camera_distance <= opaque[1].camera_distance);
        assert!(opaque[1].camera_distance <= opaque[2].camera_distance);
    }

    #[test]
    fn test_queue_sort_transparent_back_to_front() {
        let mut queue = RenderQueueV2::new(RenderQueueConfig::default());
        queue.submit(DrawCommand::new(1, 1, RenderBucket::Transparent).with_position(0.0, 0.0, 10.0));
        queue.submit(DrawCommand::new(2, 1, RenderBucket::Transparent).with_position(0.0, 0.0, 5.0));
        queue.submit(DrawCommand::new(3, 1, RenderBucket::Transparent).with_position(0.0, 0.0, 20.0));
        queue.sort([0.0, 0.0, 0.0]);

        let transparent = queue.bucket_commands(RenderBucket::Transparent);
        assert_eq!(transparent.len(), 3);
        // Back-to-front: farthest first.
        assert!(transparent[0].camera_distance >= transparent[1].camera_distance);
        assert!(transparent[1].camera_distance >= transparent[2].camera_distance);
    }

    #[test]
    fn test_bucket_ordering() {
        let mut queue = RenderQueueV2::new(RenderQueueConfig::default());
        queue.submit(DrawCommand::new(1, 1, RenderBucket::Transparent).with_position(0.0, 0.0, 5.0));
        queue.submit(DrawCommand::new(2, 1, RenderBucket::Opaque).with_position(0.0, 0.0, 5.0));
        queue.submit(DrawCommand::new(3, 1, RenderBucket::Sky).with_position(0.0, 0.0, 5.0));
        queue.submit(DrawCommand::new(4, 1, RenderBucket::ShadowCaster).with_position(0.0, 0.0, 5.0));
        queue.sort([0.0, 0.0, 0.0]);

        // After sorting, shadow casters should come before opaque, opaque before transparent, etc.
        let buckets: Vec<RenderBucket> = queue.iter().map(|c| c.bucket).collect();
        let bucket_order: Vec<u8> = buckets.iter().map(|b| *b as u8).collect();
        for i in 1..bucket_order.len() {
            assert!(bucket_order[i] >= bucket_order[i - 1]);
        }
    }

    #[test]
    fn test_instance_batcher() {
        let batcher = InstanceBatcher::new(2);
        let commands = vec![
            DrawCommand::new(1, 1, RenderBucket::Opaque),
            DrawCommand::new(1, 1, RenderBucket::Opaque),
            DrawCommand::new(1, 1, RenderBucket::Opaque),
            DrawCommand::new(2, 1, RenderBucket::Opaque),
        ];
        let batched = batcher.batch(&commands);
        // 3 identical commands should be batched into 1, plus 1 standalone.
        assert_eq!(batched.len(), 2);
        let instanced = batched.iter().find(|c| c.instance_count == 3);
        assert!(instanced.is_some());
    }

    #[test]
    fn test_queue_clear_and_rebuild() {
        let mut queue = RenderQueueV2::new(RenderQueueConfig::default());
        queue.submit(DrawCommand::new(1, 1, RenderBucket::Opaque));
        assert_eq!(queue.len(), 1);
        queue.clear();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_shadow_cascades() {
        let mut queue = RenderQueueV2::new(RenderQueueConfig::default());
        for cascade in 0..4u8 {
            let mut cmd = DrawCommand::new(1, 1, RenderBucket::ShadowCaster);
            cmd.shadow_cascade = cascade;
            queue.submit(cmd);
        }
        queue.sort([0.0, 0.0, 0.0]);
        let cascades = queue.shadow_cascades();
        assert_eq!(cascades.len(), 4);
        for (i, cascade) in cascades.iter().enumerate() {
            assert_eq!(cascade.len(), 1);
            assert_eq!(cascade[0].shadow_cascade, i as u8);
        }
    }
}
