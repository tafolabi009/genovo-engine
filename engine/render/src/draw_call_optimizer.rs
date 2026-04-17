// engine/render/src/draw_call_optimizer.rs
//
// Draw call optimization for the Genovo engine.
//
// Reduces GPU state changes and draw call overhead by:
//
// - **State sorting** -- Sorts draw commands by shader, then material, then mesh
//   to minimize pipeline/descriptor set changes.
// - **Compatible draw call merging** -- Merges draw calls that share the same
//   pipeline, material, and vertex layout into a single instanced draw.
// - **Indirect draw batching** -- Builds GPU indirect draw buffers for hardware
//   that supports multi-draw-indirect.
// - **Instance merging** -- Combines per-instance transforms into instance
//   buffers for instanced rendering.
// - **Draw call statistics** -- Tracks draw calls, state changes, and merging
//   effectiveness per frame.
// - **State change counting** -- Counts each category of state change to
//   identify optimization opportunities.

use std::fmt;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Shader program identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderId(pub u32);

/// Material identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialId(pub u32);

/// Mesh/vertex buffer identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshId(pub u32);

/// Texture binding identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureBindingId(pub u32);

// ---------------------------------------------------------------------------
// Draw command
// ---------------------------------------------------------------------------

/// A single draw command submitted by the renderer before optimization.
#[derive(Debug, Clone)]
pub struct DrawCommand {
    /// Shader/pipeline to use.
    pub shader: ShaderId,
    /// Material (descriptor set / uniform data).
    pub material: MaterialId,
    /// Mesh (vertex + index buffer binding).
    pub mesh: MeshId,
    /// Texture bindings for this draw call.
    pub texture_bindings: Vec<TextureBindingId>,
    /// Index offset in the index buffer.
    pub index_offset: u32,
    /// Number of indices to draw.
    pub index_count: u32,
    /// Vertex offset.
    pub vertex_offset: i32,
    /// World transform (column-major 4x4).
    pub transform: [f32; 16],
    /// Render layer/queue (0 = opaque, 1 = transparent, etc.).
    pub layer: u8,
    /// Sort depth for transparent objects (camera distance).
    pub sort_depth: f32,
    /// Whether this command is eligible for instanced merging.
    pub instancable: bool,
    /// Custom sort key (overrides automatic sorting if non-zero).
    pub custom_sort_key: u64,
}

impl DrawCommand {
    /// Compute the automatic sort key: layer | shader | material | mesh.
    pub fn compute_sort_key(&self) -> u64 {
        if self.custom_sort_key != 0 {
            return self.custom_sort_key;
        }
        let layer_bits = (self.layer as u64) << 56;
        let shader_bits = (self.shader.0 as u64 & 0xFFFF) << 40;
        let material_bits = (self.material.0 as u64 & 0xFFFF) << 24;
        let mesh_bits = (self.mesh.0 as u64 & 0xFFFFFF) << 0;
        layer_bits | shader_bits | material_bits | mesh_bits
    }

    /// Check if two draw commands can be merged (same pipeline state).
    pub fn can_merge_with(&self, other: &DrawCommand) -> bool {
        self.shader == other.shader
            && self.material == other.material
            && self.mesh == other.mesh
            && self.texture_bindings == other.texture_bindings
            && self.layer == other.layer
            && self.instancable
            && other.instancable
            && self.index_offset == other.index_offset
            && self.index_count == other.index_count
            && self.vertex_offset == other.vertex_offset
    }
}

// ---------------------------------------------------------------------------
// Optimized draw batch
// ---------------------------------------------------------------------------

/// A batch of draw commands that share the same pipeline state.
#[derive(Debug, Clone)]
pub struct DrawBatch {
    /// Shader/pipeline for this batch.
    pub shader: ShaderId,
    /// Material for this batch.
    pub material: MaterialId,
    /// Mesh for this batch.
    pub mesh: MeshId,
    /// Texture bindings for this batch.
    pub texture_bindings: Vec<TextureBindingId>,
    /// Render layer.
    pub layer: u8,
    /// Index offset in the index buffer.
    pub index_offset: u32,
    /// Number of indices per instance.
    pub index_count: u32,
    /// Vertex offset.
    pub vertex_offset: i32,
    /// Instance count.
    pub instance_count: u32,
    /// Per-instance transforms.
    pub instance_transforms: Vec<[f32; 16]>,
    /// Original draw command indices (for debugging).
    pub source_commands: Vec<usize>,
}

impl DrawBatch {
    /// Create a batch from a single draw command.
    pub fn from_command(cmd: &DrawCommand, source_index: usize) -> Self {
        Self {
            shader: cmd.shader,
            material: cmd.material,
            mesh: cmd.mesh,
            texture_bindings: cmd.texture_bindings.clone(),
            layer: cmd.layer,
            index_offset: cmd.index_offset,
            index_count: cmd.index_count,
            vertex_offset: cmd.vertex_offset,
            instance_count: 1,
            instance_transforms: vec![cmd.transform],
            source_commands: vec![source_index],
        }
    }

    /// Add another instance to this batch.
    pub fn add_instance(&mut self, cmd: &DrawCommand, source_index: usize) {
        self.instance_transforms.push(cmd.transform);
        self.source_commands.push(source_index);
        self.instance_count += 1;
    }

    /// Returns the total number of triangles in this batch.
    pub fn triangle_count(&self) -> u32 {
        (self.index_count / 3) * self.instance_count
    }
}

// ---------------------------------------------------------------------------
// Indirect draw command (for GPU indirect draw buffers)
// ---------------------------------------------------------------------------

/// GPU indirect draw command (matches Vulkan/DX12 DrawIndexedIndirect layout).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct IndirectDrawCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

impl IndirectDrawCommand {
    /// Create from a draw batch.
    pub fn from_batch(batch: &DrawBatch, first_instance: u32) -> Self {
        Self {
            index_count: batch.index_count,
            instance_count: batch.instance_count,
            first_index: batch.index_offset,
            vertex_offset: batch.vertex_offset,
            first_instance,
        }
    }
}

// ---------------------------------------------------------------------------
// State change types
// ---------------------------------------------------------------------------

/// Categories of GPU state changes tracked by the optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateChangeType {
    /// Pipeline / shader program change.
    Pipeline,
    /// Material / descriptor set change.
    Material,
    /// Mesh / vertex buffer binding change.
    Mesh,
    /// Texture binding change.
    Texture,
    /// Render target change.
    RenderTarget,
    /// Blend state change.
    BlendState,
    /// Depth/stencil state change.
    DepthStencilState,
}

// ---------------------------------------------------------------------------
// Draw call statistics
// ---------------------------------------------------------------------------

/// Per-frame draw call statistics.
#[derive(Debug, Clone, Default)]
pub struct DrawCallStats {
    /// Number of draw commands submitted before optimization.
    pub input_draw_calls: u32,
    /// Number of draw batches after optimization.
    pub output_batches: u32,
    /// Number of draw calls saved by merging.
    pub merged_draw_calls: u32,
    /// Merge ratio (0.0 = no merging, 1.0 = everything merged into one).
    pub merge_ratio: f32,
    /// Total triangles across all batches.
    pub total_triangles: u64,
    /// Total instances across all batches.
    pub total_instances: u32,
    /// Number of state changes by type.
    pub state_changes: HashMap<StateChangeType, u32>,
    /// Number of indirect draw commands generated.
    pub indirect_draw_count: u32,
    /// Maximum instances in a single batch.
    pub max_batch_instances: u32,
    /// Average instances per batch.
    pub avg_instances_per_batch: f32,
    /// Time spent optimizing (microseconds).
    pub optimization_time_us: u64,
}

impl DrawCallStats {
    /// Reset all counters.
    pub fn reset(&mut self) {
        self.input_draw_calls = 0;
        self.output_batches = 0;
        self.merged_draw_calls = 0;
        self.merge_ratio = 0.0;
        self.total_triangles = 0;
        self.total_instances = 0;
        self.state_changes.clear();
        self.indirect_draw_count = 0;
        self.max_batch_instances = 0;
        self.avg_instances_per_batch = 0.0;
        self.optimization_time_us = 0;
    }
}

impl fmt::Display for DrawCallStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DrawCalls: {} -> {} batches ({} merged, {:.1}%), {} tris, {} instances",
            self.input_draw_calls,
            self.output_batches,
            self.merged_draw_calls,
            self.merge_ratio * 100.0,
            self.total_triangles,
            self.total_instances,
        )
    }
}

// ---------------------------------------------------------------------------
// Optimizer configuration
// ---------------------------------------------------------------------------

/// Configuration for the draw call optimizer.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Enable draw call sorting.
    pub enable_sorting: bool,
    /// Enable compatible draw call merging.
    pub enable_merging: bool,
    /// Enable indirect draw buffer generation.
    pub enable_indirect: bool,
    /// Maximum instances per batch (0 = unlimited).
    pub max_instances_per_batch: u32,
    /// Sort transparent objects back-to-front.
    pub sort_transparent_back_to_front: bool,
    /// Layer index at which transparency sorting begins.
    pub transparent_layer_start: u8,
    /// Enable state change tracking.
    pub track_state_changes: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            enable_sorting: true,
            enable_merging: true,
            enable_indirect: false,
            max_instances_per_batch: 1024,
            sort_transparent_back_to_front: true,
            transparent_layer_start: 1,
            track_state_changes: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Draw call optimizer
// ---------------------------------------------------------------------------

/// Optimizes draw commands by sorting, merging, and batching.
pub struct DrawCallOptimizer {
    /// Configuration.
    config: OptimizerConfig,
    /// Input draw commands for the current frame.
    commands: Vec<DrawCommand>,
    /// Output optimized batches.
    batches: Vec<DrawBatch>,
    /// Indirect draw commands (if enabled).
    indirect_commands: Vec<IndirectDrawCommand>,
    /// Per-frame statistics.
    stats: DrawCallStats,
    /// Sort key buffer (reused to avoid allocation).
    sort_keys: Vec<(u64, usize)>,
}

impl DrawCallOptimizer {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            commands: Vec::with_capacity(4096),
            batches: Vec::with_capacity(1024),
            indirect_commands: Vec::with_capacity(1024),
            stats: DrawCallStats::default(),
            sort_keys: Vec::with_capacity(4096),
        }
    }

    /// Create a new optimizer with default configuration.
    pub fn default_config() -> Self {
        Self::new(OptimizerConfig::default())
    }

    /// Get the current configuration.
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: OptimizerConfig) {
        self.config = config;
    }

    /// Clear all commands and batches for a new frame.
    pub fn begin_frame(&mut self) {
        self.commands.clear();
        self.batches.clear();
        self.indirect_commands.clear();
        self.stats.reset();
        self.sort_keys.clear();
    }

    /// Submit a draw command for optimization.
    pub fn submit(&mut self, cmd: DrawCommand) {
        self.commands.push(cmd);
    }

    /// Submit multiple draw commands.
    pub fn submit_batch(&mut self, commands: impl IntoIterator<Item = DrawCommand>) {
        self.commands.extend(commands);
    }

    /// Process all submitted commands and produce optimized batches.
    pub fn optimize(&mut self) {
        let start = std::time::Instant::now();

        self.stats.input_draw_calls = self.commands.len() as u32;

        if self.commands.is_empty() {
            return;
        }

        // Step 1: Sort commands.
        if self.config.enable_sorting {
            self.sort_commands();
        }

        // Step 2: Merge compatible commands into batches.
        if self.config.enable_merging {
            self.merge_commands();
        } else {
            // No merging -- each command becomes its own batch.
            for (i, cmd) in self.commands.iter().enumerate() {
                self.batches.push(DrawBatch::from_command(cmd, i));
            }
        }

        // Step 3: Generate indirect draw commands.
        if self.config.enable_indirect {
            self.generate_indirect();
        }

        // Step 4: Count state changes.
        if self.config.track_state_changes {
            self.count_state_changes();
        }

        // Compute statistics.
        self.stats.output_batches = self.batches.len() as u32;
        self.stats.merged_draw_calls =
            self.stats.input_draw_calls.saturating_sub(self.stats.output_batches);
        self.stats.merge_ratio = if self.stats.input_draw_calls > 0 {
            self.stats.merged_draw_calls as f32 / self.stats.input_draw_calls as f32
        } else {
            0.0
        };
        self.stats.total_triangles = self
            .batches
            .iter()
            .map(|b| b.triangle_count() as u64)
            .sum();
        self.stats.total_instances = self
            .batches
            .iter()
            .map(|b| b.instance_count)
            .sum();
        self.stats.max_batch_instances = self
            .batches
            .iter()
            .map(|b| b.instance_count)
            .max()
            .unwrap_or(0);
        self.stats.avg_instances_per_batch = if !self.batches.is_empty() {
            self.stats.total_instances as f32 / self.batches.len() as f32
        } else {
            0.0
        };

        self.stats.optimization_time_us = start.elapsed().as_micros() as u64;
    }

    /// Sort commands by sort key.
    fn sort_commands(&mut self) {
        self.sort_keys.clear();
        for (i, cmd) in self.commands.iter().enumerate() {
            let key = cmd.compute_sort_key();
            self.sort_keys.push((key, i));
        }

        // Sort transparent layers back-to-front by depth.
        self.sort_keys.sort_by(|a, b| {
            let cmd_a = &self.commands[a.1];
            let cmd_b = &self.commands[b.1];

            // Transparent objects get sorted differently.
            if self.config.sort_transparent_back_to_front
                && cmd_a.layer >= self.config.transparent_layer_start
                && cmd_b.layer >= self.config.transparent_layer_start
            {
                // Back-to-front: larger depth first.
                return cmd_b
                    .sort_depth
                    .partial_cmp(&cmd_a.sort_depth)
                    .unwrap_or(std::cmp::Ordering::Equal);
            }

            a.0.cmp(&b.0)
        });

        // Reorder commands array according to sorted keys.
        let sorted_indices: Vec<usize> = self.sort_keys.iter().map(|&(_, i)| i).collect();
        let mut sorted_commands = Vec::with_capacity(self.commands.len());
        for &idx in &sorted_indices {
            sorted_commands.push(self.commands[idx].clone());
        }
        self.commands = sorted_commands;
    }

    /// Merge compatible consecutive commands into batches.
    fn merge_commands(&mut self) {
        self.batches.clear();

        if self.commands.is_empty() {
            return;
        }

        let mut current_batch = DrawBatch::from_command(&self.commands[0], 0);

        for i in 1..self.commands.len() {
            let cmd = &self.commands[i];
            let max_inst = self.config.max_instances_per_batch;
            let can_merge = self.commands[i - 1].can_merge_with(cmd)
                && (max_inst == 0 || current_batch.instance_count < max_inst);

            if can_merge {
                current_batch.add_instance(cmd, i);
            } else {
                self.batches.push(current_batch);
                current_batch = DrawBatch::from_command(cmd, i);
            }
        }
        self.batches.push(current_batch);
    }

    /// Generate GPU indirect draw commands from batches.
    fn generate_indirect(&mut self) {
        self.indirect_commands.clear();
        let mut first_instance: u32 = 0;

        for batch in &self.batches {
            self.indirect_commands.push(IndirectDrawCommand::from_batch(batch, first_instance));
            first_instance += batch.instance_count;
        }
        self.stats.indirect_draw_count = self.indirect_commands.len() as u32;
    }

    /// Count state changes between consecutive batches.
    fn count_state_changes(&mut self) {
        self.stats.state_changes.clear();

        if self.batches.len() < 2 {
            return;
        }

        for i in 1..self.batches.len() {
            let prev = &self.batches[i - 1];
            let curr = &self.batches[i];

            if prev.shader != curr.shader {
                *self
                    .stats
                    .state_changes
                    .entry(StateChangeType::Pipeline)
                    .or_insert(0) += 1;
            }
            if prev.material != curr.material {
                *self
                    .stats
                    .state_changes
                    .entry(StateChangeType::Material)
                    .or_insert(0) += 1;
            }
            if prev.mesh != curr.mesh {
                *self
                    .stats
                    .state_changes
                    .entry(StateChangeType::Mesh)
                    .or_insert(0) += 1;
            }
            if prev.texture_bindings != curr.texture_bindings {
                *self
                    .stats
                    .state_changes
                    .entry(StateChangeType::Texture)
                    .or_insert(0) += 1;
            }
        }
    }

    /// Get the optimized draw batches.
    pub fn batches(&self) -> &[DrawBatch] {
        &self.batches
    }

    /// Get the indirect draw commands (only populated if indirect is enabled).
    pub fn indirect_commands(&self) -> &[IndirectDrawCommand] {
        &self.indirect_commands
    }

    /// Get the per-frame statistics.
    pub fn stats(&self) -> &DrawCallStats {
        &self.stats
    }

    /// Get the total number of state changes across all categories.
    pub fn total_state_changes(&self) -> u32 {
        self.stats.state_changes.values().sum()
    }

    /// Get the number of submitted commands (before optimization).
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Get instance transforms for all batches concatenated (for GPU upload).
    pub fn all_instance_transforms(&self) -> Vec<[f32; 16]> {
        let total: usize = self.batches.iter().map(|b| b.instance_transforms.len()).sum();
        let mut result = Vec::with_capacity(total);
        for batch in &self.batches {
            result.extend_from_slice(&batch.instance_transforms);
        }
        result
    }
}

impl Default for DrawCallOptimizer {
    fn default() -> Self {
        Self::default_config()
    }
}

impl fmt::Debug for DrawCallOptimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DrawCallOptimizer")
            .field("commands", &self.commands.len())
            .field("batches", &self.batches.len())
            .field("stats", &self.stats)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_matrix() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn make_cmd(shader: u32, material: u32, mesh: u32) -> DrawCommand {
        DrawCommand {
            shader: ShaderId(shader),
            material: MaterialId(material),
            mesh: MeshId(mesh),
            texture_bindings: vec![],
            index_offset: 0,
            index_count: 36,
            vertex_offset: 0,
            transform: identity_matrix(),
            layer: 0,
            sort_depth: 0.0,
            instancable: true,
            custom_sort_key: 0,
        }
    }

    #[test]
    fn test_sorting() {
        let mut opt = DrawCallOptimizer::default();
        opt.begin_frame();
        opt.submit(make_cmd(2, 1, 1));
        opt.submit(make_cmd(1, 1, 1));
        opt.submit(make_cmd(1, 2, 1));
        opt.optimize();

        // Should be sorted: shader 1 before shader 2.
        assert_eq!(opt.batches()[0].shader, ShaderId(1));
    }

    #[test]
    fn test_merging() {
        let mut opt = DrawCallOptimizer::default();
        opt.begin_frame();
        opt.submit(make_cmd(1, 1, 1));
        opt.submit(make_cmd(1, 1, 1));
        opt.submit(make_cmd(1, 1, 1));
        opt.optimize();

        assert_eq!(opt.batches().len(), 1);
        assert_eq!(opt.batches()[0].instance_count, 3);
        assert_eq!(opt.stats().merged_draw_calls, 2);
    }

    #[test]
    fn test_no_merge_different_material() {
        let mut opt = DrawCallOptimizer::default();
        opt.begin_frame();
        opt.submit(make_cmd(1, 1, 1));
        opt.submit(make_cmd(1, 2, 1));
        opt.optimize();

        assert_eq!(opt.batches().len(), 2);
    }

    #[test]
    fn test_indirect_generation() {
        let mut opt = DrawCallOptimizer::new(OptimizerConfig {
            enable_indirect: true,
            ..Default::default()
        });
        opt.begin_frame();
        opt.submit(make_cmd(1, 1, 1));
        opt.submit(make_cmd(1, 1, 1));
        opt.submit(make_cmd(2, 1, 1));
        opt.optimize();

        assert!(!opt.indirect_commands().is_empty());
        assert_eq!(opt.stats().indirect_draw_count, 2);
    }
}
