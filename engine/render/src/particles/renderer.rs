// engine/render/src/particles/renderer.rs
//
// Particle rendering: billboard computation, stretched billboards, vertex
// generation for particle quads, instanced rendering data preparation,
// soft particles, and sprite sheet animation.

use glam::{Vec2, Vec3};
use super::particle::ParticlePool;

// ---------------------------------------------------------------------------
// ParticleRenderMode
// ---------------------------------------------------------------------------

/// Determines how particles are rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParticleRenderMode {
    /// Camera-facing quads (standard billboards).
    Billboard,
    /// Billboards stretched along the velocity vector.
    StretchedBillboard,
    /// Each particle is rendered as a 3D mesh instance.
    Mesh,
    /// Particles leave a ribbon trail behind them.
    Trail,
    /// Horizontal billboards (always face up, used for ground effects).
    HorizontalBillboard,
    /// Vertical billboards (always face the camera around the Y axis).
    VerticalBillboard,
}

impl Default for ParticleRenderMode {
    fn default() -> Self {
        ParticleRenderMode::Billboard
    }
}

// ---------------------------------------------------------------------------
// SpriteSheet
// ---------------------------------------------------------------------------

/// Configuration for animated sprite sheets (flipbook animation).
#[derive(Debug, Clone)]
pub struct SpriteSheet {
    /// Number of columns in the sprite sheet.
    pub columns: u32,
    /// Number of rows in the sprite sheet.
    pub rows: u32,
    /// Total number of frames (may be less than rows * columns).
    pub frame_count: u32,
    /// Frames per second.
    pub frame_rate: f32,
    /// How the animation loops.
    pub loop_mode: SpriteLoopMode,
    /// If `true`, interpolate between frames for smoother animation.
    pub interpolate: bool,
}

/// Controls how sprite sheet animation loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpriteLoopMode {
    /// Loop the animation forever.
    Loop,
    /// Play once and hold the last frame.
    Once,
    /// Play once over the particle's lifetime (speed adjusts automatically).
    OverLifetime,
    /// Ping-pong (play forward then backward).
    PingPong,
}

impl SpriteSheet {
    /// Creates a new sprite sheet configuration.
    pub fn new(columns: u32, rows: u32) -> Self {
        let frame_count = columns * rows;
        Self {
            columns,
            rows,
            frame_count,
            frame_rate: 30.0,
            loop_mode: SpriteLoopMode::OverLifetime,
            interpolate: true,
        }
    }

    /// Sets the frame rate.
    pub fn with_frame_rate(mut self, fps: f32) -> Self {
        self.frame_rate = fps;
        self
    }

    /// Sets the loop mode.
    pub fn with_loop_mode(mut self, mode: SpriteLoopMode) -> Self {
        self.loop_mode = mode;
        self
    }

    /// Computes the UV coordinates for a given frame index.
    ///
    /// Returns `(uv_min, uv_max)` for the frame's rectangle in the atlas.
    pub fn frame_uvs(&self, frame: u32) -> (Vec2, Vec2) {
        let frame = frame.min(self.frame_count.saturating_sub(1));
        let col = frame % self.columns;
        let row = frame / self.columns;

        let u_size = 1.0 / self.columns as f32;
        let v_size = 1.0 / self.rows as f32;

        let u_min = col as f32 * u_size;
        let v_min = row as f32 * v_size;

        (
            Vec2::new(u_min, v_min),
            Vec2::new(u_min + u_size, v_min + v_size),
        )
    }

    /// Evaluates which frame(s) to display for a particle based on its age.
    ///
    /// Returns `(frame_a, frame_b, blend)` where `blend` is the interpolation
    /// factor between frames A and B.
    pub fn evaluate(
        &self,
        particle_age: f32,
        particle_lifetime: f32,
    ) -> (u32, u32, f32) {
        if self.frame_count == 0 {
            return (0, 0, 0.0);
        }

        let progress = match self.loop_mode {
            SpriteLoopMode::OverLifetime => {
                if particle_lifetime > 0.0 {
                    (particle_age / particle_lifetime).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
            SpriteLoopMode::Loop => {
                let cycle_time = self.frame_count as f32 / self.frame_rate;
                if cycle_time > 0.0 {
                    (particle_age % cycle_time) / cycle_time
                } else {
                    0.0
                }
            }
            SpriteLoopMode::Once => {
                let cycle_time = self.frame_count as f32 / self.frame_rate;
                if cycle_time > 0.0 {
                    (particle_age / cycle_time).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
            SpriteLoopMode::PingPong => {
                let cycle_time = self.frame_count as f32 / self.frame_rate;
                if cycle_time > 0.0 {
                    let t = (particle_age % (cycle_time * 2.0)) / cycle_time;
                    if t <= 1.0 { t } else { 2.0 - t }
                } else {
                    0.0
                }
            }
        };

        let float_frame = progress * (self.frame_count - 1) as f32;
        let frame_a = float_frame.floor() as u32;
        let frame_b = (frame_a + 1).min(self.frame_count - 1);
        let blend = if self.interpolate {
            float_frame.fract()
        } else {
            0.0
        };

        (frame_a, frame_b, blend)
    }
}

impl Default for SpriteSheet {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

// ---------------------------------------------------------------------------
// SoftParticleSettings
// ---------------------------------------------------------------------------

/// Configuration for soft particles (depth-fade at geometry intersections).
#[derive(Debug, Clone, Copy)]
pub struct SoftParticleSettings {
    /// If `true`, soft particles are enabled.
    pub enabled: bool,
    /// Distance over which particles fade near depth buffer geometry.
    /// Larger values = softer fade.
    pub fade_distance: f32,
    /// Power curve for the fade (1.0 = linear, 2.0 = quadratic).
    pub fade_power: f32,
}

impl SoftParticleSettings {
    pub fn new(fade_distance: f32) -> Self {
        Self {
            enabled: true,
            fade_distance,
            fade_power: 1.0,
        }
    }

    /// Computes the soft particle alpha multiplier given the difference
    /// between the scene depth and the particle depth.
    ///
    /// `depth_diff` = scene_depth - particle_depth (positive when particle
    /// is in front of geometry).
    pub fn compute_fade(&self, depth_diff: f32) -> f32 {
        if !self.enabled || self.fade_distance <= 0.0 {
            return 1.0;
        }
        let t = (depth_diff / self.fade_distance).clamp(0.0, 1.0);
        t.powf(self.fade_power)
    }
}

impl Default for SoftParticleSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            fade_distance: 0.5,
            fade_power: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ParticleVertex
// ---------------------------------------------------------------------------

/// A single vertex in a particle quad. Matches the GPU vertex layout.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ParticleVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// Texture coordinates.
    pub uv: [f32; 2],
    /// RGBA color (premultiplied alpha).
    pub color: [f32; 4],
    /// Sprite sheet blend (frame_b UVs and blend factor).
    /// [u_b, v_b, blend, _pad]
    pub sprite_blend: [f32; 4],
}

impl ParticleVertex {
    pub fn new(position: Vec3, uv: Vec2, color: [f32; 4]) -> Self {
        Self {
            position: [position.x, position.y, position.z],
            uv: [uv.x, uv.y],
            color,
            sprite_blend: [0.0; 4],
        }
    }
}

// ---------------------------------------------------------------------------
// ParticleInstanceData
// ---------------------------------------------------------------------------

/// Per-particle instance data for instanced rendering.
///
/// When using instanced rendering, each particle is a single instance
/// of a shared quad mesh. This struct contains the per-instance data.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ParticleInstanceData {
    /// World-space position.
    pub position: [f32; 3],
    /// Size (uniform scale).
    pub size: f32,
    /// RGBA color.
    pub color: [f32; 4],
    /// Rotation angle (radians, around camera forward axis).
    pub rotation: f32,
    /// Sprite sheet frame (float for interpolation).
    pub sprite_frame: f32,
    /// Sprite blend factor.
    pub sprite_blend: f32,
    /// Velocity magnitude (for stretched billboards).
    pub velocity_length: f32,
    /// Velocity direction (for stretched billboards).
    pub velocity_dir: [f32; 3],
    /// Padding to align to 16 bytes.
    pub _pad: f32,
}

// ---------------------------------------------------------------------------
// ParticleRenderer
// ---------------------------------------------------------------------------

/// Generates renderable data (vertices or instances) from a particle pool.
pub struct ParticleRenderer {
    /// How particles are rendered.
    pub render_mode: ParticleRenderMode,
    /// Optional sprite sheet animation.
    pub sprite_sheet: Option<SpriteSheet>,
    /// Soft particle settings.
    pub soft_particles: SoftParticleSettings,
    /// Stretch factor for stretched billboards (multiplied by velocity).
    pub stretch_factor: f32,
    /// Minimum stretch length (prevents disappearance at low velocity).
    pub min_stretch: f32,
    /// Maximum stretch length.
    pub max_stretch: f32,
    /// Vertex buffer (reused each frame to avoid allocation).
    vertex_buffer: Vec<ParticleVertex>,
    /// Index buffer (reused each frame).
    index_buffer: Vec<u32>,
    /// Instance data buffer (for instanced rendering).
    instance_buffer: Vec<ParticleInstanceData>,
}

impl ParticleRenderer {
    pub fn new(render_mode: ParticleRenderMode) -> Self {
        Self {
            render_mode,
            sprite_sheet: None,
            soft_particles: SoftParticleSettings::default(),
            stretch_factor: 1.0,
            min_stretch: 0.1,
            max_stretch: 10.0,
            vertex_buffer: Vec::new(),
            index_buffer: Vec::new(),
            instance_buffer: Vec::new(),
        }
    }

    pub fn with_sprite_sheet(mut self, sheet: SpriteSheet) -> Self {
        self.sprite_sheet = Some(sheet);
        self
    }

    pub fn with_soft_particles(mut self, settings: SoftParticleSettings) -> Self {
        self.soft_particles = settings;
        self
    }

    pub fn with_stretch(mut self, factor: f32, min: f32, max: f32) -> Self {
        self.stretch_factor = factor;
        self.min_stretch = min;
        self.max_stretch = max;
        self
    }

    /// Generates billboard quad vertices for all alive particles.
    ///
    /// Each particle produces 4 vertices and 6 indices (two triangles).
    ///
    /// # Arguments
    /// * `pool`        - The particle pool.
    /// * `camera_pos`  - World-space camera position.
    /// * `camera_right`- Camera right direction (unit vector).
    /// * `camera_up`   - Camera up direction (unit vector).
    /// * `camera_fwd`  - Camera forward direction (unit vector).
    pub fn generate_billboards(
        &mut self,
        pool: &ParticlePool,
        camera_pos: Vec3,
        camera_right: Vec3,
        camera_up: Vec3,
        camera_fwd: Vec3,
    ) {
        let count = pool.alive();
        self.vertex_buffer.clear();
        self.index_buffer.clear();

        if count == 0 {
            return;
        }

        self.vertex_buffer.reserve(count * 4);
        self.index_buffer.reserve(count * 6);

        for i in 0..count {
            let pos = pool.positions[i];
            let size = pool.sizes[i];
            let color = pool.colors[i];
            let rotation = pool.rotations[i];

            // Compute sprite sheet UVs.
            let (uv_min, uv_max, sprite_data) = if let Some(sheet) = &self.sprite_sheet {
                let (fa, fb, blend) = sheet.evaluate(
                    pool.lifetimes[i],
                    pool.max_lifetimes[i],
                );
                let (uv_a_min, uv_a_max) = sheet.frame_uvs(fa);
                let (uv_b_min, _uv_b_max) = sheet.frame_uvs(fb);
                (uv_a_min, uv_a_max, [uv_b_min.x, uv_b_min.y, blend, 0.0])
            } else {
                (Vec2::ZERO, Vec2::ONE, [0.0; 4])
            };

            match self.render_mode {
                ParticleRenderMode::Billboard => {
                    self.emit_billboard_quad(
                        pos, size, rotation, color,
                        camera_right, camera_up,
                        uv_min, uv_max, sprite_data,
                    );
                }
                ParticleRenderMode::StretchedBillboard => {
                    let vel = pool.velocities[i];
                    self.emit_stretched_billboard(
                        pos, size, vel, color,
                        camera_pos, camera_right, camera_up, camera_fwd,
                        uv_min, uv_max, sprite_data,
                    );
                }
                ParticleRenderMode::HorizontalBillboard => {
                    self.emit_horizontal_billboard(
                        pos, size, rotation, color,
                        uv_min, uv_max, sprite_data,
                    );
                }
                ParticleRenderMode::VerticalBillboard => {
                    self.emit_vertical_billboard(
                        pos, size, rotation, color,
                        camera_pos,
                        uv_min, uv_max, sprite_data,
                    );
                }
                _ => {
                    // Mesh and Trail modes don't use billboard quads.
                    self.emit_billboard_quad(
                        pos, size, rotation, color,
                        camera_right, camera_up,
                        uv_min, uv_max, sprite_data,
                    );
                }
            }
        }
    }

    /// Emits a standard camera-facing billboard quad.
    fn emit_billboard_quad(
        &mut self,
        center: Vec3,
        size: f32,
        rotation: f32,
        color: [f32; 4],
        camera_right: Vec3,
        camera_up: Vec3,
        uv_min: Vec2,
        uv_max: Vec2,
        sprite_blend: [f32; 4],
    ) {
        let half = size * 0.5;

        // Apply rotation around the billboard normal (camera forward).
        let (sin_r, cos_r) = rotation.sin_cos();
        let right = (camera_right * cos_r + camera_up * sin_r) * half;
        let up = (-camera_right * sin_r + camera_up * cos_r) * half;

        let base_idx = self.vertex_buffer.len() as u32;

        // Quad corners: bottom-left, bottom-right, top-right, top-left.
        let p0 = center - right - up;
        let p1 = center + right - up;
        let p2 = center + right + up;
        let p3 = center - right + up;

        self.vertex_buffer.push(ParticleVertex {
            position: [p0.x, p0.y, p0.z],
            uv: [uv_min.x, uv_max.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p1.x, p1.y, p1.z],
            uv: [uv_max.x, uv_max.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p2.x, p2.y, p2.z],
            uv: [uv_max.x, uv_min.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p3.x, p3.y, p3.z],
            uv: [uv_min.x, uv_min.y],
            color,
            sprite_blend,
        });

        // Two triangles: 0-1-2, 0-2-3.
        self.index_buffer.push(base_idx);
        self.index_buffer.push(base_idx + 1);
        self.index_buffer.push(base_idx + 2);
        self.index_buffer.push(base_idx);
        self.index_buffer.push(base_idx + 2);
        self.index_buffer.push(base_idx + 3);
    }

    /// Emits a stretched billboard oriented along the velocity vector.
    fn emit_stretched_billboard(
        &mut self,
        center: Vec3,
        size: f32,
        velocity: Vec3,
        color: [f32; 4],
        camera_pos: Vec3,
        _camera_right: Vec3,
        _camera_up: Vec3,
        camera_fwd: Vec3,
        uv_min: Vec2,
        uv_max: Vec2,
        sprite_blend: [f32; 4],
    ) {
        let speed = velocity.length();
        let half_width = size * 0.5;

        // Stretch length based on velocity.
        let stretch = (speed * self.stretch_factor)
            .clamp(self.min_stretch, self.max_stretch);
        let half_length = stretch * 0.5;

        // Forward direction = velocity direction (or camera forward if stationary).
        let forward = if speed > 1e-6 {
            velocity / speed
        } else {
            camera_fwd
        };

        // "Right" direction = cross(forward, view_dir).
        let view_dir = (center - camera_pos).normalize();
        let right = forward.cross(view_dir);
        let right_len = right.length();
        let right = if right_len > 1e-6 {
            right / right_len
        } else {
            // Degenerate case: velocity parallel to view direction.
            let alt = if forward.y.abs() < 0.99 {
                Vec3::Y
            } else {
                Vec3::X
            };
            forward.cross(alt).normalize()
        };

        let base_idx = self.vertex_buffer.len() as u32;

        // Quad: stretched along `forward`, width along `right`.
        let p0 = center - forward * half_length - right * half_width;
        let p1 = center - forward * half_length + right * half_width;
        let p2 = center + forward * half_length + right * half_width;
        let p3 = center + forward * half_length - right * half_width;

        self.vertex_buffer.push(ParticleVertex {
            position: [p0.x, p0.y, p0.z],
            uv: [uv_min.x, uv_max.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p1.x, p1.y, p1.z],
            uv: [uv_max.x, uv_max.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p2.x, p2.y, p2.z],
            uv: [uv_max.x, uv_min.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p3.x, p3.y, p3.z],
            uv: [uv_min.x, uv_min.y],
            color,
            sprite_blend,
        });

        self.index_buffer.push(base_idx);
        self.index_buffer.push(base_idx + 1);
        self.index_buffer.push(base_idx + 2);
        self.index_buffer.push(base_idx);
        self.index_buffer.push(base_idx + 2);
        self.index_buffer.push(base_idx + 3);
    }

    /// Emits a horizontal billboard (always faces up, Y-axis normal).
    fn emit_horizontal_billboard(
        &mut self,
        center: Vec3,
        size: f32,
        rotation: f32,
        color: [f32; 4],
        uv_min: Vec2,
        uv_max: Vec2,
        sprite_blend: [f32; 4],
    ) {
        let half = size * 0.5;
        let (sin_r, cos_r) = rotation.sin_cos();

        // Rotate in the XZ plane.
        let right = Vec3::new(cos_r, 0.0, sin_r) * half;
        let forward = Vec3::new(-sin_r, 0.0, cos_r) * half;

        let base_idx = self.vertex_buffer.len() as u32;

        let p0 = center - right - forward;
        let p1 = center + right - forward;
        let p2 = center + right + forward;
        let p3 = center - right + forward;

        self.vertex_buffer.push(ParticleVertex {
            position: [p0.x, p0.y, p0.z],
            uv: [uv_min.x, uv_max.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p1.x, p1.y, p1.z],
            uv: [uv_max.x, uv_max.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p2.x, p2.y, p2.z],
            uv: [uv_max.x, uv_min.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p3.x, p3.y, p3.z],
            uv: [uv_min.x, uv_min.y],
            color,
            sprite_blend,
        });

        self.index_buffer.push(base_idx);
        self.index_buffer.push(base_idx + 1);
        self.index_buffer.push(base_idx + 2);
        self.index_buffer.push(base_idx);
        self.index_buffer.push(base_idx + 2);
        self.index_buffer.push(base_idx + 3);
    }

    /// Emits a vertical billboard (faces camera around Y axis only).
    fn emit_vertical_billboard(
        &mut self,
        center: Vec3,
        size: f32,
        rotation: f32,
        color: [f32; 4],
        camera_pos: Vec3,
        uv_min: Vec2,
        uv_max: Vec2,
        sprite_blend: [f32; 4],
    ) {
        let half = size * 0.5;

        // Face the camera in the XZ plane.
        let to_camera = Vec3::new(
            camera_pos.x - center.x,
            0.0,
            camera_pos.z - center.z,
        );
        let dist = to_camera.length();
        let right = if dist > 1e-6 {
            let forward = to_camera / dist;
            Vec3::new(-forward.z, 0.0, forward.x)
        } else {
            Vec3::X
        };
        let up = Vec3::Y;

        // Apply rotation.
        let (sin_r, cos_r) = rotation.sin_cos();
        let r = (right * cos_r + up * sin_r) * half;
        let u = (-right * sin_r + up * cos_r) * half;

        let base_idx = self.vertex_buffer.len() as u32;

        let p0 = center - r - u;
        let p1 = center + r - u;
        let p2 = center + r + u;
        let p3 = center - r + u;

        self.vertex_buffer.push(ParticleVertex {
            position: [p0.x, p0.y, p0.z],
            uv: [uv_min.x, uv_max.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p1.x, p1.y, p1.z],
            uv: [uv_max.x, uv_max.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p2.x, p2.y, p2.z],
            uv: [uv_max.x, uv_min.y],
            color,
            sprite_blend,
        });
        self.vertex_buffer.push(ParticleVertex {
            position: [p3.x, p3.y, p3.z],
            uv: [uv_min.x, uv_min.y],
            color,
            sprite_blend,
        });

        self.index_buffer.push(base_idx);
        self.index_buffer.push(base_idx + 1);
        self.index_buffer.push(base_idx + 2);
        self.index_buffer.push(base_idx);
        self.index_buffer.push(base_idx + 2);
        self.index_buffer.push(base_idx + 3);
    }

    /// Generates instance data for instanced rendering (alternative to
    /// `generate_billboards`; the actual quad is shared and billboard
    /// computation happens in the vertex shader).
    pub fn generate_instances(
        &mut self,
        pool: &ParticlePool,
        _camera_pos: Vec3,
    ) {
        let count = pool.alive();
        self.instance_buffer.clear();
        self.instance_buffer.reserve(count);

        for i in 0..count {
            let vel = pool.velocities[i];
            let speed = vel.length();
            let vel_dir = if speed > 1e-6 {
                vel / speed
            } else {
                Vec3::ZERO
            };

            let (sprite_frame, sprite_blend) = if let Some(sheet) = &self.sprite_sheet {
                let (fa, _fb, blend) = sheet.evaluate(
                    pool.lifetimes[i],
                    pool.max_lifetimes[i],
                );
                (fa as f32, blend)
            } else {
                (0.0, 0.0)
            };

            self.instance_buffer.push(ParticleInstanceData {
                position: [
                    pool.positions[i].x,
                    pool.positions[i].y,
                    pool.positions[i].z,
                ],
                size: pool.sizes[i],
                color: pool.colors[i],
                rotation: pool.rotations[i],
                sprite_frame,
                sprite_blend,
                velocity_length: speed,
                velocity_dir: [vel_dir.x, vel_dir.y, vel_dir.z],
                _pad: 0.0,
            });
        }
    }

    /// Returns the generated vertex data.
    pub fn vertices(&self) -> &[ParticleVertex] {
        &self.vertex_buffer
    }

    /// Returns the generated index data.
    pub fn indices(&self) -> &[u32] {
        &self.index_buffer
    }

    /// Returns the generated instance data.
    pub fn instances(&self) -> &[ParticleInstanceData] {
        &self.instance_buffer
    }

    /// Returns the number of generated vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertex_buffer.len()
    }

    /// Returns the number of generated indices.
    pub fn index_count(&self) -> usize {
        self.index_buffer.len()
    }

    /// Returns the total memory footprint in bytes.
    pub fn memory_usage(&self) -> usize {
        self.vertex_buffer.capacity() * std::mem::size_of::<ParticleVertex>()
            + self.index_buffer.capacity() * std::mem::size_of::<u32>()
            + self.instance_buffer.capacity() * std::mem::size_of::<ParticleInstanceData>()
    }
}

impl Default for ParticleRenderer {
    fn default() -> Self {
        Self::new(ParticleRenderMode::Billboard)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::emitter::SpawnParams;

    fn make_pool(n: usize) -> ParticlePool {
        let mut pool = ParticlePool::new(n + 10);
        for i in 0..n {
            pool.spawn(&SpawnParams {
                position: Vec3::new(i as f32, 0.0, 0.0),
                velocity: Vec3::new(0.0, 1.0, 0.0),
                color: [1.0, 1.0, 1.0, 1.0],
                size: 1.0,
                lifetime: 5.0,
                rotation: 0.0,
                angular_velocity: 0.0,
                custom_data: [0.0; 4],
            });
        }
        pool
    }

    #[test]
    fn billboard_generates_correct_count() {
        let pool = make_pool(10);
        let mut renderer = ParticleRenderer::new(ParticleRenderMode::Billboard);
        renderer.generate_billboards(
            &pool,
            Vec3::new(0.0, 0.0, -10.0),
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
        );
        assert_eq!(renderer.vertex_count(), 40, "10 particles * 4 verts");
        assert_eq!(renderer.index_count(), 60, "10 particles * 6 indices");
    }

    #[test]
    fn sprite_sheet_frame_evaluation() {
        let sheet = SpriteSheet::new(4, 4)
            .with_loop_mode(SpriteLoopMode::OverLifetime);
        // At 50% lifetime, should be at frame ~7 (15 * 0.5 = 7.5).
        let (fa, fb, blend) = sheet.evaluate(1.0, 2.0);
        assert!(fa >= 7 && fa <= 8);
        assert!(fb >= fa);
    }

    #[test]
    fn sprite_sheet_uvs() {
        let sheet = SpriteSheet::new(4, 2);
        let (uv_min, uv_max) = sheet.frame_uvs(0);
        assert!((uv_min.x - 0.0).abs() < 0.001);
        assert!((uv_min.y - 0.0).abs() < 0.001);
        assert!((uv_max.x - 0.25).abs() < 0.001);
        assert!((uv_max.y - 0.5).abs() < 0.001);

        let (uv_min, uv_max) = sheet.frame_uvs(5);
        assert!((uv_min.x - 0.25).abs() < 0.001);
        assert!((uv_min.y - 0.5).abs() < 0.001);
    }

    #[test]
    fn soft_particle_fade() {
        let settings = SoftParticleSettings::new(1.0);
        assert!((settings.compute_fade(0.0) - 0.0).abs() < 0.01);
        assert!((settings.compute_fade(0.5) - 0.5).abs() < 0.01);
        assert!((settings.compute_fade(1.0) - 1.0).abs() < 0.01);
        assert!((settings.compute_fade(2.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn instance_data_generation() {
        let pool = make_pool(5);
        let mut renderer = ParticleRenderer::new(ParticleRenderMode::Billboard);
        renderer.generate_instances(&pool, Vec3::ZERO);
        assert_eq!(renderer.instances().len(), 5);
    }
}
