// engine/render/src/particles/collision.rs
//
// Particle collision detection and response. Supports plane, sphere, box,
// and heightmap colliders with configurable bounce, friction, and lifetime
// loss. Includes sub-frame collision detection to prevent tunnelling
// through thin surfaces.

use glam::Vec3;
use super::particle::ParticlePool;

// ---------------------------------------------------------------------------
// CollisionSettings
// ---------------------------------------------------------------------------

/// Configuration for how particles respond to collisions.
#[derive(Debug, Clone, Copy)]
pub struct CollisionSettings {
    /// Coefficient of restitution (bounce factor). 0 = no bounce, 1 = perfect.
    pub bounce: f32,
    /// Friction coefficient. Reduces tangential velocity on impact.
    pub friction: f32,
    /// Fraction of remaining lifetime lost on each collision. 0 = no loss.
    pub lifetime_loss: f32,
    /// Minimum particle radius for collision detection. Particles smaller
    /// than this are treated as points.
    pub radius_threshold: f32,
    /// If `true`, kill the particle on collision instead of bouncing.
    pub kill_on_collision: bool,
    /// Minimum velocity magnitude after collision. Below this, the particle
    /// is considered at rest and velocity is zeroed.
    pub min_bounce_velocity: f32,
    /// Maximum number of sub-steps per frame for tunnelling prevention.
    pub max_substeps: u32,
}

impl CollisionSettings {
    pub fn new() -> Self {
        Self {
            bounce: 0.5,
            friction: 0.3,
            lifetime_loss: 0.0,
            radius_threshold: 0.01,
            kill_on_collision: false,
            min_bounce_velocity: 0.1,
            max_substeps: 4,
        }
    }

    pub fn with_bounce(mut self, bounce: f32) -> Self {
        self.bounce = bounce;
        self
    }

    pub fn with_friction(mut self, friction: f32) -> Self {
        self.friction = friction;
        self
    }

    pub fn with_lifetime_loss(mut self, loss: f32) -> Self {
        self.lifetime_loss = loss;
        self
    }

    pub fn with_kill_on_collision(mut self, kill: bool) -> Self {
        self.kill_on_collision = kill;
        self
    }
}

impl Default for CollisionSettings {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ParticleCollider
// ---------------------------------------------------------------------------

/// A geometry primitive that particles can collide with.
#[derive(Debug, Clone)]
pub enum ParticleCollider {
    /// An infinite plane defined by a point and normal.
    Plane {
        /// A point on the plane.
        point: Vec3,
        /// The plane's outward-facing normal (unit vector).
        normal: Vec3,
    },
    /// A sphere collider.
    Sphere {
        /// Center of the sphere.
        center: Vec3,
        /// Radius.
        radius: f32,
        /// If `true`, particles collide with the inside surface (containment).
        invert: bool,
    },
    /// An axis-aligned box collider.
    Box {
        /// Minimum corner.
        min: Vec3,
        /// Maximum corner.
        max: Vec3,
        /// If `true`, particles collide with the inside surface.
        invert: bool,
    },
    /// A heightmap collider (terrain-like).
    Heightmap {
        /// Origin position (corner of the heightmap).
        origin: Vec3,
        /// Size of the heightmap in world units (X and Z).
        size: [f32; 2],
        /// Height values in row-major order.
        heights: Vec<f32>,
        /// Resolution (number of samples in X and Z).
        resolution: [u32; 2],
    },
}

impl ParticleCollider {
    /// Creates a ground plane at `y = height`.
    pub fn ground(height: f32) -> Self {
        ParticleCollider::Plane {
            point: Vec3::new(0.0, height, 0.0),
            normal: Vec3::Y,
        }
    }

    /// Creates a sphere collider.
    pub fn sphere(center: Vec3, radius: f32) -> Self {
        ParticleCollider::Sphere {
            center,
            radius,
            invert: false,
        }
    }

    /// Creates an inverted sphere (particles are contained inside).
    pub fn sphere_inverted(center: Vec3, radius: f32) -> Self {
        ParticleCollider::Sphere {
            center,
            radius,
            invert: true,
        }
    }

    /// Creates an axis-aligned box collider.
    pub fn aabb(min: Vec3, max: Vec3) -> Self {
        ParticleCollider::Box {
            min,
            max,
            invert: false,
        }
    }

    /// Creates an inverted box (particles contained inside).
    pub fn aabb_inverted(min: Vec3, max: Vec3) -> Self {
        ParticleCollider::Box {
            min,
            max,
            invert: true,
        }
    }

    /// Tests a single particle against this collider and returns collision
    /// info if a collision occurred.
    ///
    /// # Arguments
    /// * `pos` - Current particle position.
    /// * `vel` - Current particle velocity.
    /// * `radius` - Particle radius (for thickness).
    ///
    /// # Returns
    /// `Some((contact_point, normal, penetration_depth))` if collision, else `None`.
    pub fn test(
        &self,
        pos: Vec3,
        vel: Vec3,
        radius: f32,
    ) -> Option<CollisionResult> {
        match self {
            ParticleCollider::Plane { point, normal } => {
                test_plane(pos, vel, radius, *point, *normal)
            }
            ParticleCollider::Sphere {
                center,
                radius: sphere_radius,
                invert,
            } => test_sphere(pos, vel, radius, *center, *sphere_radius, *invert),
            ParticleCollider::Box { min, max, invert } => {
                test_box(pos, vel, radius, *min, *max, *invert)
            }
            ParticleCollider::Heightmap {
                origin,
                size,
                heights,
                resolution,
            } => test_heightmap(pos, vel, radius, *origin, size, heights, resolution),
        }
    }

    /// Tests a moving particle (sweep test) to prevent tunnelling.
    ///
    /// # Arguments
    /// * `pos_prev` - Position at start of frame.
    /// * `pos_next` - Position at end of frame (after integration).
    /// * `vel` - Current velocity.
    /// * `radius` - Particle radius.
    ///
    /// # Returns
    /// `Some((t, contact_point, normal))` where `t` is the fraction of the
    /// timestep at which collision occurs (0..1).
    pub fn sweep_test(
        &self,
        pos_prev: Vec3,
        pos_next: Vec3,
        vel: Vec3,
        radius: f32,
    ) -> Option<SweepResult> {
        match self {
            ParticleCollider::Plane { point, normal } => {
                sweep_plane(pos_prev, pos_next, radius, *point, *normal)
            }
            ParticleCollider::Sphere {
                center,
                radius: sphere_radius,
                invert,
            } => sweep_sphere(pos_prev, pos_next, radius, *center, *sphere_radius, *invert),
            _ => {
                // For complex shapes, fall back to end-of-frame test.
                if let Some(result) = self.test(pos_next, vel, radius) {
                    Some(SweepResult {
                        t: 1.0,
                        contact_point: result.contact_point,
                        normal: result.normal,
                    })
                } else {
                    None
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Collision results
// ---------------------------------------------------------------------------

/// Result of a static collision test.
#[derive(Debug, Clone, Copy)]
pub struct CollisionResult {
    /// The point of contact on the collider surface.
    pub contact_point: Vec3,
    /// The outward normal at the contact point.
    pub normal: Vec3,
    /// How far the particle has penetrated the surface.
    pub penetration: f32,
}

/// Result of a sweep (continuous) collision test.
#[derive(Debug, Clone, Copy)]
pub struct SweepResult {
    /// Fraction of the timestep at which collision occurs [0, 1].
    pub t: f32,
    /// Contact point.
    pub contact_point: Vec3,
    /// Surface normal at contact.
    pub normal: Vec3,
}

// ---------------------------------------------------------------------------
// Static collision tests
// ---------------------------------------------------------------------------

fn test_plane(
    pos: Vec3,
    _vel: Vec3,
    radius: f32,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<CollisionResult> {
    let dist = plane_normal.dot(pos - plane_point);
    if dist < radius {
        let penetration = radius - dist;
        let contact_point = pos - plane_normal * dist;
        Some(CollisionResult {
            contact_point,
            normal: plane_normal,
            penetration,
        })
    } else {
        None
    }
}

fn test_sphere(
    pos: Vec3,
    _vel: Vec3,
    radius: f32,
    center: Vec3,
    sphere_radius: f32,
    invert: bool,
) -> Option<CollisionResult> {
    let to_particle = pos - center;
    let dist = to_particle.length();

    if invert {
        // Particle should stay inside the sphere.
        let effective_radius = sphere_radius - radius;
        if dist > effective_radius && effective_radius > 0.0 {
            let normal = -to_particle / dist;
            let penetration = dist - effective_radius;
            let contact_point = center + to_particle.normalize() * sphere_radius;
            Some(CollisionResult {
                contact_point,
                normal,
                penetration,
            })
        } else {
            None
        }
    } else {
        // Particle should stay outside the sphere.
        let min_dist = sphere_radius + radius;
        if dist < min_dist && dist > 1e-6 {
            let normal = to_particle / dist;
            let penetration = min_dist - dist;
            let contact_point = center + normal * sphere_radius;
            Some(CollisionResult {
                contact_point,
                normal,
                penetration,
            })
        } else {
            None
        }
    }
}

fn test_box(
    pos: Vec3,
    _vel: Vec3,
    radius: f32,
    min: Vec3,
    max: Vec3,
    invert: bool,
) -> Option<CollisionResult> {
    if invert {
        // Particle should stay inside the box.
        let mut min_pen = f32::MAX;
        let mut best_normal = Vec3::ZERO;

        // Check each face.
        let faces = [
            (pos.x - (min.x + radius), Vec3::NEG_X), // left
            ((max.x - radius) - pos.x, Vec3::X),     // right
            (pos.y - (min.y + radius), Vec3::NEG_Y), // bottom
            ((max.y - radius) - pos.y, Vec3::Y),     // top
            (pos.z - (min.z + radius), Vec3::NEG_Z), // back
            ((max.z - radius) - pos.z, Vec3::Z),     // front
        ];

        let mut colliding = false;
        for (dist, normal) in &faces {
            if *dist < 0.0 {
                colliding = true;
                let pen = -dist;
                if pen < min_pen {
                    min_pen = pen;
                    best_normal = -*normal;
                }
            }
        }

        if colliding {
            let contact = pos + best_normal * min_pen;
            Some(CollisionResult {
                contact_point: contact,
                normal: best_normal,
                penetration: min_pen,
            })
        } else {
            None
        }
    } else {
        // Particle should stay outside the box.
        // Find the closest point on the box surface.
        let closest = Vec3::new(
            pos.x.clamp(min.x, max.x),
            pos.y.clamp(min.y, max.y),
            pos.z.clamp(min.z, max.z),
        );

        let to_particle = pos - closest;
        let dist = to_particle.length();

        if dist < radius && dist > 1e-6 {
            let normal = to_particle / dist;
            Some(CollisionResult {
                contact_point: closest,
                normal,
                penetration: radius - dist,
            })
        } else if dist < 1e-6 {
            // Particle is inside the box -- push it out through the nearest face.
            let center = (min + max) * 0.5;
            let half = (max - min) * 0.5;
            let local = pos - center;

            let mut min_overlap = f32::MAX;
            let mut push_normal = Vec3::Y;

            for axis in 0..3 {
                let overlap = half[axis] - local[axis].abs();
                if overlap < min_overlap {
                    min_overlap = overlap;
                    let mut n = Vec3::ZERO;
                    n[axis] = if local[axis] >= 0.0 { 1.0 } else { -1.0 };
                    push_normal = n;
                }
            }

            Some(CollisionResult {
                contact_point: pos,
                normal: push_normal,
                penetration: min_overlap + radius,
            })
        } else {
            None
        }
    }
}

fn test_heightmap(
    pos: Vec3,
    _vel: Vec3,
    radius: f32,
    origin: Vec3,
    size: &[f32; 2],
    heights: &[f32],
    resolution: &[u32; 2],
) -> Option<CollisionResult> {
    if resolution[0] < 2 || resolution[1] < 2 || heights.is_empty() {
        return None;
    }

    // Map world XZ to heightmap UV.
    let local_x = pos.x - origin.x;
    let local_z = pos.z - origin.z;

    if local_x < 0.0 || local_x > size[0] || local_z < 0.0 || local_z > size[1] {
        return None;
    }

    let u = (local_x / size[0]) * (resolution[0] - 1) as f32;
    let v = (local_z / size[1]) * (resolution[1] - 1) as f32;

    let ix = u.floor() as u32;
    let iz = v.floor() as u32;
    let fx = u.fract();
    let fz = v.fract();

    // Clamp indices.
    let ix = ix.min(resolution[0] - 2);
    let iz = iz.min(resolution[1] - 2);

    // Bilinear interpolation of height.
    let idx = |x: u32, z: u32| -> usize {
        (z * resolution[0] + x) as usize
    };

    let h00 = heights.get(idx(ix, iz)).copied().unwrap_or(0.0);
    let h10 = heights.get(idx(ix + 1, iz)).copied().unwrap_or(0.0);
    let h01 = heights.get(idx(ix, iz + 1)).copied().unwrap_or(0.0);
    let h11 = heights.get(idx(ix + 1, iz + 1)).copied().unwrap_or(0.0);

    let h0 = h00 + (h10 - h00) * fx;
    let h1 = h01 + (h11 - h01) * fx;
    let height = h0 + (h1 - h0) * fz + origin.y;

    if pos.y < height + radius {
        // Compute normal via central differences.
        let cell_x = size[0] / (resolution[0] - 1) as f32;
        let cell_z = size[1] / (resolution[1] - 1) as f32;

        let hx_minus = if ix > 0 {
            heights.get(idx(ix - 1, iz)).copied().unwrap_or(h00)
        } else {
            h00
        };
        let hx_plus = heights
            .get(idx((ix + 1).min(resolution[0] - 1), iz))
            .copied()
            .unwrap_or(h00);
        let hz_minus = if iz > 0 {
            heights.get(idx(ix, iz - 1)).copied().unwrap_or(h00)
        } else {
            h00
        };
        let hz_plus = heights
            .get(idx(ix, (iz + 1).min(resolution[1] - 1)))
            .copied()
            .unwrap_or(h00);

        let dx = (hx_plus - hx_minus) / (2.0 * cell_x);
        let dz = (hz_plus - hz_minus) / (2.0 * cell_z);

        let normal = Vec3::new(-dx, 1.0, -dz).normalize();

        let penetration = height + radius - pos.y;
        let contact_point = Vec3::new(pos.x, height, pos.z);

        Some(CollisionResult {
            contact_point,
            normal,
            penetration,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Sweep (continuous) collision tests
// ---------------------------------------------------------------------------

fn sweep_plane(
    pos_prev: Vec3,
    pos_next: Vec3,
    radius: f32,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<SweepResult> {
    let d_prev = plane_normal.dot(pos_prev - plane_point) - radius;
    let d_next = plane_normal.dot(pos_next - plane_point) - radius;

    // If both on the same side and both positive, no collision.
    if d_prev >= 0.0 && d_next >= 0.0 {
        return None;
    }

    // If starting below the plane, report immediate collision.
    if d_prev < 0.0 {
        return Some(SweepResult {
            t: 0.0,
            contact_point: pos_prev - plane_normal * d_prev,
            normal: plane_normal,
        });
    }

    // Crossing: find the exact t.
    let denom = d_prev - d_next;
    if denom.abs() < 1e-9 {
        return None;
    }

    let t = d_prev / denom;
    let t = t.clamp(0.0, 1.0);
    let contact_pos = pos_prev + (pos_next - pos_prev) * t;
    let contact_point = contact_pos - plane_normal * radius;

    Some(SweepResult {
        t,
        contact_point,
        normal: plane_normal,
    })
}

fn sweep_sphere(
    pos_prev: Vec3,
    pos_next: Vec3,
    particle_radius: f32,
    center: Vec3,
    sphere_radius: f32,
    invert: bool,
) -> Option<SweepResult> {
    let combined_radius = if invert {
        sphere_radius - particle_radius
    } else {
        sphere_radius + particle_radius
    };

    if combined_radius <= 0.0 {
        return None;
    }

    // Ray-sphere intersection.
    let ray_origin = pos_prev - center;
    let ray_dir = pos_next - pos_prev;
    let ray_len_sq = ray_dir.length_squared();

    if ray_len_sq < 1e-12 {
        // Not moving; check static.
        let dist = ray_origin.length();
        if invert && dist > combined_radius {
            let normal = -ray_origin.normalize();
            return Some(SweepResult {
                t: 0.0,
                contact_point: center + ray_origin.normalize() * sphere_radius,
                normal,
            });
        } else if !invert && dist < combined_radius {
            let normal = ray_origin.normalize();
            return Some(SweepResult {
                t: 0.0,
                contact_point: center + normal * sphere_radius,
                normal,
            });
        }
        return None;
    }

    let a = ray_len_sq;
    let b = 2.0 * ray_origin.dot(ray_dir);
    let c = ray_origin.length_squared() - combined_radius * combined_radius;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }

    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);

    let t = if invert {
        // For inverted sphere, we want the exit point (t2).
        // Collision happens when particle crosses the boundary going outward.
        if t2 >= 0.0 && t2 <= 1.0 {
            t2
        } else if t1 >= 0.0 && t1 <= 1.0 {
            t1
        } else {
            return None;
        }
    } else {
        // Normal sphere: first intersection.
        if t1 >= 0.0 && t1 <= 1.0 {
            t1
        } else if t2 >= 0.0 && t2 <= 1.0 {
            t2
        } else {
            return None;
        }
    };

    let hit_pos = pos_prev + ray_dir * t;
    let to_hit = hit_pos - center;
    let dist = to_hit.length();
    let normal = if invert {
        if dist > 1e-6 { -to_hit / dist } else { Vec3::Y }
    } else {
        if dist > 1e-6 { to_hit / dist } else { Vec3::Y }
    };
    let contact_point = center + (if invert { -normal } else { normal }) * sphere_radius;

    Some(SweepResult {
        t,
        contact_point,
        normal,
    })
}

// ---------------------------------------------------------------------------
// CollisionWorld
// ---------------------------------------------------------------------------

/// Manages a set of colliders and resolves particle collisions.
pub struct CollisionWorld {
    /// Active colliders.
    pub colliders: Vec<ParticleCollider>,
    /// Collision response settings.
    pub settings: CollisionSettings,
}

impl CollisionWorld {
    pub fn new() -> Self {
        Self {
            colliders: Vec::new(),
            settings: CollisionSettings::default(),
        }
    }

    pub fn with_settings(mut self, settings: CollisionSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Adds a collider.
    pub fn add(&mut self, collider: ParticleCollider) {
        self.colliders.push(collider);
    }

    /// Removes all colliders.
    pub fn clear(&mut self) {
        self.colliders.clear();
    }

    /// Resolves collisions for all alive particles in the pool.
    ///
    /// This modifies particle positions and velocities in-place. Particles
    /// that should be killed (due to `kill_on_collision` or lifetime loss)
    /// have their max_lifetime set to their current lifetime so they will
    /// be removed on the next update.
    ///
    /// # Arguments
    /// * `pool` - The particle pool to check.
    /// * `dt` - Frame delta time (for sub-step tunnelling prevention).
    /// * `collision_events` - Optional output buffer for collision events.
    pub fn resolve(
        &self,
        pool: &mut ParticlePool,
        dt: f32,
        mut collision_events: Option<&mut Vec<CollisionEvent>>,
    ) {
        if self.colliders.is_empty() || pool.alive_count == 0 {
            return;
        }

        for i in 0..pool.alive_count {
            let pos = pool.positions[i];
            let vel = pool.velocities[i];
            let particle_radius = pool.sizes[i] * 0.5;
            let radius = particle_radius.max(self.settings.radius_threshold);

            for collider in &self.colliders {
                // Use sweep test for fast-moving particles.
                let speed = vel.length();
                let travel = speed * dt;

                let collision = if travel > radius * 2.0
                    && self.settings.max_substeps > 0
                {
                    // Sub-frame test: the particle travels more than its
                    // own diameter, so it might tunnel.
                    let prev_pos = pos - vel * dt;
                    collider.sweep_test(prev_pos, pos, vel, radius)
                        .map(|sweep| {
                            CollisionResult {
                                contact_point: sweep.contact_point,
                                normal: sweep.normal,
                                penetration: 0.0, // resolved by repositioning
                            }
                        })
                } else {
                    collider.test(pos, vel, radius)
                };

                if let Some(result) = collision {
                    if self.settings.kill_on_collision {
                        // Kill the particle.
                        pool.max_lifetimes[i] = pool.lifetimes[i];
                        if let Some(events) = collision_events.as_deref_mut() {
                            events.push(CollisionEvent {
                                particle_index: i,
                                position: result.contact_point,
                                normal: result.normal,
                                velocity: vel,
                            });
                        }
                        break;
                    }

                    // Resolve penetration.
                    if result.penetration > 0.0 {
                        pool.positions[i] += result.normal * result.penetration;
                    }

                    // Reflect velocity.
                    let vel_normal = result.normal * vel.dot(result.normal);
                    let vel_tangent = vel - vel_normal;

                    let reflected_normal = -vel_normal * self.settings.bounce;
                    let reflected_tangent =
                        vel_tangent * (1.0 - self.settings.friction);

                    let new_vel = reflected_normal + reflected_tangent;

                    // Apply minimum bounce velocity.
                    if new_vel.length() < self.settings.min_bounce_velocity {
                        pool.velocities[i] = Vec3::ZERO;
                    } else {
                        pool.velocities[i] = new_vel;
                    }

                    // Apply lifetime loss.
                    if self.settings.lifetime_loss > 0.0 {
                        pool.reduce_lifetime(
                            i,
                            pool.max_lifetimes[i] * self.settings.lifetime_loss,
                        );
                    }

                    if let Some(events) = collision_events.as_deref_mut() {
                        events.push(CollisionEvent {
                            particle_index: i,
                            position: result.contact_point,
                            normal: result.normal,
                            velocity: vel,
                        });
                    }
                }
            }
        }
    }
}

impl Default for CollisionWorld {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CollisionEvent
// ---------------------------------------------------------------------------

/// Describes a particle collision event. Can be used to trigger sub-emitters,
/// sound effects, or decal spawning.
#[derive(Debug, Clone, Copy)]
pub struct CollisionEvent {
    /// Index of the colliding particle in the pool.
    pub particle_index: usize,
    /// World-space contact position.
    pub position: Vec3,
    /// Surface normal at the contact point.
    pub normal: Vec3,
    /// Particle velocity at the moment of impact.
    pub velocity: Vec3,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_collision_basic() {
        let collider = ParticleCollider::ground(0.0);
        // Particle below the plane.
        let result = collider.test(
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::NEG_Y,
            0.1,
        );
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.normal.y > 0.0);
    }

    #[test]
    fn plane_no_collision_above() {
        let collider = ParticleCollider::ground(0.0);
        let result = collider.test(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::NEG_Y,
            0.1,
        );
        assert!(result.is_none());
    }

    #[test]
    fn sphere_collision_outside() {
        let collider = ParticleCollider::sphere(Vec3::ZERO, 1.0);
        // Particle inside the sphere.
        let result = collider.test(Vec3::new(0.5, 0.0, 0.0), Vec3::ZERO, 0.1);
        assert!(result.is_some());
    }

    #[test]
    fn inverted_sphere_contains() {
        let collider = ParticleCollider::sphere_inverted(Vec3::ZERO, 5.0);
        // Particle outside should collide.
        let result = collider.test(Vec3::new(6.0, 0.0, 0.0), Vec3::X, 0.1);
        assert!(result.is_some());
    }

    #[test]
    fn sweep_plane_tunnelling() {
        let collider = ParticleCollider::ground(0.0);
        // Particle passes through the plane in one frame.
        let result = collider.sweep_test(
            Vec3::new(0.0, 1.0, 0.0),   // above
            Vec3::new(0.0, -1.0, 0.0),  // below
            Vec3::new(0.0, -2.0, 0.0),
            0.01,
        );
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.t >= 0.0 && r.t <= 1.0);
    }

    #[test]
    fn collision_world_resolves() {
        let mut world = CollisionWorld::new()
            .with_settings(CollisionSettings::new().with_bounce(0.5));
        world.add(ParticleCollider::ground(0.0));

        let mut pool = ParticlePool::new(10);
        pool.spawn(&super::super::emitter::SpawnParams {
            position: Vec3::new(0.0, -0.1, 0.0),
            velocity: Vec3::new(0.0, -10.0, 0.0),
            color: [1.0; 4],
            size: 0.2,
            lifetime: 5.0,
            rotation: 0.0,
            angular_velocity: 0.0,
            custom_data: [0.0; 4],
        });

        let mut events = Vec::new();
        world.resolve(&mut pool, 0.016, Some(&mut events));

        // Particle should have been pushed above the plane.
        assert!(pool.positions[0].y >= 0.0, "Particle should be above plane");
        // Velocity should be reflected upward.
        assert!(pool.velocities[0].y > 0.0, "Velocity should bounce up");
    }

    #[test]
    fn heightmap_collision() {
        // 3x3 heightmap with a bump in the middle.
        let heights = vec![
            0.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let collider = ParticleCollider::Heightmap {
            origin: Vec3::ZERO,
            size: [2.0, 2.0],
            heights,
            resolution: [3, 3],
        };

        // Test at the center (should be at height ~2.0).
        let result = collider.test(
            Vec3::new(1.0, 1.5, 1.0),
            Vec3::NEG_Y,
            0.1,
        );
        assert!(result.is_some(), "Should collide with heightmap");
    }
}
