// engine/render/src/particles/particle.rs
//
// Structure-of-Arrays (SoA) particle pool and simulation. The SoA layout
// ensures that per-attribute iteration (e.g., "update all positions") touches
// contiguous memory, maximising cache-line utilisation and enabling future
// SIMD vectorisation.
//
// Dead particles are removed via swap-remove so the alive set is always
// densely packed at the front of each array.

use glam::Vec3;
use super::emitter::SpawnParams;
use super::{ColorGradient, Curve};

// ---------------------------------------------------------------------------
// SortMode
// ---------------------------------------------------------------------------

/// How particles are sorted before rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SortMode {
    /// No sorting (fastest).
    None,
    /// Sort back-to-front relative to the camera (required for correct
    /// alpha blending).
    BackToFront,
    /// Sort front-to-back (useful for opaque particles with early-Z).
    FrontToBack,
    /// Sort by age (oldest first).
    OldestFirst,
    /// Sort by age (youngest first).
    YoungestFirst,
    /// Sort by remaining lifetime (shortest first).
    ShortestLifetimeFirst,
}

impl Default for SortMode {
    fn default() -> Self {
        SortMode::None
    }
}

// ---------------------------------------------------------------------------
// ParticlePool
// ---------------------------------------------------------------------------

/// A Structure-of-Arrays pool of particles.
///
/// Each attribute is stored in its own `Vec` so that iteration over a single
/// attribute (e.g., all positions) is cache-friendly. The first
/// `alive_count` entries in each array are live particles; everything past
/// that index is dead / garbage.
#[derive(Debug, Clone)]
pub struct ParticlePool {
    /// World-space positions.
    pub positions: Vec<Vec3>,
    /// Velocities (units per second).
    pub velocities: Vec<Vec3>,
    /// RGBA colours.
    pub colors: Vec<[f32; 4]>,
    /// Billboard sizes (diameter or radius depending on the renderer).
    pub sizes: Vec<f32>,
    /// Elapsed time since birth (seconds).
    pub lifetimes: Vec<f32>,
    /// Total lifetime at birth (seconds). The particle dies when
    /// `lifetimes[i] >= max_lifetimes[i]`.
    pub max_lifetimes: Vec<f32>,
    /// 2-D rotation angle in radians (for billboard spin).
    pub rotations: Vec<f32>,
    /// Angular velocity in radians per second.
    pub angular_velocities: Vec<f32>,
    /// User-defined custom data channel (4 floats per particle).
    pub custom_data: Vec<[f32; 4]>,
    /// Initial sizes at birth (cached for size-over-lifetime evaluation).
    pub initial_sizes: Vec<f32>,
    /// Initial colors at birth (cached for color-over-lifetime evaluation).
    pub initial_colors: Vec<[f32; 4]>,
    /// Initial speeds at birth (cached for speed-over-lifetime evaluation).
    pub initial_speeds: Vec<f32>,
    /// Normalized velocity directions at birth.
    pub initial_directions: Vec<Vec3>,
    /// Number of live particles.
    pub alive_count: usize,
    /// Sort keys (reused each frame to avoid allocation).
    sort_keys: Vec<f32>,
    /// Sort indices (reused each frame).
    sort_indices: Vec<usize>,
}

impl ParticlePool {
    /// Creates a new pool pre-allocated for `capacity` particles.
    pub fn new(capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity),
            colors: Vec::with_capacity(capacity),
            sizes: Vec::with_capacity(capacity),
            lifetimes: Vec::with_capacity(capacity),
            max_lifetimes: Vec::with_capacity(capacity),
            rotations: Vec::with_capacity(capacity),
            angular_velocities: Vec::with_capacity(capacity),
            custom_data: Vec::with_capacity(capacity),
            initial_sizes: Vec::with_capacity(capacity),
            initial_colors: Vec::with_capacity(capacity),
            initial_speeds: Vec::with_capacity(capacity),
            initial_directions: Vec::with_capacity(capacity),
            alive_count: 0,
            sort_keys: Vec::with_capacity(capacity),
            sort_indices: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of live particles.
    #[inline]
    pub fn alive(&self) -> usize {
        self.alive_count
    }

    /// Returns `true` if there are no live particles.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.alive_count == 0
    }

    /// Returns the allocated capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.positions.capacity()
    }

    /// Spawns a particle from the given parameters.
    ///
    /// If the pool is already at capacity, the particle is silently dropped.
    pub fn spawn(&mut self, params: &SpawnParams) {
        if self.alive_count >= self.positions.capacity() {
            return;
        }

        let speed = params.velocity.length();
        let dir = if speed > 1e-9 {
            params.velocity / speed
        } else {
            Vec3::Y
        };

        if self.alive_count < self.positions.len() {
            // Reuse slot (overwrite dead data).
            let i = self.alive_count;
            self.positions[i] = params.position;
            self.velocities[i] = params.velocity;
            self.colors[i] = params.color;
            self.sizes[i] = params.size;
            self.lifetimes[i] = 0.0;
            self.max_lifetimes[i] = params.lifetime;
            self.rotations[i] = params.rotation;
            self.angular_velocities[i] = params.angular_velocity;
            self.custom_data[i] = params.custom_data;
            self.initial_sizes[i] = params.size;
            self.initial_colors[i] = params.color;
            self.initial_speeds[i] = speed;
            self.initial_directions[i] = dir;
        } else {
            // Grow arrays.
            self.positions.push(params.position);
            self.velocities.push(params.velocity);
            self.colors.push(params.color);
            self.sizes.push(params.size);
            self.lifetimes.push(0.0);
            self.max_lifetimes.push(params.lifetime);
            self.rotations.push(params.rotation);
            self.angular_velocities.push(params.angular_velocity);
            self.custom_data.push(params.custom_data);
            self.initial_sizes.push(params.size);
            self.initial_colors.push(params.color);
            self.initial_speeds.push(speed);
            self.initial_directions.push(dir);
        }
        self.alive_count += 1;
    }

    /// Spawns multiple particles at once.
    pub fn spawn_batch(&mut self, params_list: &[SpawnParams]) {
        for p in params_list {
            self.spawn(p);
        }
    }

    /// Integrates all live particles forward by `dt` seconds.
    ///
    /// This performs:
    /// 1. Velocity integration (position += velocity * dt)
    /// 2. Rotation integration
    /// 3. Lifetime advancement
    /// 4. Size-over-lifetime evaluation
    /// 5. Color-over-lifetime evaluation
    /// 6. Speed-over-lifetime evaluation
    /// 7. Dead-particle removal via swap-remove
    pub fn update(
        &mut self,
        dt: f32,
        gravity: Vec3,
        size_curve: &Curve,
        color_gradient: &ColorGradient,
        speed_curve: &Curve,
    ) {
        if self.alive_count == 0 {
            return;
        }

        // -- Phase 1: Integrate and age --
        // We iterate backwards so swap-remove doesn't skip elements.
        let mut i = 0;
        while i < self.alive_count {
            // Age the particle.
            self.lifetimes[i] += dt;

            // Check death.
            if self.lifetimes[i] >= self.max_lifetimes[i] {
                self.swap_remove(i);
                // Don't increment i; the swapped-in particle needs processing.
                continue;
            }

            // Normalized age [0, 1].
            let t = self.lifetimes[i] / self.max_lifetimes[i];

            // Apply gravity.
            self.velocities[i] += gravity * dt;

            // Speed-over-lifetime: scale velocity magnitude.
            let speed_mult = speed_curve.evaluate(t);
            let current_speed = self.velocities[i].length();
            if current_speed > 1e-9 {
                let desired_speed = self.initial_speeds[i] * speed_mult;
                // Blend towards the desired speed while preserving gravity effects.
                let dir = self.velocities[i] / current_speed;
                // Don't apply speed curve aggressively; blend it.
                self.velocities[i] = self.velocities[i] * (1.0 - 0.5 * dt).max(0.0)
                    + dir * desired_speed * (0.5 * dt).min(1.0);
            }

            // Integrate position.
            self.positions[i] += self.velocities[i] * dt;

            // Integrate rotation.
            self.rotations[i] += self.angular_velocities[i] * dt;

            // Size over lifetime.
            let size_mult = size_curve.evaluate(t);
            self.sizes[i] = self.initial_sizes[i] * size_mult;

            // Color over lifetime.
            self.colors[i] = color_gradient.evaluate(t);

            i += 1;
        }
    }

    /// Simple update without curves (just physics integration).
    pub fn update_simple(&mut self, dt: f32, gravity: Vec3) {
        let mut i = 0;
        while i < self.alive_count {
            self.lifetimes[i] += dt;
            if self.lifetimes[i] >= self.max_lifetimes[i] {
                self.swap_remove(i);
                continue;
            }

            self.velocities[i] += gravity * dt;
            self.positions[i] += self.velocities[i] * dt;
            self.rotations[i] += self.angular_velocities[i] * dt;

            // Simple linear interpolation for color.
            let t = self.lifetimes[i] / self.max_lifetimes[i];
            let ic = &self.initial_colors[i];
            // Fade alpha linearly.
            self.colors[i][3] = ic[3] * (1.0 - t);

            i += 1;
        }
    }

    /// Applies an external force to all particles. Used by the force field
    /// system to avoid duplicate iteration.
    pub fn apply_force(&mut self, force: Vec3, dt: f32) {
        for i in 0..self.alive_count {
            self.velocities[i] += force * dt;
        }
    }

    /// Applies a position-dependent acceleration to all particles.
    /// The closure receives (position, velocity) and returns an acceleration.
    pub fn apply_force_fn<F>(&mut self, dt: f32, f: F)
    where
        F: Fn(Vec3, Vec3) -> Vec3,
    {
        for i in 0..self.alive_count {
            let acc = f(self.positions[i], self.velocities[i]);
            self.velocities[i] += acc * dt;
        }
    }

    /// Sorts particles according to the given mode.
    ///
    /// For `BackToFront` and `FrontToBack`, `camera_pos` is used.
    /// The sort is performed in-place by computing sort keys, sorting
    /// indices, and then reordering all arrays according to the sorted
    /// order.
    pub fn sort(&mut self, mode: SortMode, camera_pos: Vec3) {
        if self.alive_count <= 1 || mode == SortMode::None {
            return;
        }

        let n = self.alive_count;

        // Resize working buffers.
        self.sort_keys.resize(n, 0.0);
        self.sort_indices.resize(n, 0);

        // Compute sort keys.
        match mode {
            SortMode::BackToFront => {
                for i in 0..n {
                    // Negative distance so that farther particles sort first.
                    self.sort_keys[i] =
                        -(self.positions[i] - camera_pos).length_squared();
                }
            }
            SortMode::FrontToBack => {
                for i in 0..n {
                    self.sort_keys[i] =
                        (self.positions[i] - camera_pos).length_squared();
                }
            }
            SortMode::OldestFirst => {
                for i in 0..n {
                    // Older = higher lifetime = sort ascending.
                    self.sort_keys[i] = -self.lifetimes[i];
                }
            }
            SortMode::YoungestFirst => {
                for i in 0..n {
                    self.sort_keys[i] = self.lifetimes[i];
                }
            }
            SortMode::ShortestLifetimeFirst => {
                for i in 0..n {
                    self.sort_keys[i] =
                        -(self.max_lifetimes[i] - self.lifetimes[i]);
                }
            }
            SortMode::None => unreachable!(),
        }

        // Initialize index array.
        for i in 0..n {
            self.sort_indices[i] = i;
        }

        // Sort indices by key.
        let keys = &self.sort_keys;
        self.sort_indices[..n].sort_by(|&a, &b| {
            keys[a]
                .partial_cmp(&keys[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Reorder all arrays in-place using the sorted index permutation.
        // We use a cycle-sort approach to do this with O(1) extra memory
        // per attribute.
        self.reorder_by_indices(n);
    }

    /// Reorders all SoA arrays according to `self.sort_indices`.
    ///
    /// Uses the classic permutation cycle algorithm: for each position,
    /// follow the chain of `sort_indices` until we return to the start,
    /// rotating elements along the way.
    fn reorder_by_indices(&mut self, n: usize) {
        // We need a visited bitmap. Use sort_keys as scratch (it's f32 but
        // we only need to track "visited" -- we'll use NaN as sentinel).
        // Actually, let's just use a small bitvec.
        let mut visited = vec![false; n];

        for start in 0..n {
            if visited[start] || self.sort_indices[start] == start {
                visited[start] = true;
                continue;
            }

            // Follow the permutation cycle.
            let mut current = start;
            let tmp_pos = self.positions[start];
            let tmp_vel = self.velocities[start];
            let tmp_col = self.colors[start];
            let tmp_size = self.sizes[start];
            let tmp_life = self.lifetimes[start];
            let tmp_max_life = self.max_lifetimes[start];
            let tmp_rot = self.rotations[start];
            let tmp_ang = self.angular_velocities[start];
            let tmp_custom = self.custom_data[start];
            let tmp_init_size = self.initial_sizes[start];
            let tmp_init_color = self.initial_colors[start];
            let tmp_init_speed = self.initial_speeds[start];
            let tmp_init_dir = self.initial_directions[start];

            loop {
                let next = self.sort_indices[current];
                visited[current] = true;

                if next == start {
                    self.positions[current] = tmp_pos;
                    self.velocities[current] = tmp_vel;
                    self.colors[current] = tmp_col;
                    self.sizes[current] = tmp_size;
                    self.lifetimes[current] = tmp_life;
                    self.max_lifetimes[current] = tmp_max_life;
                    self.rotations[current] = tmp_rot;
                    self.angular_velocities[current] = tmp_ang;
                    self.custom_data[current] = tmp_custom;
                    self.initial_sizes[current] = tmp_init_size;
                    self.initial_colors[current] = tmp_init_color;
                    self.initial_speeds[current] = tmp_init_speed;
                    self.initial_directions[current] = tmp_init_dir;
                    break;
                }

                self.positions[current] = self.positions[next];
                self.velocities[current] = self.velocities[next];
                self.colors[current] = self.colors[next];
                self.sizes[current] = self.sizes[next];
                self.lifetimes[current] = self.lifetimes[next];
                self.max_lifetimes[current] = self.max_lifetimes[next];
                self.rotations[current] = self.rotations[next];
                self.angular_velocities[current] = self.angular_velocities[next];
                self.custom_data[current] = self.custom_data[next];
                self.initial_sizes[current] = self.initial_sizes[next];
                self.initial_colors[current] = self.initial_colors[next];
                self.initial_speeds[current] = self.initial_speeds[next];
                self.initial_directions[current] = self.initial_directions[next];

                current = next;
            }
        }
    }

    /// Removes the particle at index `i` by swapping it with the last alive
    /// particle. This maintains density (no holes).
    fn swap_remove(&mut self, i: usize) {
        let last = self.alive_count - 1;
        if i != last {
            self.positions.swap(i, last);
            self.velocities.swap(i, last);
            self.colors.swap(i, last);
            self.sizes.swap(i, last);
            self.lifetimes.swap(i, last);
            self.max_lifetimes.swap(i, last);
            self.rotations.swap(i, last);
            self.angular_velocities.swap(i, last);
            self.custom_data.swap(i, last);
            self.initial_sizes.swap(i, last);
            self.initial_colors.swap(i, last);
            self.initial_speeds.swap(i, last);
            self.initial_directions.swap(i, last);
        }
        self.alive_count -= 1;
    }

    /// Kills all particles.
    pub fn clear(&mut self) {
        self.alive_count = 0;
    }

    /// Returns the normalized age of particle `i` in `[0, 1]`.
    #[inline]
    pub fn normalized_age(&self, i: usize) -> f32 {
        if self.max_lifetimes[i] > 0.0 {
            (self.lifetimes[i] / self.max_lifetimes[i]).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Returns the remaining lifetime of particle `i`.
    #[inline]
    pub fn remaining_lifetime(&self, i: usize) -> f32 {
        (self.max_lifetimes[i] - self.lifetimes[i]).max(0.0)
    }

    /// Returns the axis-aligned bounding box of all live particles.
    pub fn compute_aabb(&self) -> (Vec3, Vec3) {
        if self.alive_count == 0 {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let mut min = self.positions[0];
        let mut max = self.positions[0];
        for i in 1..self.alive_count {
            let p = self.positions[i];
            min = min.min(p);
            max = max.max(p);
        }
        // Expand by max particle size to account for billboard extents.
        let max_size = self.sizes[..self.alive_count]
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);
        let expand = Vec3::splat(max_size * 0.5);
        (min - expand, max + expand)
    }

    /// Returns the center of mass of all live particles.
    pub fn center_of_mass(&self) -> Vec3 {
        if self.alive_count == 0 {
            return Vec3::ZERO;
        }
        let mut sum = Vec3::ZERO;
        for i in 0..self.alive_count {
            sum += self.positions[i];
        }
        sum / self.alive_count as f32
    }

    /// Returns the maximum speed among all live particles.
    pub fn max_speed(&self) -> f32 {
        let mut max_sq = 0.0f32;
        for i in 0..self.alive_count {
            let sq = self.velocities[i].length_squared();
            if sq > max_sq {
                max_sq = sq;
            }
        }
        max_sq.sqrt()
    }

    /// Returns iterators over position and color slices for the alive particles.
    #[inline]
    pub fn alive_positions(&self) -> &[Vec3] {
        &self.positions[..self.alive_count]
    }

    #[inline]
    pub fn alive_velocities(&self) -> &[Vec3] {
        &self.velocities[..self.alive_count]
    }

    #[inline]
    pub fn alive_colors(&self) -> &[[f32; 4]] {
        &self.colors[..self.alive_count]
    }

    #[inline]
    pub fn alive_sizes(&self) -> &[f32] {
        &self.sizes[..self.alive_count]
    }

    #[inline]
    pub fn alive_rotations(&self) -> &[f32] {
        &self.rotations[..self.alive_count]
    }

    #[inline]
    pub fn alive_lifetimes(&self) -> &[f32] {
        &self.lifetimes[..self.alive_count]
    }

    #[inline]
    pub fn alive_max_lifetimes(&self) -> &[f32] {
        &self.max_lifetimes[..self.alive_count]
    }

    /// Returns position and velocity at index `i` (for force field evaluation).
    #[inline]
    pub fn get_pos_vel(&self, i: usize) -> (Vec3, Vec3) {
        (self.positions[i], self.velocities[i])
    }

    /// Sets the velocity at index `i`.
    #[inline]
    pub fn set_velocity(&mut self, i: usize, vel: Vec3) {
        self.velocities[i] = vel;
    }

    /// Reduces the lifetime of particle `i` by the given amount.
    pub fn reduce_lifetime(&mut self, i: usize, amount: f32) {
        self.max_lifetimes[i] = (self.max_lifetimes[i] - amount).max(self.lifetimes[i]);
    }

    /// Sets a particle's position (for collision response).
    #[inline]
    pub fn set_position(&mut self, i: usize, pos: Vec3) {
        self.positions[i] = pos;
    }
}

impl Default for ParticlePool {
    fn default() -> Self {
        Self::new(1000)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_and_kill() {
        let mut pool = ParticlePool::new(100);
        let params = SpawnParams {
            position: Vec3::ZERO,
            velocity: Vec3::Y,
            color: [1.0; 4],
            size: 1.0,
            lifetime: 0.5,
            rotation: 0.0,
            angular_velocity: 0.0,
            custom_data: [0.0; 4],
        };
        pool.spawn(&params);
        assert_eq!(pool.alive(), 1);

        // After 0.6s the particle should be dead.
        let gravity = Vec3::ZERO;
        let size = Curve::constant(1.0);
        let color = ColorGradient::default();
        let speed = Curve::constant(1.0);
        pool.update(0.6, gravity, &size, &color, &speed);
        assert_eq!(pool.alive(), 0);
    }

    #[test]
    fn sort_back_to_front() {
        let mut pool = ParticlePool::new(10);
        // Spawn 3 particles at different distances from origin.
        for z in [1.0f32, 5.0, 3.0] {
            pool.spawn(&SpawnParams {
                position: Vec3::new(0.0, 0.0, z),
                velocity: Vec3::ZERO,
                color: [1.0; 4],
                size: 1.0,
                lifetime: 10.0,
                rotation: 0.0,
                angular_velocity: 0.0,
                custom_data: [0.0; 4],
            });
        }
        pool.sort(SortMode::BackToFront, Vec3::ZERO);
        // Back-to-front: farthest first => z=5, z=3, z=1.
        assert!(pool.positions[0].z > pool.positions[1].z);
        assert!(pool.positions[1].z > pool.positions[2].z);
    }

    #[test]
    fn capacity_limit() {
        let mut pool = ParticlePool::new(3);
        let params = SpawnParams {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            color: [1.0; 4],
            size: 1.0,
            lifetime: 10.0,
            rotation: 0.0,
            angular_velocity: 0.0,
            custom_data: [0.0; 4],
        };
        for _ in 0..10 {
            pool.spawn(&params);
        }
        assert_eq!(pool.alive(), 3, "Should cap at capacity");
    }

    #[test]
    fn aabb_computation() {
        let mut pool = ParticlePool::new(10);
        pool.spawn(&SpawnParams {
            position: Vec3::new(-1.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
            color: [1.0; 4],
            size: 0.5,
            lifetime: 10.0,
            rotation: 0.0,
            angular_velocity: 0.0,
            custom_data: [0.0; 4],
        });
        pool.spawn(&SpawnParams {
            position: Vec3::new(1.0, 2.0, 3.0),
            velocity: Vec3::ZERO,
            color: [1.0; 4],
            size: 0.5,
            lifetime: 10.0,
            rotation: 0.0,
            angular_velocity: 0.0,
            custom_data: [0.0; 4],
        });
        let (min, max) = pool.compute_aabb();
        assert!(min.x <= -1.0);
        assert!(max.x >= 1.0);
        assert!(max.y >= 2.0);
    }
}
