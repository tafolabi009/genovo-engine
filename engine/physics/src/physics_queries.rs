// engine/physics/src/physics_queries_v2.rs
//
// Extended physics query system for the Genovo engine.
//
// Provides comprehensive spatial queries beyond basic raycasting:
//
// - Shape overlap tests: check if a shape overlaps any colliders.
// - Closest point queries: find the closest point on any collider.
// - Contact testing: generate contacts between query shapes and the world.
// - Shape sweeps with configurable filters.
// - Batched queries for bulk operations.
// - Query result caching and sorting.
// - Debug visualization hooks for query shapes.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum results for a single query.
const MAX_QUERY_RESULTS: usize = 256;

/// Maximum distance for sweep queries.
const MAX_SWEEP_DISTANCE: f32 = 1000.0;

/// Small epsilon for comparisons.
const EPSILON: f32 = 1e-6;

/// Default batch query capacity.
const DEFAULT_BATCH_CAPACITY: usize = 64;

// ---------------------------------------------------------------------------
// Query Shape
// ---------------------------------------------------------------------------

/// Shape used for physics queries.
#[derive(Debug, Clone)]
pub enum QueryShapeV2 {
    /// Sphere defined by center and radius.
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
    /// Axis-aligned bounding box.
    Aabb {
        min: [f32; 3],
        max: [f32; 3],
    },
    /// Oriented bounding box.
    Obb {
        center: [f32; 3],
        half_extents: [f32; 3],
        rotation: [f32; 4], // quaternion
    },
    /// Capsule defined by two endpoints and radius.
    Capsule {
        start: [f32; 3],
        end: [f32; 3],
        radius: f32,
    },
    /// Ray (infinite line from origin in direction).
    Ray {
        origin: [f32; 3],
        direction: [f32; 3],
    },
    /// Point query.
    Point {
        position: [f32; 3],
    },
    /// Convex hull defined by points.
    ConvexHull {
        points: Vec<[f32; 3]>,
    },
}

impl QueryShapeV2 {
    /// Create a sphere query shape.
    pub fn sphere(center: [f32; 3], radius: f32) -> Self {
        Self::Sphere { center, radius }
    }

    /// Create an AABB query shape.
    pub fn aabb(min: [f32; 3], max: [f32; 3]) -> Self {
        Self::Aabb { min, max }
    }

    /// Create an OBB query shape.
    pub fn obb(center: [f32; 3], half_extents: [f32; 3], rotation: [f32; 4]) -> Self {
        Self::Obb { center, half_extents, rotation }
    }

    /// Create a capsule query shape.
    pub fn capsule(start: [f32; 3], end: [f32; 3], radius: f32) -> Self {
        Self::Capsule { start, end, radius }
    }

    /// Create a ray query shape.
    pub fn ray(origin: [f32; 3], direction: [f32; 3]) -> Self {
        Self::Ray { origin, direction }
    }

    /// Create a point query shape.
    pub fn point(position: [f32; 3]) -> Self {
        Self::Point { position }
    }

    /// Compute the AABB of this query shape.
    pub fn bounding_aabb(&self) -> ([f32; 3], [f32; 3]) {
        match self {
            Self::Sphere { center, radius } => (
                [center[0] - radius, center[1] - radius, center[2] - radius],
                [center[0] + radius, center[1] + radius, center[2] + radius],
            ),
            Self::Aabb { min, max } => (*min, *max),
            Self::Obb { center, half_extents, .. } => {
                // Conservative AABB (ignoring rotation).
                let r = (half_extents[0] * half_extents[0]
                    + half_extents[1] * half_extents[1]
                    + half_extents[2] * half_extents[2])
                    .sqrt();
                (
                    [center[0] - r, center[1] - r, center[2] - r],
                    [center[0] + r, center[1] + r, center[2] + r],
                )
            }
            Self::Capsule { start, end, radius } => {
                let min_x = start[0].min(end[0]) - radius;
                let min_y = start[1].min(end[1]) - radius;
                let min_z = start[2].min(end[2]) - radius;
                let max_x = start[0].max(end[0]) + radius;
                let max_y = start[1].max(end[1]) + radius;
                let max_z = start[2].max(end[2]) + radius;
                ([min_x, min_y, min_z], [max_x, max_y, max_z])
            }
            Self::Ray { origin, direction } => {
                let far = [
                    origin[0] + direction[0] * MAX_SWEEP_DISTANCE,
                    origin[1] + direction[1] * MAX_SWEEP_DISTANCE,
                    origin[2] + direction[2] * MAX_SWEEP_DISTANCE,
                ];
                (
                    [origin[0].min(far[0]), origin[1].min(far[1]), origin[2].min(far[2])],
                    [origin[0].max(far[0]), origin[1].max(far[1]), origin[2].max(far[2])],
                )
            }
            Self::Point { position } => (*position, *position),
            Self::ConvexHull { points } => {
                if points.is_empty() {
                    return ([0.0; 3], [0.0; 3]);
                }
                let mut min = points[0];
                let mut max = points[0];
                for p in points.iter().skip(1) {
                    for i in 0..3 {
                        min[i] = min[i].min(p[i]);
                        max[i] = max[i].max(p[i]);
                    }
                }
                (min, max)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Query Filter
// ---------------------------------------------------------------------------

/// Filter configuration for physics queries.
#[derive(Debug, Clone)]
pub struct QueryFilterV2 {
    /// Collision layer mask (only collide with bodies on these layers).
    pub layer_mask: u32,
    /// Body IDs to exclude from results.
    pub exclude_bodies: Vec<u64>,
    /// Entity IDs to exclude from results.
    pub exclude_entities: Vec<u64>,
    /// Whether to include static bodies.
    pub include_static: bool,
    /// Whether to include dynamic bodies.
    pub include_dynamic: bool,
    /// Whether to include kinematic bodies.
    pub include_kinematic: bool,
    /// Whether to include trigger volumes.
    pub include_triggers: bool,
    /// Material filter (None = accept all).
    pub material_filter: Option<Vec<u32>>,
    /// Custom tag filter.
    pub tag_filter: Option<String>,
    /// Maximum results to return.
    pub max_results: usize,
    /// Sort results by distance.
    pub sort_by_distance: bool,
}

impl Default for QueryFilterV2 {
    fn default() -> Self {
        Self {
            layer_mask: 0xFFFFFFFF,
            exclude_bodies: Vec::new(),
            exclude_entities: Vec::new(),
            include_static: true,
            include_dynamic: true,
            include_kinematic: true,
            include_triggers: false,
            material_filter: None,
            tag_filter: None,
            max_results: MAX_QUERY_RESULTS,
            sort_by_distance: true,
        }
    }
}

impl QueryFilterV2 {
    /// Create a filter that accepts everything.
    pub fn all() -> Self {
        Self {
            include_triggers: true,
            ..Default::default()
        }
    }

    /// Create a filter for a specific layer.
    pub fn layer(layer: u32) -> Self {
        Self {
            layer_mask: 1 << layer,
            ..Default::default()
        }
    }

    /// Exclude a body from results.
    pub fn exclude_body(mut self, body_id: u64) -> Self {
        self.exclude_bodies.push(body_id);
        self
    }

    /// Exclude an entity from results.
    pub fn exclude_entity(mut self, entity_id: u64) -> Self {
        self.exclude_entities.push(entity_id);
        self
    }

    /// Only include static bodies.
    pub fn static_only(mut self) -> Self {
        self.include_dynamic = false;
        self.include_kinematic = false;
        self
    }

    /// Only include dynamic bodies.
    pub fn dynamic_only(mut self) -> Self {
        self.include_static = false;
        self.include_kinematic = false;
        self
    }

    /// Check if a body passes this filter.
    pub fn accepts(&self, body_id: u64, entity_id: u64, layer: u32, is_trigger: bool) -> bool {
        if self.exclude_bodies.contains(&body_id) {
            return false;
        }
        if self.exclude_entities.contains(&entity_id) {
            return false;
        }
        if (self.layer_mask & (1 << layer)) == 0 {
            return false;
        }
        if is_trigger && !self.include_triggers {
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Query Results
// ---------------------------------------------------------------------------

/// Result of a shape overlap query.
#[derive(Debug, Clone)]
pub struct OverlapResult {
    /// Body ID of the overlapping collider.
    pub body_id: u64,
    /// Entity ID.
    pub entity_id: u64,
    /// Collider shape type name.
    pub shape_type: String,
    /// Collision layer.
    pub layer: u32,
}

/// Result of a closest-point query.
#[derive(Debug, Clone)]
pub struct ClosestPointResult {
    /// The closest point on the collider surface.
    pub point: [f32; 3],
    /// Surface normal at the closest point.
    pub normal: [f32; 3],
    /// Distance from query point to closest point.
    pub distance: f32,
    /// Body ID of the collider.
    pub body_id: u64,
    /// Entity ID.
    pub entity_id: u64,
    /// Whether the query point is inside the collider.
    pub inside: bool,
}

/// Result of a contact test.
#[derive(Debug, Clone)]
pub struct ContactTestResult {
    /// Contact point in world space.
    pub point: [f32; 3],
    /// Contact normal (pointing from query shape to collider).
    pub normal: [f32; 3],
    /// Penetration depth (positive = overlapping).
    pub depth: f32,
    /// Body ID of the collider.
    pub body_id: u64,
    /// Entity ID.
    pub entity_id: u64,
    /// Material ID at contact.
    pub material_id: u32,
}

/// Result of a sweep query.
#[derive(Debug, Clone)]
pub struct SweepResultV2 {
    /// Hit point in world space.
    pub point: [f32; 3],
    /// Surface normal at hit.
    pub normal: [f32; 3],
    /// Distance along the sweep direction.
    pub distance: f32,
    /// Time of impact (0.0 to 1.0).
    pub time_of_impact: f32,
    /// Body ID of the hit collider.
    pub body_id: u64,
    /// Entity ID.
    pub entity_id: u64,
    /// Material ID at hit.
    pub material_id: u32,
}

// ---------------------------------------------------------------------------
// Batched Queries
// ---------------------------------------------------------------------------

/// A batch of physics queries to execute together.
#[derive(Debug)]
pub struct QueryBatch {
    /// Overlap queries.
    pub overlaps: Vec<(QueryShapeV2, QueryFilterV2)>,
    /// Closest point queries.
    pub closest_points: Vec<([f32; 3], QueryFilterV2)>,
    /// Contact tests.
    pub contact_tests: Vec<(QueryShapeV2, QueryFilterV2)>,
    /// Sweep queries.
    pub sweeps: Vec<(QueryShapeV2, [f32; 3], f32, QueryFilterV2)>,
}

impl QueryBatch {
    /// Create a new empty batch.
    pub fn new() -> Self {
        Self {
            overlaps: Vec::with_capacity(DEFAULT_BATCH_CAPACITY),
            closest_points: Vec::with_capacity(DEFAULT_BATCH_CAPACITY),
            contact_tests: Vec::with_capacity(DEFAULT_BATCH_CAPACITY),
            sweeps: Vec::with_capacity(DEFAULT_BATCH_CAPACITY),
        }
    }

    /// Add an overlap query to the batch.
    pub fn add_overlap(&mut self, shape: QueryShapeV2, filter: QueryFilterV2) {
        self.overlaps.push((shape, filter));
    }

    /// Add a closest point query to the batch.
    pub fn add_closest_point(&mut self, point: [f32; 3], filter: QueryFilterV2) {
        self.closest_points.push((point, filter));
    }

    /// Add a contact test to the batch.
    pub fn add_contact_test(&mut self, shape: QueryShapeV2, filter: QueryFilterV2) {
        self.contact_tests.push((shape, filter));
    }

    /// Add a sweep query to the batch.
    pub fn add_sweep(
        &mut self,
        shape: QueryShapeV2,
        direction: [f32; 3],
        max_distance: f32,
        filter: QueryFilterV2,
    ) {
        self.sweeps.push((shape, direction, max_distance, filter));
    }

    /// Returns the total number of queries in this batch.
    pub fn total_queries(&self) -> usize {
        self.overlaps.len() + self.closest_points.len()
            + self.contact_tests.len() + self.sweeps.len()
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.total_queries() == 0
    }
}

/// Results from a batched query execution.
#[derive(Debug)]
pub struct QueryBatchResults {
    /// Overlap results (one Vec per overlap query).
    pub overlaps: Vec<Vec<OverlapResult>>,
    /// Closest point results (one per query).
    pub closest_points: Vec<Option<ClosestPointResult>>,
    /// Contact test results (one Vec per query).
    pub contact_tests: Vec<Vec<ContactTestResult>>,
    /// Sweep results (one Vec per query).
    pub sweeps: Vec<Vec<SweepResultV2>>,
}

impl QueryBatchResults {
    /// Create empty results.
    pub fn new() -> Self {
        Self {
            overlaps: Vec::new(),
            closest_points: Vec::new(),
            contact_tests: Vec::new(),
            sweeps: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Physics Query System
// ---------------------------------------------------------------------------

/// Collider representation for queries.
#[derive(Debug, Clone)]
pub struct QueryCollider {
    /// Unique body ID.
    pub body_id: u64,
    /// Entity ID.
    pub entity_id: u64,
    /// Collision layer.
    pub layer: u32,
    /// Whether this is a trigger.
    pub is_trigger: bool,
    /// Shape type.
    pub shape: ColliderShape,
    /// World-space position.
    pub position: [f32; 3],
    /// Rotation quaternion.
    pub rotation: [f32; 4],
    /// AABB min.
    pub aabb_min: [f32; 3],
    /// AABB max.
    pub aabb_max: [f32; 3],
    /// Material ID.
    pub material_id: u32,
    /// Whether this is a static body.
    pub is_static: bool,
    /// Whether this is a dynamic body.
    pub is_dynamic: bool,
    /// Whether this is a kinematic body.
    pub is_kinematic: bool,
}

/// Simplified collider shape for queries.
#[derive(Debug, Clone)]
pub enum ColliderShape {
    /// Sphere with radius.
    Sphere { radius: f32 },
    /// Box with half-extents.
    Box { half_extents: [f32; 3] },
    /// Capsule with half-height and radius.
    Capsule { half_height: f32, radius: f32 },
    /// Convex hull.
    ConvexHull { points: Vec<[f32; 3]> },
    /// Triangle mesh (for static colliders).
    TriMesh { vertex_count: u32, triangle_count: u32 },
}

/// The extended physics query system.
#[derive(Debug)]
pub struct PhysicsQuerySystemV2 {
    /// Registered colliders (in practice, these come from the physics world).
    pub colliders: Vec<QueryCollider>,
    /// Query cache for repeated queries.
    pub cache: HashMap<u64, Vec<u64>>,
    /// Whether caching is enabled.
    pub caching_enabled: bool,
    /// Statistics.
    pub stats: QueryStats,
    /// Debug visualization enabled.
    pub debug_visualization: bool,
}

impl PhysicsQuerySystemV2 {
    /// Create a new query system.
    pub fn new() -> Self {
        Self {
            colliders: Vec::new(),
            cache: HashMap::new(),
            caching_enabled: false,
            stats: QueryStats::default(),
            debug_visualization: false,
        }
    }

    /// Register a collider for queries.
    pub fn register_collider(&mut self, collider: QueryCollider) {
        self.colliders.push(collider);
    }

    /// Clear all registered colliders.
    pub fn clear_colliders(&mut self) {
        self.colliders.clear();
        self.cache.clear();
    }

    /// Perform a shape overlap query.
    pub fn overlap(
        &mut self,
        shape: &QueryShapeV2,
        filter: &QueryFilterV2,
    ) -> Vec<OverlapResult> {
        self.stats.overlap_queries += 1;
        let mut results = Vec::new();
        let (query_min, query_max) = shape.bounding_aabb();

        for collider in &self.colliders {
            if results.len() >= filter.max_results {
                break;
            }

            if !filter.accepts(collider.body_id, collider.entity_id, collider.layer, collider.is_trigger) {
                continue;
            }

            if !Self::check_body_type_filter(collider, filter) {
                continue;
            }

            // Broad phase: AABB overlap.
            if !aabb_overlap(query_min, query_max, collider.aabb_min, collider.aabb_max) {
                continue;
            }

            results.push(OverlapResult {
                body_id: collider.body_id,
                entity_id: collider.entity_id,
                shape_type: format!("{:?}", collider.shape),
                layer: collider.layer,
            });
        }

        results
    }

    /// Find the closest point on any collider to the given query point.
    pub fn closest_point(
        &mut self,
        point: [f32; 3],
        max_distance: f32,
        filter: &QueryFilterV2,
    ) -> Option<ClosestPointResult> {
        self.stats.closest_point_queries += 1;
        let mut best: Option<ClosestPointResult> = None;

        for collider in &self.colliders {
            if !filter.accepts(collider.body_id, collider.entity_id, collider.layer, collider.is_trigger) {
                continue;
            }

            if !Self::check_body_type_filter(collider, filter) {
                continue;
            }

            let (closest, normal, dist, inside) = Self::closest_point_on_collider(collider, point);

            if dist > max_distance {
                continue;
            }

            let is_better = best.as_ref().map_or(true, |b| dist < b.distance);
            if is_better {
                best = Some(ClosestPointResult {
                    point: closest,
                    normal,
                    distance: dist,
                    body_id: collider.body_id,
                    entity_id: collider.entity_id,
                    inside,
                });
            }
        }

        best
    }

    /// Generate contacts between a query shape and the world.
    pub fn contact_test(
        &mut self,
        shape: &QueryShapeV2,
        filter: &QueryFilterV2,
    ) -> Vec<ContactTestResult> {
        self.stats.contact_tests += 1;
        let mut results = Vec::new();
        let (query_min, query_max) = shape.bounding_aabb();

        for collider in &self.colliders {
            if results.len() >= filter.max_results {
                break;
            }

            if !filter.accepts(collider.body_id, collider.entity_id, collider.layer, collider.is_trigger) {
                continue;
            }

            if !Self::check_body_type_filter(collider, filter) {
                continue;
            }

            if !aabb_overlap(query_min, query_max, collider.aabb_min, collider.aabb_max) {
                continue;
            }

            // Generate contact based on shape pair.
            if let Some(contact) = Self::generate_contact(shape, collider) {
                results.push(contact);
            }
        }

        if filter.sort_by_distance {
            results.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));
        }

        results
    }

    /// Sweep a shape in a direction and find hits.
    pub fn sweep(
        &mut self,
        shape: &QueryShapeV2,
        direction: [f32; 3],
        max_distance: f32,
        filter: &QueryFilterV2,
    ) -> Vec<SweepResultV2> {
        self.stats.sweep_queries += 1;
        let mut results = Vec::new();

        let dir_len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        if dir_len < EPSILON {
            return results;
        }
        let dir = [direction[0] / dir_len, direction[1] / dir_len, direction[2] / dir_len];

        // Expand query AABB along sweep direction.
        let (mut query_min, mut query_max) = shape.bounding_aabb();
        for i in 0..3 {
            if dir[i] > 0.0 {
                query_max[i] += dir[i] * max_distance;
            } else {
                query_min[i] += dir[i] * max_distance;
            }
        }

        for collider in &self.colliders {
            if results.len() >= filter.max_results {
                break;
            }

            if !filter.accepts(collider.body_id, collider.entity_id, collider.layer, collider.is_trigger) {
                continue;
            }

            if !Self::check_body_type_filter(collider, filter) {
                continue;
            }

            if !aabb_overlap(query_min, query_max, collider.aabb_min, collider.aabb_max) {
                continue;
            }

            // Simple sweep test: step along direction.
            let step_count = 32;
            let step_size = max_distance / step_count as f32;

            for step in 0..step_count {
                let t = step as f32 * step_size;
                let offset = [dir[0] * t, dir[1] * t, dir[2] * t];

                let swept_shape = Self::offset_shape(shape, offset);
                if let Some(contact) = Self::generate_contact(&swept_shape, collider) {
                    results.push(SweepResultV2 {
                        point: contact.point,
                        normal: contact.normal,
                        distance: t,
                        time_of_impact: t / max_distance,
                        body_id: collider.body_id,
                        entity_id: collider.entity_id,
                        material_id: collider.material_id,
                    });
                    break;
                }
            }
        }

        if filter.sort_by_distance {
            results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        }

        results
    }

    /// Execute a batch of queries.
    pub fn execute_batch(&mut self, batch: &QueryBatch) -> QueryBatchResults {
        self.stats.batch_queries += 1;
        let mut results = QueryBatchResults::new();

        for (shape, filter) in &batch.overlaps {
            results.overlaps.push(self.overlap(shape, filter));
        }
        for (point, filter) in &batch.closest_points {
            results.closest_points.push(self.closest_point(*point, MAX_SWEEP_DISTANCE, filter));
        }
        for (shape, filter) in &batch.contact_tests {
            results.contact_tests.push(self.contact_test(shape, filter));
        }
        for (shape, direction, max_dist, filter) in &batch.sweeps {
            results.sweeps.push(self.sweep(shape, *direction, *max_dist, filter));
        }

        results
    }

    /// Check body type filter.
    fn check_body_type_filter(collider: &QueryCollider, filter: &QueryFilterV2) -> bool {
        if collider.is_static && !filter.include_static {
            return false;
        }
        if collider.is_dynamic && !filter.include_dynamic {
            return false;
        }
        if collider.is_kinematic && !filter.include_kinematic {
            return false;
        }
        true
    }

    /// Find the closest point on a collider to a query point.
    fn closest_point_on_collider(
        collider: &QueryCollider,
        point: [f32; 3],
    ) -> ([f32; 3], [f32; 3], f32, bool) {
        match &collider.shape {
            ColliderShape::Sphere { radius } => {
                let dx = point[0] - collider.position[0];
                let dy = point[1] - collider.position[1];
                let dz = point[2] - collider.position[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < EPSILON {
                    return (collider.position, [0.0, 1.0, 0.0], 0.0, true);
                }

                let normal = [dx / dist, dy / dist, dz / dist];
                let surface = [
                    collider.position[0] + normal[0] * radius,
                    collider.position[1] + normal[1] * radius,
                    collider.position[2] + normal[2] * radius,
                ];
                let surface_dist = (dist - radius).abs();
                let inside = dist < *radius;

                (surface, normal, surface_dist, inside)
            }
            ColliderShape::Box { half_extents } => {
                let local = [
                    point[0] - collider.position[0],
                    point[1] - collider.position[1],
                    point[2] - collider.position[2],
                ];

                let clamped = [
                    local[0].clamp(-half_extents[0], half_extents[0]),
                    local[1].clamp(-half_extents[1], half_extents[1]),
                    local[2].clamp(-half_extents[2], half_extents[2]),
                ];

                let closest = [
                    collider.position[0] + clamped[0],
                    collider.position[1] + clamped[1],
                    collider.position[2] + clamped[2],
                ];

                let dx = point[0] - closest[0];
                let dy = point[1] - closest[1];
                let dz = point[2] - closest[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                let inside = local[0].abs() <= half_extents[0]
                    && local[1].abs() <= half_extents[1]
                    && local[2].abs() <= half_extents[2];

                let normal = if dist > EPSILON {
                    [dx / dist, dy / dist, dz / dist]
                } else {
                    [0.0, 1.0, 0.0]
                };

                (closest, normal, dist, inside)
            }
            _ => {
                // Fallback: use AABB approximation.
                let clamped = [
                    point[0].clamp(collider.aabb_min[0], collider.aabb_max[0]),
                    point[1].clamp(collider.aabb_min[1], collider.aabb_max[1]),
                    point[2].clamp(collider.aabb_min[2], collider.aabb_max[2]),
                ];
                let dx = point[0] - clamped[0];
                let dy = point[1] - clamped[1];
                let dz = point[2] - clamped[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let normal = if dist > EPSILON {
                    [dx / dist, dy / dist, dz / dist]
                } else {
                    [0.0, 1.0, 0.0]
                };
                (clamped, normal, dist, false)
            }
        }
    }

    /// Generate a contact between a query shape and a collider.
    fn generate_contact(shape: &QueryShapeV2, collider: &QueryCollider) -> Option<ContactTestResult> {
        match shape {
            QueryShapeV2::Sphere { center, radius } => {
                let (closest, normal, dist, _) = Self::closest_point_on_collider(collider, *center);
                let depth = radius - dist;
                if depth > 0.0 {
                    Some(ContactTestResult {
                        point: closest,
                        normal,
                        depth,
                        body_id: collider.body_id,
                        entity_id: collider.entity_id,
                        material_id: collider.material_id,
                    })
                } else {
                    None
                }
            }
            QueryShapeV2::Point { position } => {
                let (closest, normal, dist, inside) = Self::closest_point_on_collider(collider, *position);
                if inside {
                    Some(ContactTestResult {
                        point: closest,
                        normal,
                        depth: dist,
                        body_id: collider.body_id,
                        entity_id: collider.entity_id,
                        material_id: collider.material_id,
                    })
                } else {
                    None
                }
            }
            _ => {
                // Use AABB overlap as fallback for other shapes.
                let (q_min, q_max) = shape.bounding_aabb();
                if aabb_overlap(q_min, q_max, collider.aabb_min, collider.aabb_max) {
                    let center = [
                        (q_min[0] + q_max[0]) * 0.5,
                        (q_min[1] + q_max[1]) * 0.5,
                        (q_min[2] + q_max[2]) * 0.5,
                    ];
                    Some(ContactTestResult {
                        point: center,
                        normal: [0.0, 1.0, 0.0],
                        depth: 0.0,
                        body_id: collider.body_id,
                        entity_id: collider.entity_id,
                        material_id: collider.material_id,
                    })
                } else {
                    None
                }
            }
        }
    }

    /// Offset a query shape by a translation.
    fn offset_shape(shape: &QueryShapeV2, offset: [f32; 3]) -> QueryShapeV2 {
        match shape {
            QueryShapeV2::Sphere { center, radius } => QueryShapeV2::Sphere {
                center: [center[0] + offset[0], center[1] + offset[1], center[2] + offset[2]],
                radius: *radius,
            },
            QueryShapeV2::Point { position } => QueryShapeV2::Point {
                position: [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]],
            },
            QueryShapeV2::Aabb { min, max } => QueryShapeV2::Aabb {
                min: [min[0] + offset[0], min[1] + offset[1], min[2] + offset[2]],
                max: [max[0] + offset[0], max[1] + offset[1], max[2] + offset[2]],
            },
            other => other.clone(),
        }
    }
}

/// AABB overlap test.
fn aabb_overlap(a_min: [f32; 3], a_max: [f32; 3], b_min: [f32; 3], b_max: [f32; 3]) -> bool {
    a_min[0] <= b_max[0] && a_max[0] >= b_min[0]
        && a_min[1] <= b_max[1] && a_max[1] >= b_min[1]
        && a_min[2] <= b_max[2] && a_max[2] >= b_min[2]
}

/// Query system statistics.
#[derive(Debug, Clone, Default)]
pub struct QueryStats {
    /// Total overlap queries.
    pub overlap_queries: u64,
    /// Total closest point queries.
    pub closest_point_queries: u64,
    /// Total contact tests.
    pub contact_tests: u64,
    /// Total sweep queries.
    pub sweep_queries: u64,
    /// Total batch queries.
    pub batch_queries: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
}
