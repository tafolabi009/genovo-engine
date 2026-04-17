// engine/scene/src/scene_queries.rs
//
// Scene queries for the Genovo engine.
//
// Provides methods to search and filter entities within a scene:
//
// - **Find by name/tag/component** -- Look up entities by metadata.
// - **Spatial queries** -- Find entities in radius, box, or frustum.
// - **Raycast** -- Cast rays through the scene to find intersections.
// - **Nearest entity** -- Find the closest entity matching criteria.
// - **Entity iteration with filters** -- Efficient filtered iteration.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryEntityId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentTypeId(pub u32);

impl fmt::Display for QueryEntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Entity metadata
// ---------------------------------------------------------------------------

/// Metadata stored per entity for querying.
#[derive(Debug, Clone)]
pub struct EntityMetadata {
    pub id: QueryEntityId,
    pub name: String,
    pub tags: Vec<String>,
    pub components: Vec<ComponentTypeId>,
    pub position: [f32; 3],
    pub bounds_radius: f32,
    pub layer: u32,
    pub active: bool,
    pub static_entity: bool,
}

impl EntityMetadata {
    pub fn new(id: QueryEntityId) -> Self {
        Self {
            id,
            name: String::new(),
            tags: Vec::new(),
            components: Vec::new(),
            position: [0.0; 3],
            bounds_radius: 1.0,
            layer: 0,
            active: true,
            static_entity: false,
        }
    }

    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    pub fn has_component(&self, component: ComponentTypeId) -> bool {
        self.components.contains(&component)
    }

    pub fn distance_to(&self, point: [f32; 3]) -> f32 {
        let dx = self.position[0] - point[0];
        let dy = self.position[1] - point[1];
        let dz = self.position[2] - point[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn distance_sq_to(&self, point: [f32; 3]) -> f32 {
        let dx = self.position[0] - point[0];
        let dy = self.position[1] - point[1];
        let dz = self.position[2] - point[2];
        dx * dx + dy * dy + dz * dz
    }
}

// ---------------------------------------------------------------------------
// Query filter
// ---------------------------------------------------------------------------

/// Filter criteria for entity queries.
#[derive(Debug, Clone, Default)]
pub struct QueryFilter {
    pub name_contains: Option<String>,
    pub exact_name: Option<String>,
    pub required_tags: Vec<String>,
    pub excluded_tags: Vec<String>,
    pub required_components: Vec<ComponentTypeId>,
    pub excluded_components: Vec<ComponentTypeId>,
    pub layer_mask: Option<u32>,
    pub active_only: bool,
    pub static_only: Option<bool>,
    pub max_distance: Option<f32>,
    pub origin: Option<[f32; 3]>,
}

impl QueryFilter {
    pub fn new() -> Self {
        Self {
            active_only: true,
            ..Default::default()
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.exact_name = Some(name.to_string());
        self
    }

    pub fn with_name_contains(mut self, substring: &str) -> Self {
        self.name_contains = Some(substring.to_string());
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.required_tags.push(tag.to_string());
        self
    }

    pub fn without_tag(mut self, tag: &str) -> Self {
        self.excluded_tags.push(tag.to_string());
        self
    }

    pub fn with_component(mut self, component: ComponentTypeId) -> Self {
        self.required_components.push(component);
        self
    }

    pub fn without_component(mut self, component: ComponentTypeId) -> Self {
        self.excluded_components.push(component);
        self
    }

    pub fn in_radius(mut self, origin: [f32; 3], radius: f32) -> Self {
        self.origin = Some(origin);
        self.max_distance = Some(radius);
        self
    }

    pub fn on_layer(mut self, layer: u32) -> Self {
        self.layer_mask = Some(1 << layer);
        self
    }

    pub fn include_inactive(mut self) -> Self {
        self.active_only = false;
        self
    }

    /// Check if an entity matches this filter.
    pub fn matches(&self, entity: &EntityMetadata) -> bool {
        if self.active_only && !entity.active {
            return false;
        }
        if let Some(ref name) = self.exact_name {
            if entity.name != *name {
                return false;
            }
        }
        if let Some(ref substr) = self.name_contains {
            if !entity.name.contains(substr.as_str()) {
                return false;
            }
        }
        for tag in &self.required_tags {
            if !entity.has_tag(tag) {
                return false;
            }
        }
        for tag in &self.excluded_tags {
            if entity.has_tag(tag) {
                return false;
            }
        }
        for comp in &self.required_components {
            if !entity.has_component(*comp) {
                return false;
            }
        }
        for comp in &self.excluded_components {
            if entity.has_component(*comp) {
                return false;
            }
        }
        if let Some(mask) = self.layer_mask {
            if (1 << entity.layer) & mask == 0 {
                return false;
            }
        }
        if let Some(ref static_flag) = self.static_only {
            if entity.static_entity != *static_flag {
                return false;
            }
        }
        if let (Some(origin), Some(max_dist)) = (self.origin, self.max_distance) {
            if entity.distance_to(origin) > max_dist {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Raycast
// ---------------------------------------------------------------------------

/// A ray for scene raycasting.
#[derive(Debug, Clone, Copy)]
pub struct SceneRay {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
    pub max_distance: f32,
}

impl SceneRay {
    pub fn new(origin: [f32; 3], direction: [f32; 3], max_distance: f32) -> Self {
        // Normalize direction.
        let len = (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]).sqrt();
        let dir = if len > 1e-6 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, 0.0, -1.0]
        };
        Self { origin, direction: dir, max_distance }
    }

    /// Point along the ray at parameter t.
    pub fn point_at(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + self.direction[0] * t,
            self.origin[1] + self.direction[1] * t,
            self.origin[2] + self.direction[2] * t,
        ]
    }
}

/// Result of a scene raycast.
#[derive(Debug, Clone)]
pub struct SceneRaycastHit {
    pub entity: QueryEntityId,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
}

// ---------------------------------------------------------------------------
// AABB for box queries
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct QueryBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl QueryBox {
    pub fn from_center_size(center: [f32; 3], half_extents: [f32; 3]) -> Self {
        Self {
            min: [center[0] - half_extents[0], center[1] - half_extents[1], center[2] - half_extents[2]],
            max: [center[0] + half_extents[0], center[1] + half_extents[1], center[2] + half_extents[2]],
        }
    }

    pub fn contains_point(&self, p: [f32; 3]) -> bool {
        p[0] >= self.min[0] && p[0] <= self.max[0]
            && p[1] >= self.min[1] && p[1] <= self.max[1]
            && p[2] >= self.min[2] && p[2] <= self.max[2]
    }

    pub fn intersects_sphere(&self, center: [f32; 3], radius: f32) -> bool {
        let mut dist_sq = 0.0_f32;
        for i in 0..3 {
            let v = center[i];
            if v < self.min[i] {
                dist_sq += (self.min[i] - v) * (self.min[i] - v);
            } else if v > self.max[i] {
                dist_sq += (v - self.max[i]) * (v - self.max[i]);
            }
        }
        dist_sq <= radius * radius
    }
}

// ---------------------------------------------------------------------------
// Scene query system
// ---------------------------------------------------------------------------

/// Scene query system for finding entities.
pub struct SceneQuerySystem {
    entities: HashMap<QueryEntityId, EntityMetadata>,
    name_index: HashMap<String, Vec<QueryEntityId>>,
    tag_index: HashMap<String, Vec<QueryEntityId>>,
}

impl SceneQuerySystem {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            name_index: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }

    /// Register an entity.
    pub fn register(&mut self, metadata: EntityMetadata) {
        let id = metadata.id;
        if !metadata.name.is_empty() {
            self.name_index.entry(metadata.name.clone()).or_default().push(id);
        }
        for tag in &metadata.tags {
            self.tag_index.entry(tag.clone()).or_default().push(id);
        }
        self.entities.insert(id, metadata);
    }

    /// Unregister an entity.
    pub fn unregister(&mut self, id: QueryEntityId) {
        if let Some(meta) = self.entities.remove(&id) {
            if let Some(ids) = self.name_index.get_mut(&meta.name) {
                ids.retain(|&e| e != id);
            }
            for tag in &meta.tags {
                if let Some(ids) = self.tag_index.get_mut(tag) {
                    ids.retain(|&e| e != id);
                }
            }
        }
    }

    /// Update an entity's position.
    pub fn update_position(&mut self, id: QueryEntityId, position: [f32; 3]) {
        if let Some(meta) = self.entities.get_mut(&id) {
            meta.position = position;
        }
    }

    /// Find entities by exact name.
    pub fn find_by_name(&self, name: &str) -> Vec<QueryEntityId> {
        self.name_index.get(name).cloned().unwrap_or_default()
    }

    /// Find entities with a specific tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<QueryEntityId> {
        self.tag_index.get(tag).cloned().unwrap_or_default()
    }

    /// Find entities matching a filter.
    pub fn query(&self, filter: &QueryFilter) -> Vec<QueryEntityId> {
        self.entities.values()
            .filter(|e| filter.matches(e))
            .map(|e| e.id)
            .collect()
    }

    /// Find entities in a sphere.
    pub fn find_in_radius(&self, center: [f32; 3], radius: f32) -> Vec<(QueryEntityId, f32)> {
        let r_sq = radius * radius;
        let mut results = Vec::new();
        for meta in self.entities.values() {
            if !meta.active { continue; }
            let dist_sq = meta.distance_sq_to(center);
            if dist_sq <= r_sq {
                results.push((meta.id, dist_sq.sqrt()));
            }
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Find entities in a box.
    pub fn find_in_box(&self, query_box: &QueryBox) -> Vec<QueryEntityId> {
        self.entities.values()
            .filter(|e| e.active && query_box.intersects_sphere(e.position, e.bounds_radius))
            .map(|e| e.id)
            .collect()
    }

    /// Find the nearest entity matching a filter.
    pub fn find_nearest(&self, point: [f32; 3], filter: &QueryFilter) -> Option<(QueryEntityId, f32)> {
        let mut best: Option<(QueryEntityId, f32)> = None;
        for meta in self.entities.values() {
            if !filter.matches(meta) { continue; }
            let dist = meta.distance_to(point);
            match best {
                None => best = Some((meta.id, dist)),
                Some((_, best_dist)) if dist < best_dist => best = Some((meta.id, dist)),
                _ => {}
            }
        }
        best
    }

    /// Raycast against entity bounding spheres.
    pub fn raycast(&self, ray: &SceneRay, filter: &QueryFilter) -> Vec<SceneRaycastHit> {
        let mut hits = Vec::new();

        for meta in self.entities.values() {
            if !filter.matches(meta) { continue; }

            // Ray-sphere intersection.
            let oc = [
                ray.origin[0] - meta.position[0],
                ray.origin[1] - meta.position[1],
                ray.origin[2] - meta.position[2],
            ];
            let d = ray.direction;
            let a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let b = 2.0 * (oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2]);
            let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2]
                - meta.bounds_radius * meta.bounds_radius;
            let discriminant = b * b - 4.0 * a * c;

            if discriminant >= 0.0 {
                let t = (-b - discriminant.sqrt()) / (2.0 * a);
                if t >= 0.0 && t <= ray.max_distance {
                    let point = ray.point_at(t);
                    let normal = [
                        (point[0] - meta.position[0]) / meta.bounds_radius,
                        (point[1] - meta.position[1]) / meta.bounds_radius,
                        (point[2] - meta.position[2]) / meta.bounds_radius,
                    ];
                    hits.push(SceneRaycastHit {
                        entity: meta.id,
                        distance: t,
                        point,
                        normal,
                    });
                }
            }
        }

        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }

    /// Get entity count.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Get entity metadata.
    pub fn get(&self, id: QueryEntityId) -> Option<&EntityMetadata> {
        self.entities.get(&id)
    }
}

impl Default for SceneQuerySystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: u32, name: &str, pos: [f32; 3], tags: &[&str]) -> EntityMetadata {
        let mut meta = EntityMetadata::new(QueryEntityId(id));
        meta.name = name.to_string();
        meta.position = pos;
        meta.tags = tags.iter().map(|s| s.to_string()).collect();
        meta
    }

    #[test]
    fn test_find_by_name() {
        let mut sys = SceneQuerySystem::new();
        sys.register(make_entity(0, "player", [0.0; 3], &[]));
        sys.register(make_entity(1, "enemy", [5.0, 0.0, 0.0], &[]));

        assert_eq!(sys.find_by_name("player").len(), 1);
        assert_eq!(sys.find_by_name("none").len(), 0);
    }

    #[test]
    fn test_find_by_tag() {
        let mut sys = SceneQuerySystem::new();
        sys.register(make_entity(0, "a", [0.0; 3], &["enemy", "boss"]));
        sys.register(make_entity(1, "b", [0.0; 3], &["enemy"]));
        sys.register(make_entity(2, "c", [0.0; 3], &["ally"]));

        assert_eq!(sys.find_by_tag("enemy").len(), 2);
        assert_eq!(sys.find_by_tag("boss").len(), 1);
    }

    #[test]
    fn test_radius_query() {
        let mut sys = SceneQuerySystem::new();
        sys.register(make_entity(0, "a", [0.0, 0.0, 0.0], &[]));
        sys.register(make_entity(1, "b", [3.0, 0.0, 0.0], &[]));
        sys.register(make_entity(2, "c", [100.0, 0.0, 0.0], &[]));

        let results = sys.find_in_radius([0.0, 0.0, 0.0], 5.0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_nearest() {
        let mut sys = SceneQuerySystem::new();
        sys.register(make_entity(0, "a", [10.0, 0.0, 0.0], &[]));
        sys.register(make_entity(1, "b", [3.0, 0.0, 0.0], &[]));

        let (nearest, _) = sys.find_nearest([0.0; 3], &QueryFilter::new()).unwrap();
        assert_eq!(nearest, QueryEntityId(1));
    }

    #[test]
    fn test_raycast() {
        let mut sys = SceneQuerySystem::new();
        let mut e = make_entity(0, "target", [5.0, 0.0, 0.0], &[]);
        e.bounds_radius = 1.0;
        sys.register(e);

        let ray = SceneRay::new([0.0; 3], [1.0, 0.0, 0.0], 100.0);
        let hits = sys.raycast(&ray, &QueryFilter::new());
        assert!(!hits.is_empty());
        assert!((hits[0].distance - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_filter() {
        let filter = QueryFilter::new().with_tag("enemy").without_tag("boss");
        let e1 = make_entity(0, "a", [0.0; 3], &["enemy"]);
        let e2 = make_entity(1, "b", [0.0; 3], &["enemy", "boss"]);
        assert!(filter.matches(&e1));
        assert!(!filter.matches(&e2));
    }
}
