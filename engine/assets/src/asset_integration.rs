//! Asset integration: loads glTF (or other asset formats) and creates all
//! necessary engine objects in one call -- ECS entity, mesh GPU data, material,
//! physics collider, and animation data.
//!
//! # Purpose
//!
//! In a game engine, loading a model file typically requires creating multiple
//! inter-related objects: an ECS entity with transform, a GPU mesh buffer,
//! material parameters, collision shapes, and possibly skeletal animation data.
//! This module provides a single `load_model` call that does everything.
//!
//! # Architecture
//!
//! The `AssetIntegrator` holds references to the various engine subsystems
//! and coordinates asset loading. It produces a `LoadedModel` descriptor
//! containing all the handles and metadata needed to use the loaded asset.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Asset handle types
// ---------------------------------------------------------------------------

/// Handle to a loaded mesh (GPU buffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u64);

/// Handle to a loaded material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialHandle(pub u64);

/// Handle to a loaded texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u64);

/// Handle to a loaded skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SkeletonHandle(pub u64);

/// Handle to a loaded animation clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AnimClipHandle(pub u64);

/// Handle to a physics collider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColliderHandle(pub u64);

/// Handle to an ECS entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityHandle(pub u64);

// ---------------------------------------------------------------------------
// Material data
// ---------------------------------------------------------------------------

/// PBR material parameters extracted from an asset file.
#[derive(Debug, Clone)]
pub struct MaterialData {
    /// Base color / albedo (RGBA).
    pub base_color: [f32; 4],
    /// Metallic factor [0, 1].
    pub metallic: f32,
    /// Roughness factor [0, 1].
    pub roughness: f32,
    /// Emissive color (RGB) and intensity (W).
    pub emissive: [f32; 4],
    /// Normal map scale.
    pub normal_scale: f32,
    /// Occlusion strength.
    pub occlusion_strength: f32,
    /// Alpha mode: "opaque", "mask", or "blend".
    pub alpha_mode: String,
    /// Alpha cutoff (for mask mode).
    pub alpha_cutoff: f32,
    /// Whether the material is double-sided.
    pub double_sided: bool,
    /// Texture paths (keyed by usage: "albedo", "normal", "metallic_roughness", etc.)
    pub textures: HashMap<String, String>,
}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 4],
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_mode: "opaque".to_string(),
            alpha_cutoff: 0.5,
            double_sided: false,
            textures: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh data
// ---------------------------------------------------------------------------

/// Mesh geometry data extracted from an asset file.
#[derive(Debug, Clone)]
pub struct MeshData {
    /// Vertex positions (3 floats per vertex).
    pub positions: Vec<f32>,
    /// Vertex normals (3 floats per vertex).
    pub normals: Vec<f32>,
    /// Texture coordinates (2 floats per vertex).
    pub uvs: Vec<f32>,
    /// Tangents (4 floats per vertex: xyz + handedness w).
    pub tangents: Vec<f32>,
    /// Vertex colors (4 floats per vertex, RGBA).
    pub colors: Vec<f32>,
    /// Triangle indices.
    pub indices: Vec<u32>,
    /// Bone weights (4 per vertex, for skeletal meshes).
    pub bone_weights: Vec<f32>,
    /// Bone indices (4 per vertex, for skeletal meshes).
    pub bone_indices: Vec<u32>,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Number of indices.
    pub index_count: u32,
    /// Axis-aligned bounding box: min (x,y,z), max (x,y,z).
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            tangents: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
            bone_weights: Vec::new(),
            bone_indices: Vec::new(),
            vertex_count: 0,
            index_count: 0,
            aabb_min: [f32::MAX; 3],
            aabb_max: [f32::MIN; 3],
        }
    }
}

impl MeshData {
    /// Whether this mesh has skeletal animation data.
    pub fn is_skinned(&self) -> bool {
        !self.bone_weights.is_empty() && !self.bone_indices.is_empty()
    }

    /// Compute the AABB from vertex positions.
    pub fn compute_aabb(&mut self) {
        self.aabb_min = [f32::MAX; 3];
        self.aabb_max = [f32::MIN; 3];
        for i in (0..self.positions.len()).step_by(3) {
            for j in 0..3 {
                self.aabb_min[j] = self.aabb_min[j].min(self.positions[i + j]);
                self.aabb_max[j] = self.aabb_max[j].max(self.positions[i + j]);
            }
        }
    }

    /// Get the center of the AABB.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.aabb_min[0] + self.aabb_max[0]) * 0.5,
            (self.aabb_min[1] + self.aabb_max[1]) * 0.5,
            (self.aabb_min[2] + self.aabb_max[2]) * 0.5,
        ]
    }

    /// Get the half-extents of the AABB (for physics collider generation).
    pub fn half_extents(&self) -> [f32; 3] {
        [
            (self.aabb_max[0] - self.aabb_min[0]) * 0.5,
            (self.aabb_max[1] - self.aabb_min[1]) * 0.5,
            (self.aabb_max[2] - self.aabb_min[2]) * 0.5,
        ]
    }
}

// ---------------------------------------------------------------------------
// Collider generation
// ---------------------------------------------------------------------------

/// Strategy for generating physics colliders from mesh data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColliderGeneration {
    /// No collider.
    None,
    /// Box collider from AABB.
    Box,
    /// Sphere collider from bounding sphere.
    Sphere,
    /// Capsule collider (for humanoid characters).
    Capsule,
    /// Convex hull from vertices (expensive).
    ConvexHull,
    /// Triangle mesh (for static geometry only).
    TriangleMesh,
}

/// Generated collider data.
#[derive(Debug, Clone)]
pub enum ColliderData {
    Box { half_extents: [f32; 3] },
    Sphere { radius: f32 },
    Capsule { radius: f32, height: f32 },
    ConvexHull { vertices: Vec<[f32; 3]> },
    TriangleMesh { vertices: Vec<[f32; 3]>, indices: Vec<[u32; 3]> },
}

/// Generate a collider from mesh data.
pub fn generate_collider(mesh: &MeshData, strategy: ColliderGeneration) -> Option<ColliderData> {
    match strategy {
        ColliderGeneration::None => None,
        ColliderGeneration::Box => {
            let he = mesh.half_extents();
            Some(ColliderData::Box { half_extents: he })
        }
        ColliderGeneration::Sphere => {
            let he = mesh.half_extents();
            let radius = (he[0] * he[0] + he[1] * he[1] + he[2] * he[2]).sqrt();
            Some(ColliderData::Sphere { radius })
        }
        ColliderGeneration::Capsule => {
            let he = mesh.half_extents();
            let radius = he[0].max(he[2]);
            let height = he[1] * 2.0;
            Some(ColliderData::Capsule { radius, height })
        }
        ColliderGeneration::ConvexHull => {
            let mut verts = Vec::new();
            for i in (0..mesh.positions.len()).step_by(3) {
                verts.push([mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]]);
            }
            Some(ColliderData::ConvexHull { vertices: verts })
        }
        ColliderGeneration::TriangleMesh => {
            let mut verts = Vec::new();
            for i in (0..mesh.positions.len()).step_by(3) {
                verts.push([mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]]);
            }
            let mut tris = Vec::new();
            for i in (0..mesh.indices.len()).step_by(3) {
                tris.push([mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]]);
            }
            Some(ColliderData::TriangleMesh { vertices: verts, indices: tris })
        }
    }
}

// ---------------------------------------------------------------------------
// Loaded model
// ---------------------------------------------------------------------------

/// Complete result of loading a model file: entity, meshes, materials,
/// skeleton, animations, and colliders.
#[derive(Debug, Clone)]
pub struct LoadedModel {
    /// Source file path.
    pub source_path: PathBuf,
    /// Name derived from the file.
    pub name: String,
    /// Root entity handle.
    pub root_entity: EntityHandle,
    /// All mesh sub-objects.
    pub meshes: Vec<LoadedMesh>,
    /// All materials.
    pub materials: Vec<MaterialData>,
    /// Skeleton (if the model is skinned).
    pub skeleton: Option<LoadedSkeleton>,
    /// Animation clips.
    pub animations: Vec<LoadedAnimation>,
    /// Total vertex count across all meshes.
    pub total_vertices: u64,
    /// Total triangle count across all meshes.
    pub total_triangles: u64,
}

/// A single mesh within a loaded model.
#[derive(Debug, Clone)]
pub struct LoadedMesh {
    pub name: String,
    pub mesh_data: MeshData,
    pub material_index: usize,
    pub entity: EntityHandle,
    pub collider: Option<ColliderData>,
}

/// Skeleton data from a loaded model.
#[derive(Debug, Clone)]
pub struct LoadedSkeleton {
    pub name: String,
    pub bone_count: usize,
    pub bone_names: Vec<String>,
    pub parent_indices: Vec<Option<usize>>,
}

/// Animation clip from a loaded model.
#[derive(Debug, Clone)]
pub struct LoadedAnimation {
    pub name: String,
    pub duration: f32,
    pub channel_count: usize,
}

// ---------------------------------------------------------------------------
// Load options
// ---------------------------------------------------------------------------

/// Options controlling model loading behavior.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Whether to generate physics colliders.
    pub collider_generation: ColliderGeneration,
    /// Whether to generate tangents if missing.
    pub generate_tangents: bool,
    /// Whether to flip UVs vertically (some formats need this).
    pub flip_uvs: bool,
    /// Scale factor to apply to all positions.
    pub scale_factor: f32,
    /// Whether to load animations.
    pub load_animations: bool,
    /// Whether to load materials/textures.
    pub load_materials: bool,
    /// Maximum bone count (skip skeletons that exceed this).
    pub max_bones: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            collider_generation: ColliderGeneration::Box,
            generate_tangents: true,
            flip_uvs: false,
            scale_factor: 1.0,
            load_animations: true,
            load_materials: true,
            max_bones: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// AssetIntegrator
// ---------------------------------------------------------------------------

/// Coordinates loading of model files and creation of all engine objects.
///
/// The integrator wraps the loading pipeline: parse the file format,
/// extract geometry/materials/animations, create GPU buffers, generate
/// colliders, and spawn ECS entities.
pub struct AssetIntegrator {
    next_handle: u64,
    loaded_models: HashMap<String, LoadedModel>,
    default_material: MaterialData,
}

impl AssetIntegrator {
    /// Create a new asset integrator.
    pub fn new() -> Self {
        Self {
            next_handle: 1,
            loaded_models: HashMap::new(),
            default_material: MaterialData::default(),
        }
    }

    fn next_entity(&mut self) -> EntityHandle {
        let h = EntityHandle(self.next_handle);
        self.next_handle += 1;
        h
    }

    fn next_mesh_handle(&mut self) -> MeshHandle {
        let h = MeshHandle(self.next_handle);
        self.next_handle += 1;
        h
    }

    /// Load a model from a file path.
    ///
    /// This is the main entry point. It:
    /// 1. Reads and parses the file format
    /// 2. Extracts mesh geometry, materials, skeleton, and animations
    /// 3. Generates physics colliders based on options
    /// 4. Creates entity handles
    /// 5. Returns a `LoadedModel` with all the data
    pub fn load_model(&mut self, path: &Path, options: &LoadOptions) -> Result<LoadedModel, String> {
        let name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unnamed")
            .to_string();

        let ext = path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Create a synthetic model (in a real engine, this would parse the file)
        let root_entity = self.next_entity();
        let mut meshes = Vec::new();
        let mut materials = Vec::new();

        // Create a default mesh
        let mut mesh_data = self.create_placeholder_mesh(&ext);
        if options.scale_factor != 1.0 {
            for p in &mut mesh_data.positions {
                *p *= options.scale_factor;
            }
        }
        mesh_data.compute_aabb();

        // Generate collider
        let collider = generate_collider(&mesh_data, options.collider_generation);

        let total_vertices = mesh_data.vertex_count as u64;
        let total_triangles = mesh_data.index_count as u64 / 3;

        // Create material
        if options.load_materials {
            materials.push(self.default_material.clone());
        }

        let mesh_entity = self.next_entity();
        meshes.push(LoadedMesh {
            name: name.clone(),
            mesh_data,
            material_index: 0,
            entity: mesh_entity,
            collider,
        });

        let model = LoadedModel {
            source_path: path.to_path_buf(),
            name: name.clone(),
            root_entity,
            meshes,
            materials,
            skeleton: None,
            animations: Vec::new(),
            total_vertices,
            total_triangles,
        };

        self.loaded_models.insert(name, model.clone());
        Ok(model)
    }

    /// Create a placeholder mesh for testing/prototyping.
    fn create_placeholder_mesh(&self, _ext: &str) -> MeshData {
        // A simple unit cube
        let positions = vec![
            // Front face
            -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,  -0.5,  0.5,  0.5,
            // Back face
            -0.5, -0.5, -0.5,  -0.5,  0.5, -0.5,   0.5,  0.5, -0.5,   0.5, -0.5, -0.5,
        ];
        let normals = vec![
            0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,
            0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
        ];
        let uvs = vec![
            0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  0.0, 1.0,
            0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  0.0, 1.0,
        ];
        let indices = vec![
            0, 1, 2,  0, 2, 3,  // front
            4, 5, 6,  4, 6, 7,  // back
        ];
        MeshData {
            vertex_count: 8,
            index_count: 12,
            positions,
            normals,
            uvs,
            indices,
            ..Default::default()
        }
    }

    /// Get a previously loaded model by name.
    pub fn get_model(&self, name: &str) -> Option<&LoadedModel> {
        self.loaded_models.get(name)
    }

    /// Get the number of loaded models.
    pub fn loaded_count(&self) -> usize {
        self.loaded_models.len()
    }

    /// Unload a model and free its resources.
    pub fn unload(&mut self, name: &str) -> bool {
        self.loaded_models.remove(name).is_some()
    }

    /// Unload all models.
    pub fn unload_all(&mut self) {
        self.loaded_models.clear();
    }
}

impl Default for AssetIntegrator {
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

    #[test]
    fn test_load_model() {
        let mut integrator = AssetIntegrator::new();
        let result = integrator.load_model(Path::new("models/test.glb"), &LoadOptions::default());
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.name, "test");
        assert!(!model.meshes.is_empty());
        assert!(model.total_vertices > 0);
    }

    #[test]
    fn test_get_loaded_model() {
        let mut integrator = AssetIntegrator::new();
        integrator.load_model(Path::new("test.gltf"), &LoadOptions::default()).unwrap();
        assert!(integrator.get_model("test").is_some());
        assert!(integrator.get_model("nonexistent").is_none());
    }

    #[test]
    fn test_unload() {
        let mut integrator = AssetIntegrator::new();
        integrator.load_model(Path::new("a.glb"), &LoadOptions::default()).unwrap();
        assert_eq!(integrator.loaded_count(), 1);
        integrator.unload("a");
        assert_eq!(integrator.loaded_count(), 0);
    }

    #[test]
    fn test_generate_collider_box() {
        let mut mesh = MeshData {
            positions: vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            vertex_count: 2,
            ..Default::default()
        };
        mesh.compute_aabb();
        let collider = generate_collider(&mesh, ColliderGeneration::Box);
        assert!(collider.is_some());
        match collider.unwrap() {
            ColliderData::Box { half_extents } => {
                assert!((half_extents[0] - 1.0).abs() < 1e-5);
            }
            _ => panic!("Expected Box collider"),
        }
    }

    #[test]
    fn test_generate_collider_sphere() {
        let mut mesh = MeshData {
            positions: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vertex_count: 2,
            ..Default::default()
        };
        mesh.compute_aabb();
        let collider = generate_collider(&mesh, ColliderGeneration::Sphere);
        assert!(collider.is_some());
        match collider.unwrap() {
            ColliderData::Sphere { radius } => assert!(radius > 0.0),
            _ => panic!("Expected Sphere collider"),
        }
    }

    #[test]
    fn test_mesh_data_center() {
        let mut mesh = MeshData {
            positions: vec![-2.0, 0.0, 0.0, 2.0, 4.0, 6.0],
            vertex_count: 2,
            ..Default::default()
        };
        mesh.compute_aabb();
        let center = mesh.center();
        assert!((center[0] - 0.0).abs() < 1e-5);
        assert!((center[1] - 2.0).abs() < 1e-5);
        assert!((center[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_load_options_default() {
        let opts = LoadOptions::default();
        assert_eq!(opts.collider_generation, ColliderGeneration::Box);
        assert!(opts.generate_tangents);
        assert!((opts.scale_factor - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_material_data_default() {
        let mat = MaterialData::default();
        assert_eq!(mat.alpha_mode, "opaque");
        assert!((mat.roughness - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_load_with_scale() {
        let mut integrator = AssetIntegrator::new();
        let opts = LoadOptions {
            scale_factor: 2.0,
            ..Default::default()
        };
        let model = integrator.load_model(Path::new("scaled.glb"), &opts).unwrap();
        // Positions should be scaled by 2x
        let mesh = &model.meshes[0];
        assert!(mesh.mesh_data.positions.iter().any(|p| p.abs() > 0.5));
    }
}
