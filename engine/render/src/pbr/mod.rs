// engine/render/src/pbr/mod.rs
//
// Physically-Based Rendering (PBR) material system. Implements the metallic-
// roughness workflow with Cook-Torrance specular BRDF, image-based lighting
// support, and dynamic WGSL shader generation.

pub mod material;
pub mod brdf;
pub mod texture_slots;
pub mod shader_gen;

pub use material::{
    AlphaMode, Material, MaterialHandle, MaterialInstance, MaterialLibrary, StandardMaterial,
};
pub use brdf::{
    BrdfLut, CookTorranceBrdf, DiffuseBrdf, FresnelFunction, GeometryFunction,
    NormalDistributionFunction,
};
pub use texture_slots::{MaterialTextureSet, TextureBinding, TextureSlot, UvTransform};
pub use shader_gen::{generate_pbr_shader, MaterialFeatures};
