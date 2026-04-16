//! Genovo Engine - Asset Management
//!
//! Provides asset loading, caching, hot-reloading, import/cook pipelines,
//! format-specific loaders, and an asset database for tracking metadata
//! and dependencies.
//!
//! # Modules
//!
//! - [`formats`] — Basic format loaders (BMP, WAV, OBJ, text, bytes)
//! - [`image`] — Extended image formats (TGA, HDR, DDS) and image operations
//! - [`gltf`] — glTF 2.0 / GLB loader with animations and skins
//! - [`font`] — TrueType font parser with SDF rasterisation
//! - [`material`] — PBR material asset definitions
//! - [`scene_format`] — Scene file format with entity hierarchies

pub mod database;
pub mod font;
pub mod formats;
pub mod gltf;
pub mod image;
pub mod loader;
pub mod material;
pub mod pipeline;
pub mod scene_format;

pub use database::{AssetDatabase, AssetMeta};
pub use loader::{AssetError, AssetHandle, AssetId, AssetLoader, AssetPath, AssetServer, LoadState};
pub use pipeline::{AssetCookPipeline, AssetImporter, AssetManifest, AssetProcessor, StreamingManager};
pub use formats::{
    BytesLoader, MeshData, ObjLoader, TextLoader, TextureData, TextureFormat, TextureLoader,
    WavLoader, AudioData,
};
pub use image::{
    DdsLoader, HdrLoader, Image, ImageFormat, ResizeFilter, TgaLoader,
    convert_format, flip_horizontal, flip_vertical, generate_mipmaps, height_to_normal, resize,
};
pub use gltf::{
    AnimationChannel, AnimationClip, AnimationProperty, AnimationValues, GltfDocument,
    GltfLoader, GltfMaterial, GltfMesh, GltfMeshData, GltfNode, GltfScene, Interpolation,
    Skeleton,
};
pub use font::{FontData, FontLoader, GlyphOutline, GlyphPoint, HMetric, PositionedGlyph, SdfGlyph};
pub use material::{MaterialAlphaMode, MaterialAsset, MaterialLoader};
pub use scene_format::{
    ComponentData, SceneAsset, SceneEntity, SceneLoader, SceneSettings, TransformData,
};
