//! Material asset definitions.
//!
//! Provides a serialisable PBR material description that can be stored as
//! a standalone `.mat` / `.material` file alongside the scene assets.
//! Materials reference textures by path and encode common PBR parameters
//! (metallic-roughness workflow).

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::loader::{AssetError, AssetLoader};

// =========================================================================
// Material types
// =========================================================================

/// A PBR material asset.
///
/// Materials can be serialised to / from JSON and reference textures by
/// their asset paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialAsset {
    /// Human-readable material name.
    pub name: String,

    /// Shading model (e.g. `"standard"`, `"unlit"`, `"subsurface"`).
    #[serde(default = "default_shading_model")]
    pub shading_model: String,

    /// Base color / albedo parameters.
    #[serde(default)]
    pub base_color: ColorProperty,

    /// Metallic value (0.0 = dielectric, 1.0 = metal).
    #[serde(default = "default_zero")]
    pub metallic: f32,

    /// Roughness value (0.0 = mirror, 1.0 = fully rough).
    #[serde(default = "default_half")]
    pub roughness: f32,

    /// Path to the metallic-roughness texture (G = roughness, B = metallic).
    #[serde(default)]
    pub metallic_roughness_texture: Option<String>,

    /// Normal map parameters.
    #[serde(default)]
    pub normal_map: Option<NormalMapProperty>,

    /// Ambient occlusion texture path.
    #[serde(default)]
    pub occlusion_texture: Option<String>,

    /// Occlusion strength (0.0 to 1.0).
    #[serde(default = "default_one")]
    pub occlusion_strength: f32,

    /// Emissive parameters.
    #[serde(default)]
    pub emissive: EmissiveProperty,

    /// Alpha blending mode.
    #[serde(default)]
    pub alpha_mode: MaterialAlphaMode,

    /// Alpha cutoff threshold (used when `alpha_mode` is `Mask`).
    #[serde(default = "default_half")]
    pub alpha_cutoff: f32,

    /// Whether the material should be rendered double-sided.
    #[serde(default)]
    pub double_sided: bool,

    /// Additional key-value parameters for custom shaders.
    #[serde(default)]
    pub custom_params: HashMap<String, ParamValue>,
}

/// The base color / albedo property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorProperty {
    /// RGBA factor applied to the base color texture (or used directly if no texture).
    #[serde(default = "default_white")]
    pub factor: [f32; 4],
    /// Path to the base color texture.
    #[serde(default)]
    pub texture: Option<String>,
    /// Texture coordinate set index (usually 0).
    #[serde(default)]
    pub tex_coord: u32,
}

impl Default for ColorProperty {
    fn default() -> Self {
        Self {
            factor: [1.0, 1.0, 1.0, 1.0],
            texture: None,
            tex_coord: 0,
        }
    }
}

/// Normal map configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalMapProperty {
    /// Path to the normal map texture.
    pub texture: String,
    /// Normal map strength (1.0 = standard).
    #[serde(default = "default_one")]
    pub strength: f32,
    /// Texture coordinate set index.
    #[serde(default)]
    pub tex_coord: u32,
}

/// Emissive light property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissiveProperty {
    /// RGB emissive factor.
    #[serde(default)]
    pub factor: [f32; 3],
    /// Path to emissive texture.
    #[serde(default)]
    pub texture: Option<String>,
    /// Emissive intensity multiplier.
    #[serde(default = "default_one")]
    pub intensity: f32,
}

impl Default for EmissiveProperty {
    fn default() -> Self {
        Self {
            factor: [0.0, 0.0, 0.0],
            texture: None,
            intensity: 1.0,
        }
    }
}

/// Alpha blending mode for a material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialAlphaMode {
    /// Fully opaque.
    Opaque,
    /// Binary transparency with a cutoff threshold.
    Mask,
    /// Smooth alpha blending.
    Blend,
}

impl Default for MaterialAlphaMode {
    fn default() -> Self {
        Self::Opaque
    }
}

/// A typed parameter value for custom shader properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    /// A floating-point scalar.
    Float(f32),
    /// An integer scalar.
    Int(i32),
    /// A boolean flag.
    Bool(bool),
    /// A string (e.g. texture path or enum name).
    String(String),
    /// A 2-component vector.
    Vec2([f32; 2]),
    /// A 3-component vector.
    Vec3([f32; 3]),
    /// A 4-component vector.
    Vec4([f32; 4]),
}

// =========================================================================
// Default value helpers
// =========================================================================

fn default_shading_model() -> String {
    "standard".to_owned()
}
fn default_white() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}
fn default_zero() -> f32 {
    0.0
}
fn default_half() -> f32 {
    0.5
}
fn default_one() -> f32 {
    1.0
}

// =========================================================================
// MaterialAsset implementation
// =========================================================================

impl MaterialAsset {
    /// Creates a new default PBR material with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            shading_model: "standard".to_owned(),
            base_color: ColorProperty::default(),
            metallic: 0.0,
            roughness: 0.5,
            metallic_roughness_texture: None,
            normal_map: None,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive: EmissiveProperty::default(),
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            custom_params: HashMap::new(),
        }
    }

    /// Serialise the material to a JSON string.
    pub fn to_json(&self) -> Result<String, AssetError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| AssetError::Other(format!("Material serialisation failed: {e}")))
    }

    /// Deserialise a material from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, AssetError> {
        serde_json::from_str(json)
            .map_err(|e| AssetError::Parse(format!("Material parse error: {e}")))
    }

    /// Returns a list of all texture paths referenced by this material.
    pub fn texture_paths(&self) -> Vec<&str> {
        let mut paths = Vec::new();
        if let Some(ref tex) = self.base_color.texture {
            paths.push(tex.as_str());
        }
        if let Some(ref tex) = self.metallic_roughness_texture {
            paths.push(tex.as_str());
        }
        if let Some(ref nm) = self.normal_map {
            paths.push(nm.texture.as_str());
        }
        if let Some(ref tex) = self.occlusion_texture {
            paths.push(tex.as_str());
        }
        if let Some(ref tex) = self.emissive.texture {
            paths.push(tex.as_str());
        }
        paths
    }

    /// Sets a custom float parameter.
    pub fn set_float(&mut self, key: &str, value: f32) {
        self.custom_params.insert(key.to_owned(), ParamValue::Float(value));
    }

    /// Sets a custom vec3 parameter.
    pub fn set_vec3(&mut self, key: &str, value: [f32; 3]) {
        self.custom_params.insert(key.to_owned(), ParamValue::Vec3(value));
    }

    /// Gets a custom float parameter.
    pub fn get_float(&self, key: &str) -> Option<f32> {
        match self.custom_params.get(key) {
            Some(ParamValue::Float(f)) => Some(*f),
            _ => None,
        }
    }

    /// Returns `true` if this material uses transparency.
    pub fn is_transparent(&self) -> bool {
        self.alpha_mode != MaterialAlphaMode::Opaque
    }
}

// =========================================================================
// MaterialLoader
// =========================================================================

/// Loads `.mat` / `.material` files (JSON-serialised [`MaterialAsset`]).
pub struct MaterialLoader;

impl AssetLoader for MaterialLoader {
    type Asset = MaterialAsset;

    fn extensions(&self) -> &[&str] {
        &["mat", "material"]
    }

    fn load(&self, _path: &Path, bytes: &[u8]) -> Result<MaterialAsset, AssetError> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| AssetError::Parse(format!("Material file not UTF-8: {e}")))?;
        MaterialAsset::from_json(text)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_default() {
        let mat = MaterialAsset::new("test");
        assert_eq!(mat.name, "test");
        assert_eq!(mat.shading_model, "standard");
        assert_eq!(mat.metallic, 0.0);
        assert_eq!(mat.roughness, 0.5);
        assert_eq!(mat.alpha_mode, MaterialAlphaMode::Opaque);
        assert!(!mat.double_sided);
        assert!(!mat.is_transparent());
    }

    #[test]
    fn test_material_json_round_trip() {
        let mut mat = MaterialAsset::new("gold");
        mat.metallic = 1.0;
        mat.roughness = 0.3;
        mat.base_color.factor = [1.0, 0.8, 0.2, 1.0];
        mat.base_color.texture = Some("textures/gold_albedo.png".to_owned());
        mat.normal_map = Some(NormalMapProperty {
            texture: "textures/gold_normal.png".to_owned(),
            strength: 1.0,
            tex_coord: 0,
        });
        mat.emissive.factor = [0.1, 0.05, 0.0];
        mat.double_sided = true;

        let json = mat.to_json().unwrap();
        let restored = MaterialAsset::from_json(&json).unwrap();

        assert_eq!(restored.name, "gold");
        assert!((restored.metallic - 1.0).abs() < 0.001);
        assert!((restored.roughness - 0.3).abs() < 0.001);
        assert!((restored.base_color.factor[1] - 0.8).abs() < 0.001);
        assert_eq!(
            restored.base_color.texture.as_deref(),
            Some("textures/gold_albedo.png")
        );
        assert!(restored.normal_map.is_some());
        assert!(restored.double_sided);
    }

    #[test]
    fn test_material_texture_paths() {
        let mut mat = MaterialAsset::new("full");
        mat.base_color.texture = Some("albedo.png".to_owned());
        mat.metallic_roughness_texture = Some("mr.png".to_owned());
        mat.normal_map = Some(NormalMapProperty {
            texture: "normal.png".to_owned(),
            strength: 1.0,
            tex_coord: 0,
        });
        mat.occlusion_texture = Some("ao.png".to_owned());
        mat.emissive.texture = Some("emissive.png".to_owned());

        let paths = mat.texture_paths();
        assert_eq!(paths.len(), 5);
        assert!(paths.contains(&"albedo.png"));
        assert!(paths.contains(&"normal.png"));
    }

    #[test]
    fn test_material_custom_params() {
        let mut mat = MaterialAsset::new("custom");
        mat.set_float("tiling", 2.5);
        mat.set_vec3("wind_direction", [1.0, 0.0, 0.0]);

        assert_eq!(mat.get_float("tiling"), Some(2.5));
        assert_eq!(mat.get_float("nonexistent"), None);
    }

    #[test]
    fn test_material_alpha_modes() {
        let mut mat = MaterialAsset::new("alpha_test");
        assert!(!mat.is_transparent());

        mat.alpha_mode = MaterialAlphaMode::Mask;
        assert!(mat.is_transparent());

        mat.alpha_mode = MaterialAlphaMode::Blend;
        assert!(mat.is_transparent());
    }

    #[test]
    fn test_material_loader_extensions() {
        let loader = MaterialLoader;
        assert_eq!(loader.extensions(), &["mat", "material"]);
    }

    #[test]
    fn test_material_loader_load() {
        let json = r#"{"name":"test_mat","metallic":0.5,"roughness":0.7}"#;
        let loader = MaterialLoader;
        let mat = loader.load(Path::new("test.mat"), json.as_bytes()).unwrap();
        assert_eq!(mat.name, "test_mat");
        assert!((mat.metallic - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_material_loader_invalid_json() {
        let loader = MaterialLoader;
        assert!(loader.load(Path::new("bad.mat"), b"not json{{{").is_err());
    }

    #[test]
    fn test_param_value_variants() {
        let f = ParamValue::Float(1.5);
        let i = ParamValue::Int(42);
        let b = ParamValue::Bool(true);
        let s = ParamValue::String("hello".to_owned());
        let v2 = ParamValue::Vec2([1.0, 2.0]);
        let v3 = ParamValue::Vec3([1.0, 2.0, 3.0]);
        let v4 = ParamValue::Vec4([1.0, 2.0, 3.0, 4.0]);

        // Just verify they all serialise without error
        let json = serde_json::to_string(&f).unwrap();
        assert!(json.contains("1.5"));
        let _ = serde_json::to_string(&i).unwrap();
        let _ = serde_json::to_string(&b).unwrap();
        let _ = serde_json::to_string(&s).unwrap();
        let _ = serde_json::to_string(&v2).unwrap();
        let _ = serde_json::to_string(&v3).unwrap();
        let _ = serde_json::to_string(&v4).unwrap();
    }

    #[test]
    fn test_color_property_default() {
        let cp = ColorProperty::default();
        assert_eq!(cp.factor, [1.0, 1.0, 1.0, 1.0]);
        assert!(cp.texture.is_none());
        assert_eq!(cp.tex_coord, 0);
    }

    #[test]
    fn test_emissive_default() {
        let ep = EmissiveProperty::default();
        assert_eq!(ep.factor, [0.0, 0.0, 0.0]);
        assert!(ep.texture.is_none());
        assert_eq!(ep.intensity, 1.0);
    }
}
