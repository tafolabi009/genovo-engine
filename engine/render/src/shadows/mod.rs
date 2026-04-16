// engine/render/src/shadows/mod.rs
//
// Shadow mapping system for the Genovo renderer. Supports cascaded shadow
// maps for directional lights, cube maps for point lights, standard maps
// for spot lights, PCF filtering, and variance shadow maps.

pub mod shadow_map;
pub mod cascade;
pub mod pcf;
pub mod vsm;

pub use shadow_map::{
    ShadowMap, ShadowMapAtlas, ShadowMapEntry, ShadowSettings,
};
pub use cascade::{
    CascadeSplit, CascadeSplitScheme, CascadedShadowMap,
    compute_cascade_matrices,
};
pub use pcf::{
    PcfKernel, PcfSettings, PcssSettings,
};
pub use vsm::{
    VsmSettings, VsmFilterPass,
};
