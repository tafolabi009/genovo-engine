// engine/render/src/lighting/mod.rs
//
// Light system for the Genovo renderer. Provides typed light sources,
// frustum-based light culling, and clustered-forward light assignment.

pub mod light_types;
pub mod light_culling;
pub mod light_clustering;

pub use light_types::{
    AreaLight, AreaLightShape, DirectionalLight, Light, LightData, LightManager, LightType,
    PointLight, SpotLight, MAX_LIGHTS,
};
pub use light_culling::{cull_lights, Frustum, FrustumPlane};
pub use light_clustering::{build_cluster_grid, ClusterGrid, ClusterSettings};
