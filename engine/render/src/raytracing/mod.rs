// engine/render/src/raytracing/mod.rs
//
// Software ray tracing system for the Genovo engine. Provides a Lumen-style
// hybrid rendering approach with BVH acceleration, screen-space probes,
// GGX importance-sampled reflections, and full path tracing.
//
// # Modules
//
// - [`bvh_tracer`] -- Two-level BVH (TLAS/BLAS) with SAH construction and
//   Moller-Trumbore ray-triangle intersection.
// - [`screen_probes`] -- Lumen-style screen-space probe system with SH
//   radiance accumulation, temporal and spatial filtering.
// - [`reflections`] -- Ray-traced reflections with GGX importance sampling,
//   spatial/temporal denoising, and SSR fallback.
// - [`gi_tracer`] -- Global illumination via multi-bounce tracing and
//   unbiased path tracing with Russian roulette and MIS.

pub mod bvh_tracer;
pub mod screen_probes;
pub mod reflections;
pub mod gi_tracer;

// Re-exports for convenience.
pub use bvh_tracer::{
    AABB, BLAS, BVHAccelerationStructure, BVHNode, BVHStats, HitInfo, MeshInstance,
    Ray, TLAS, Triangle, TriangleHit, AnyHitQuery,
    build_bvh_sah, intersect_ray_triangle,
};
pub use screen_probes::{
    ScreenProbe, ScreenProbeGrid, ScreenProbeSettings, Surfel, SurfaceCache,
    hemisphere_sample_cosine_weighted, hemisphere_sample_uniform,
    PROBE_TILE_SIZE, DEFAULT_RAYS_PER_PROBE, MAX_RAYS_PER_PROBE,
};
pub use reflections::{
    ReflectionBuffer, ReflectionSettings, ReflectionTracer, SSRResult, SSRTracer,
    reflect, sample_ggx_reflection,
};
pub use gi_tracer::{
    GIMode, GIResultBuffer, GISettings, GITracer, IrradianceField,
    MaterialResponse, balance_heuristic, power_heuristic,
};
