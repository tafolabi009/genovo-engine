// engine/render/src/virtual_geometry/mod.rs
//
// Nanite-style virtual geometry system for the Genovo engine. Provides mesh
// clustering, hierarchical LOD construction, BVH acceleration, streaming,
// and mesh simplification via Quadric Error Metrics (QEM).
//
// # Architecture
//
// A mesh is first partitioned into clusters of ~128 triangles using spatial
// locality. These clusters form a DAG where parent clusters are simplified
// versions of their children. At runtime the system selects which clusters
// to render based on screen-space error, streams them in/out of GPU memory,
// and feeds them into the GPU-driven rendering pipeline.

pub mod cluster;
pub mod bvh;
pub mod streaming;
pub mod simplification;

// Re-exports for convenience.
pub use cluster::{
    MeshCluster, ClusterDAG, ClusterGroup, ClusterBounds, build_cluster_dag,
};
pub use bvh::{BVH, BVHNode, Hit, build_bvh};
pub use streaming::{VirtualGeometryStreaming, StreamRequest, StreamPriority, PageAllocation};
pub use simplification::{
    QuadricErrorMetric, EdgeCollapse, SimplifiedMesh, simplify, simplify_preserve_boundaries,
};
