//! # genovo-debug
//!
//! Debug, profiling, and development tools for the Genovo game engine.
//!
//! This crate provides:
//!
//! - **CPU Profiler** — hierarchical frame profiler with Chrome trace export
//! - **GPU Profiler** — GPU timing, pipeline stats, and memory tracking
//! - **Developer Console** — in-game command-line with variables and scripting
//! - **Debug Renderer** — immediate-mode wireframe shape drawing
//! - **Memory Profiler** — per-category allocation tracking and leak detection
//! - **Analytics** — event/metric tracking, sessions, heatmaps, and data export

pub mod analytics;
pub mod console;
pub mod debug_render;
pub mod gizmos;
pub mod memory_profiler;
pub mod profiler;

// Re-exports for ergonomic access.
pub use analytics::{
    Analytics, AnalyticsEvent, HeatmapData, MetricSample, MetricStats, MetricTimeSeries,
    PerfMetrics, Session, SessionEndReason,
};
pub use console::{Console, ConsoleCommand, ConsoleVar, ConsoleVarValue};
pub use debug_render::{Color, DebugRenderer, DebugVertex, DepthMode};
pub use gizmos::{
    ColliderShape, GizmoBatch, GizmoColor, GizmoDepthMode, GizmoLabel, GizmoVertex, Gizmos,
    GizmosComponent,
};
pub use memory_profiler::{MemoryCategory, MemoryProfiler, MemoryReport};
pub use profiler::{
    ProfileFrame, ProfileNode, ProfileReport, ProfileScope, Profiler, ScopeGuard,
};
pub use profiler::gpu_profiler::{GpuProfiler, GpuProfileReport};
