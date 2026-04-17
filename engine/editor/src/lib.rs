//! # Genovo Editor
//!
//! The editor module provides a full-featured game editor built on top of the
//! Genovo engine runtime. It includes a 3D viewport with gizmo manipulation,
//! a property inspector with undo/redo, an asset browser, a scene hierarchy
//! panel, project management, a visual node graph editor, an animation curve
//! editor, a node-graph-based material editor, terrain editing tools, and
//! extended scene view features (bookmarks, snapping, measurements, multi-
//! viewport layouts, statistics overlays).

pub mod asset_browser;
pub mod curve_editor;
pub mod hierarchy;
pub mod inspector;
pub mod material_editor;
pub mod node_graph;
pub mod project;
pub mod scene_view;
pub mod selection;
pub mod snapping;
pub mod terrain_editor;
pub mod undo_system;
pub mod viewport;

// Re-export primary types for convenience.
pub use asset_browser::{AssetBrowser, AssetImportDialog, AssetThumbnailGenerator, DragDropPayload};
pub use curve_editor::{AnimationCurve, CurveEditor, Keyframe, TangentMode};
pub use hierarchy::HierarchyPanel;
pub use inspector::{EditorCommand, Inspectable, Inspector, PropertyWidget, UndoRedoStack};
pub use material_editor::MaterialEditor;
pub use node_graph::{GraphNode, NodeGraph, NodePin, PinDataType, Value};
pub use project::{Project, ProjectSettings};
pub use scene_view::{
    BookmarkManager, CameraBookmark, MeasurementTool, MultiViewportState, SceneStatistics,
    SceneViewState, SnapSettings, ViewportLayout,
};
pub use selection::{
    CameraProjection, EntityHandle, Outliner, ScreenRect, Selection, SelectionHistory,
    SelectionMode, SelectionTransform,
};
pub use snapping::{
    apply_rotation_snap, apply_scale_snap, apply_snap, find_alignment_guides, snap_angle_degrees,
    snap_angle_radians, snap_rotation, snap_scale, snap_to_edge, snap_to_grid, snap_to_surface,
    snap_to_vertex, snap_uniform_scale, Edge, SnapGuide, SnapGuideType, SurfaceHit, Triangle,
};
pub use terrain_editor::{BrushSettings, BrushType, TerrainEditor};
pub use undo_system::{
    ChangeComponentOp, CompoundOp, CompoundOpBuilder, DespawnEntityOp, EntityId, MoveEntityOp,
    Operation, OperationResult, OperationType, ReparentOp, RotateEntityOp, ScaleEntityOp,
    SerializedComponentData, SerializedEntityData, SpawnEntityOp, UndoEvent, UndoStack,
};
pub use viewport::{
    CameraMode, EditorViewport, GizmoMode, GizmoRenderer, PickRay, SelectionManager,
    ViewportCamera,
};
