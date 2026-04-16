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
pub mod terrain_editor;
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
pub use terrain_editor::{BrushSettings, BrushType, TerrainEditor};
pub use viewport::{
    CameraMode, EditorViewport, GizmoMode, GizmoRenderer, PickRay, SelectionManager,
    ViewportCamera,
};
