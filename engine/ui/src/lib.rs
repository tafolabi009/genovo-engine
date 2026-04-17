//! # Genovo UI
//!
//! A complete immediate-mode-inspired retained UI framework for the Genovo game
//! engine. Provides a layout engine, a rich widget library, text shaping and
//! rendering, a styling/theming system, UI animations, and backend-agnostic draw
//! command generation.
//!
//! # Architecture
//!
//! ```text
//!  Widgets  -->  UITree  -->  LayoutEngine  -->  DrawList  -->  UIRenderer
//!     ^             |              |                               (backend)
//!     |          UIEvent      StyleSheet
//!     +-- user code / scripting
//! ```
//!
//! All coordinates are in logical pixels. The renderer backend is responsible
//! for DPI scaling when submitting GPU commands.

pub mod animation;
pub mod brush_system;
pub mod core;
pub mod data_binding;
pub mod dock_system;
pub mod drag_drop;
pub mod editor_widgets;
pub mod gpu_renderer;
pub mod layout;
pub mod render_commands;
pub mod retained_widgets;
pub mod rich_text;
pub mod slate_accessibility;
pub mod slate_complex_widgets;
pub mod slate_debug;
pub mod slate_input;
pub mod slate_multiwindow;
pub mod slate_notifications;
pub mod slate_animation;
pub mod slate_layout;
pub mod slate_performance;
pub mod slate_platform;
pub mod slate_property_editor;
pub mod slate_text;
pub mod slate_widgets;
pub mod style;
pub mod text;
pub mod ui_animation;
pub mod ui_framework;
pub mod widgets;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use crate::core::{
    Anchor, Margin, Padding, UIContext, UIEvent, UIEventHandler, UIId, UINode, UITree,
};
pub use animation::{
    AnimationGroup, AnimationSequence, EasingFunction, SpringAnimation, UIAnimation,
};
pub use layout::{
    Constraint, FlexLayout, GridLayout, LayoutAlign, LayoutDirection, LayoutEngine,
    ScrollLayout, Size, StackLayout,
};
pub use render_commands::{DrawCommand, DrawList, UIRenderer};
pub use style::{PseudoState, Style, StyleSheet, Theme, ThemeManager, Transition};
pub use text::{Font, FontAtlas, FontLibrary, GlyphRun, ShapedText, TextLayout, TextMeasurement};
pub use widgets::*;
pub use data_binding::{
    BindingContext, BindingConverter, BindingDirection, BindingError, BindingExpression,
    BindingId, BindingMode, BindingPath, BindingScope, BindingValidator, ChangeNotification,
    CollectionBinding, CollectionChange, CollectionUpdateCommand, ObservableCollection,
    PropertyAccessor, PropertyId, ValidationResult,
};
pub use drag_drop::{
    DragDropEvent, DragDropManager, DragPayload, DragPayloadType, DragPhase, DragRect,
    DragSession, DragSessionId, DragSource, DragSourceId, DragVisual, DropFeedback,
    DropTarget, DropTargetId, DropValidation, DropValidator, SnapSlot,
};
pub use rich_text::{
    FadeEffect, GradientDirection, GradientParams, ImageAlignment, InlineImage, OutlineParams,
    PulseEffect, RainbowEffect, RichTextColor, RichTextDocument, RichTextElement,
    RichTextParser, RichTextSpan, ShakeEffect, ShadowParams, TextAlignment, TextEffect,
    TextStyle, TypewriterAnimator, TypewriterEffect, WaveEffect,
};
pub use gpu_renderer::UIGpuRenderer;
pub use ui_framework::{UI, UIInputState, UIStyle, WidgetId};
pub use dock_system::{
    DockArea, DockEvent, DockLayout, DockNode, DockNodeId, DockState, DockStyle, DockTab,
    DockTabId, DockTarget, FloatingWindow, SplitDirection,
};
pub use retained_widgets::{
    ButtonWidget, EventReply, InvalidateReason, LabelWidget, MouseEvent as WidgetMouseEvent,
    KeyEvent as WidgetKeyEvent, PaintCanvas, PanelWidget, StackLayoutWidget, Widget,
    WidgetEvent, WidgetTree, WidgetId as RetainedWidgetId,
};
pub use brush_system::{
    Brush, BrushMargin, BrushTextStyle, ButtonStyle, CheckboxStyle, CornerRadius, GradientDir,
    InputStyle, PanelStyle, ScrollbarStyle, SliderStyle, StyleManager, StyleSet, TabStyle,
    TextOutline, TextShadow, TilingMode, TooltipStyle, TreeViewStyle,
};
pub use ui_animation::{
    AnimatedValue, AnimationController, AnimPresets, CurveHandle, CurveSequence, EasingKind,
    Lerp, PlaybackState, TransformAnimation, WidgetSpring, WidgetSpring2D,
};
pub use slate_widgets::{
    BorderWidget, Brush as SlateBrush, CheckBox, CheckState, ComboBox,
    EditableTextBox, EventReply as SlateEventReply, ExpandableArea, HAlign, Hyperlink,
    ImageWidget, InlineEditableTextBlock, MultiLineEditableText, Orientation,
    ProgressBar, ProgressMode, RichTextBlock, RichTextSpan as SlateRichTextSpan,
    SearchBox, Separator, SlateButton, SlateWidget, Slider, Spacer, SpinBox,
    StretchMode, TextBlock, TextDecoration, TextWrapping, Throbber, ThrobberStyle, VAlign,
};
pub use slate_complex_widgets::{
    BreadcrumbTrail, ColorPicker, ColumnHeader, CurveEditor, CurveKeyframe,
    FlattenedTreeRow, GradientColorStop, GradientEditor, GraphEditor, GraphNode,
    GraphPin, GraphWire, HeaderRow, ListView, ListViewItem, MenuAnchor, MenuItem,
    MenuPlacement, NotificationAnchor, NotificationItem, NotificationList,
    NotificationState, PinDirection, PinType, SelectionMode, SelectionState,
    SortDirection, TangentMode, TileView, TileViewItem, Tooltip, TreeView, TreeViewNode,
};
pub use slate_input::{
    ActiveDrag, CommandExecution, CommandList, CursorManager, CursorShape,
    DropTargetConfig, DropValidationResult, FocusCause, FocusScope, FocusSystem,
    HighPrecisionMouseMode, HitTestConfig, HitTestManager, HitTestResult,
    HitTestShape, InputChord, KeyEvent as SlateKeyEvent, MouseButtonMask,
    MouseCaptureManager, MouseEvent as SlateMouseEvent, ScrollEvent, ScrollPhase,
    SlateDragDropEvent, SlateDragDropManager, SlateDragPhase, SlateInputProcessor,
    TextInputEvent, UICommand, DragPayloadData, DragVisual as SlateDragVisual,
};
pub use slate_text::{
    AtlasGlyph, ClipboardIntegration, DecoratorToken, HighlightLanguage,
    HighlightToken, PositionedGlyph, ShapeTextAlign, SlateBoldDecorator,
    SlateColorDecorator, SlateFontAtlas, SlateImageDecorator, SlateItalicDecorator,
    SlateLinkDecorator, SlateRichTextDecorator, SlateTextMeasurement,
    SyntaxHighlighter, TextCursor, TextEditOperation, TextPosition, TextSelection,
    TextShaper, TextUndoStack, TokenType,
};
pub use slate_performance::{
    BatchStatistics, CacheStatistics, DirtyRect, DirtyRectTracking, DrawElementBatch,
    ElementBatching, FrameTimings, InvalidateReason as SlateInvalidateReason,
    InvalidationPanel, OffscreenBuffer, PerformanceManager, PerformanceStats,
    RecycledWidget, RetainerBox, ScrollState, SlateSleep, VirtualizedListCore,
    VirtualizedRow, WakeReason, WidgetCaching,
};
pub use slate_accessibility::{
    AccessibilityManager, AccessibleNode, AccessibleRole, AccessibleState,
    AccessibleWidget, AnimationType, ArrowKey, CheckedState,
    FocusDirection, HighContrastColor, HighContrastMode, KeyboardNavigation,
    LiveRegionMode, ReducedMotion,
};
pub use slate_debug::{
    DebugOverlayKind, DebugOverlays, FrameStats, GeometryOverlay, HotStyleReload,
    OverlayRect, PickResult, SlateStats, UITestAction, UITestDriver,
    WidgetInfo, WidgetReflector,
};
pub use slate_notifications::{
    AnimationPhase, ConfirmDialog, DialogButton, DialogButtons, DialogIcon, DialogId,
    DialogResult, ModalDialog, Notification, NotificationHandle,
    NotificationId, NotificationLevel, NotificationManager, ProgressDialog, ProgressStage,
};
pub use slate_multiwindow::{
    CursorStyle as SlateWindowCursorStyle, FloatingWindow as SlateFloatingWindow,
    ModalWindow, MonitorInfo, MultiWindowManager, PopupAnchor, PopupWindow, SnapZone,
    WindowDragState, WindowEdge, WindowId, WindowLayout, WindowState,
};
pub use slate_property_editor::{
    ArrayEditorState, AssetRefEditorState, ColorPickerState, MultiObjectState,
    PropId, PropertyCategory, PropertyChange, PropertyEditor, PropertyEditorConfig,
    PropertyFilter, PropertyMetadata, PropertyRow, PropertyType, PropertyValue,
    ShowCondition, UndoEntry, UndoRedoStack,
};
pub use slate_layout::{
    BoxSlot, Canvas, CanvasSlot, GridPanel, GridSlot, HorizontalBox,
    Overlay, OverlaySlot, ScaleBox, SizeBox, SizeRule,
    SlateMargin, UniformGridPanel, VerticalBox, WidgetSwitcher, WrapBox,
    HAlign as SlateHAlign, VAlign as SlateVAlign, StretchMode as SlateStretchMode,
};
pub use slate_animation::{
    ActiveTransition, AnimatedColor, AnimatedFloat, AnimatedVec2,
    CriticalSpring, CurveHandle as SlateCurveHandle,
    CurveSequence as SlateCurveSequence, SlateEasing, SlideDirection,
    TransitionManager, WidgetTransition,
};
pub use slate_platform::{
    ClipboardManager as SlateClipboardManager, CursorType, DPIManager,
    FileDialog, FileFilter, ImageData, MonitorInfo as SlateMonitorInfo,
    PlatformContext, SystemInfo, WindowEvent, WindowId as SlateWindowId,
    WindowManager as SlateWindowManager, WindowProperties, WindowState as SlateWindowState,
    WindowType,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the UI crate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum UIError {
    /// A UI node was not found by its identifier.
    #[error("UI node not found: {0:?}")]
    NodeNotFound(UIId),

    /// Layout computation failed.
    #[error("layout error: {0}")]
    LayoutError(String),

    /// A font could not be loaded or parsed.
    #[error("font error: {0}")]
    FontError(String),

    /// A style property was invalid.
    #[error("style error: {0}")]
    StyleError(String),

    /// An animation parameter was invalid.
    #[error("animation error: {0}")]
    AnimationError(String),

    /// A widget-specific error.
    #[error("widget error: {0}")]
    WidgetError(String),
}

/// Convenience alias.
pub type UIResult<T> = Result<T, UIError>;
