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
pub mod core;
pub mod data_binding;
pub mod drag_drop;
pub mod editor_widgets;
pub mod gpu_renderer;
pub mod layout;
pub mod render_commands;
pub mod rich_text;
pub mod style;
pub mod text;
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
