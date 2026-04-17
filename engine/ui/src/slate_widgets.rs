//! Core Slate widgets -- retained-mode UI primitives for the Genovo engine.
//!
//! Each widget is a self-contained struct that stores its own visual and
//! interaction state. The three core methods every widget implements:
//!
//! - `compute_desired_size()` -- returns the natural size the widget wants.
//! - `paint()` -- emits draw commands into a [`DrawList`].
//! - `handle_event()` -- processes a [`UIEvent`] and returns an [`EventReply`].
//!
//! Widgets here are *primitive* -- they do not contain child widget trees
//! (except for single-child containers like [`Border`]).  Complex composite
//! widgets live in `slate_complex_widgets`.

use std::collections::VecDeque;

use glam::Vec2;
use genovo_core::Rect;

use crate::core::{KeyCode, KeyModifiers, MouseButton, Padding, UIEvent, UIId};
use crate::render_commands::{
    Border as BorderSpec, Color, CornerRadii, DrawCommand, DrawList, ImageScaleMode, Shadow,
    TextAlign, TextVerticalAlign, TextureId,
};

// ---------------------------------------------------------------------------
// EventReply
// ---------------------------------------------------------------------------

/// Result returned from a widget's event handler indicating how the event was
/// processed and what side-effects should occur.
#[derive(Debug, Clone, PartialEq)]
pub enum EventReply {
    /// The event was not consumed; it should continue bubbling.
    Unhandled,
    /// The event was consumed; stop bubbling.
    Handled,
    /// Request exclusive mouse capture on this widget.
    CaptureMouse,
    /// Release a previous mouse capture.
    ReleaseMouse,
    /// Request keyboard focus for this widget.
    SetFocus,
    /// Release keyboard focus.
    ClearFocus,
    /// Multiple replies chained together (processed in order).
    Chain(Vec<EventReply>),
}

impl EventReply {
    /// Chain another reply onto this one.
    pub fn then(self, other: EventReply) -> Self {
        match self {
            EventReply::Chain(mut v) => {
                v.push(other);
                EventReply::Chain(v)
            }
            _ => EventReply::Chain(vec![self, other]),
        }
    }

    /// Returns true if the event was handled (or any chained reply was handled).
    pub fn is_handled(&self) -> bool {
        match self {
            EventReply::Handled => true,
            EventReply::CaptureMouse => true,
            EventReply::SetFocus => true,
            EventReply::Chain(v) => v.iter().any(|r| r.is_handled()),
            _ => false,
        }
    }

    /// Returns true if any reply requests mouse capture.
    pub fn wants_capture(&self) -> bool {
        match self {
            EventReply::CaptureMouse => true,
            EventReply::Chain(v) => v.iter().any(|r| r.wants_capture()),
            _ => false,
        }
    }

    /// Returns true if any reply requests focus.
    pub fn wants_focus(&self) -> bool {
        match self {
            EventReply::SetFocus => true,
            EventReply::Chain(v) => v.iter().any(|r| r.wants_focus()),
            _ => false,
        }
    }

    /// Returns true if any reply releases mouse.
    pub fn releases_mouse(&self) -> bool {
        match self {
            EventReply::ReleaseMouse => true,
            EventReply::Chain(v) => v.iter().any(|r| r.releases_mouse()),
            _ => false,
        }
    }

    /// Returns true if any reply clears focus.
    pub fn clears_focus(&self) -> bool {
        match self {
            EventReply::ClearFocus => true,
            EventReply::Chain(v) => v.iter().any(|r| r.clears_focus()),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Brush -- paint material for widget backgrounds / foregrounds
// ---------------------------------------------------------------------------

/// A brush defines how a region is filled -- solid colour, texture, or
/// gradient.  Modelled after UE5 FSlateBrush.
#[derive(Debug, Clone)]
pub enum Brush {
    /// A solid colour fill.
    SolidColor(Color),
    /// A textured fill with optional tint.
    Image {
        texture: TextureId,
        tint: Color,
        scale_mode: ImageScaleMode,
    },
    /// Linear gradient between two colours.
    LinearGradient {
        start_color: Color,
        end_color: Color,
        /// Angle in radians (0 = left-to-right, PI/2 = top-to-bottom).
        angle: f32,
    },
    /// No fill (transparent).
    None,
}

impl Brush {
    pub fn solid(color: Color) -> Self {
        Brush::SolidColor(color)
    }

    pub fn image(texture: TextureId) -> Self {
        Brush::Image {
            texture,
            tint: Color::WHITE,
            scale_mode: ImageScaleMode::Stretch,
        }
    }

    pub fn image_tinted(texture: TextureId, tint: Color) -> Self {
        Brush::Image {
            texture,
            tint,
            scale_mode: ImageScaleMode::Stretch,
        }
    }

    pub fn gradient(start: Color, end: Color, angle: f32) -> Self {
        Brush::LinearGradient {
            start_color: start,
            end_color: end,
            angle,
        }
    }

    /// Paint the brush into a rect on a draw list.
    pub fn paint_rect(&self, draw_list: &mut DrawList, rect: Rect, radii: CornerRadii) {
        match self {
            Brush::SolidColor(c) => {
                draw_list.draw_rounded_rect(rect, *c, radii, BorderSpec::default());
            }
            Brush::Image {
                texture,
                tint,
                scale_mode,
            } => {
                draw_list.push(DrawCommand::Image {
                    rect,
                    texture: *texture,
                    tint: *tint,
                    corner_radii: radii,
                    scale_mode: *scale_mode,
                    uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
                });
            }
            Brush::LinearGradient {
                start_color,
                end_color,
                angle,
            } => {
                let cx = (rect.min.x + rect.max.x) * 0.5;
                let cy = (rect.min.y + rect.max.y) * 0.5;
                let hw = (rect.max.x - rect.min.x) * 0.5;
                let hh = (rect.max.y - rect.min.y) * 0.5;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let start = Vec2::new(cx - cos_a * hw, cy - sin_a * hh);
                let end = Vec2::new(cx + cos_a * hw, cy + sin_a * hh);
                draw_list.push(DrawCommand::GradientRect {
                    rect,
                    gradient: crate::render_commands::Gradient::Linear {
                        start,
                        end,
                        stops: vec![
                            crate::render_commands::GradientStop {
                                offset: 0.0,
                                color: *start_color,
                            },
                            crate::render_commands::GradientStop {
                                offset: 1.0,
                                color: *end_color,
                            },
                        ],
                    },
                    corner_radii: radii,
                });
            }
            Brush::None => {}
        }
    }
}

impl Default for Brush {
    fn default() -> Self {
        Brush::None
    }
}

// ---------------------------------------------------------------------------
// Text wrapping / alignment helpers
// ---------------------------------------------------------------------------

/// How text wraps within its allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextWrapping {
    /// No wrapping.
    NoWrap,
    /// Wrap at word boundaries.
    WordWrap,
    /// Wrap at character boundaries.
    CharWrap,
}

impl Default for TextWrapping {
    fn default() -> Self {
        TextWrapping::NoWrap
    }
}

/// Horizontal alignment for text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HAlign {
    Left,
    Center,
    Right,
}

impl Default for HAlign {
    fn default() -> Self {
        HAlign::Left
    }
}

/// Vertical alignment for text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VAlign {
    Top,
    Center,
    Bottom,
}

impl Default for VAlign {
    fn default() -> Self {
        VAlign::Top
    }
}

// ---------------------------------------------------------------------------
// Widget trait
// ---------------------------------------------------------------------------

/// Trait shared by all Slate widgets.
pub trait SlateWidget {
    /// Compute the natural size this widget would like to be, given an optional
    /// maximum width constraint.
    fn compute_desired_size(&self, max_width: Option<f32>) -> Vec2;

    /// Emit draw commands for this widget into the draw list.
    fn paint(&self, draw_list: &mut DrawList, rect: Rect);

    /// Handle an input event.  Returns an [`EventReply`] describing how the
    /// event was processed.
    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply;
}

// =========================================================================
//  1. TextBlock
// =========================================================================

/// Simple text display widget with configurable wrapping, alignment, shadow,
/// and outline.
#[derive(Debug, Clone)]
pub struct TextBlock {
    pub id: UIId,
    pub text: String,
    pub font_size: f32,
    pub font_id: u32,
    pub color: Color,
    pub shadow_color: Color,
    pub shadow_offset: Vec2,
    pub outline_color: Color,
    pub outline_width: f32,
    pub wrapping: TextWrapping,
    pub h_align: HAlign,
    pub v_align: VAlign,
    pub auto_size: bool,
    pub line_height_multiplier: f32,
    pub max_lines: Option<usize>,
    pub ellipsis: bool,
    pub min_desired_width: f32,
    pub opacity: f32,
    pub visible: bool,
}

impl TextBlock {
    pub fn new(text: &str) -> Self {
        Self {
            id: UIId::INVALID,
            text: text.to_string(),
            font_size: 14.0,
            font_id: 0,
            color: Color::WHITE,
            shadow_color: Color::TRANSPARENT,
            shadow_offset: Vec2::new(1.0, 1.0),
            outline_color: Color::TRANSPARENT,
            outline_width: 0.0,
            wrapping: TextWrapping::NoWrap,
            h_align: HAlign::Left,
            v_align: VAlign::Top,
            auto_size: true,
            line_height_multiplier: 1.2,
            max_lines: None,
            ellipsis: false,
            min_desired_width: 0.0,
            opacity: 1.0,
            visible: true,
        }
    }

    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn with_wrapping(mut self, wrap: TextWrapping) -> Self {
        self.wrapping = wrap;
        self
    }

    pub fn with_alignment(mut self, h: HAlign, v: VAlign) -> Self {
        self.h_align = h;
        self.v_align = v;
        self
    }

    pub fn with_shadow(mut self, color: Color, offset: Vec2) -> Self {
        self.shadow_color = color;
        self.shadow_offset = offset;
        self
    }

    pub fn with_outline(mut self, color: Color, width: f32) -> Self {
        self.outline_color = color;
        self.outline_width = width;
        self
    }

    pub fn with_max_lines(mut self, n: usize) -> Self {
        self.max_lines = Some(n);
        self.ellipsis = true;
        self
    }

    pub fn with_opacity(mut self, a: f32) -> Self {
        self.opacity = a;
        self
    }

    /// Rough character width estimate for layout.
    fn char_width(&self) -> f32 {
        self.font_size * 0.6
    }

    /// Line height including multiplier.
    fn line_height(&self) -> f32 {
        self.font_size * self.line_height_multiplier
    }

    /// Word-wrap the text into lines given a max width.  Returns a list of
    /// line strings.
    fn wrap_lines(&self, max_width: Option<f32>) -> Vec<String> {
        if self.text.is_empty() {
            return vec![String::new()];
        }

        let mw = match (max_width, self.wrapping) {
            (Some(w), TextWrapping::WordWrap) | (Some(w), TextWrapping::CharWrap) => w,
            _ => return self.text.lines().map(String::from).collect(),
        };

        let cw = self.char_width();
        let max_chars = (mw / cw).max(1.0) as usize;

        let mut lines = Vec::new();
        for paragraph in self.text.lines() {
            if paragraph.is_empty() {
                lines.push(String::new());
                continue;
            }

            match self.wrapping {
                TextWrapping::WordWrap => {
                    let words: Vec<&str> = paragraph.split_whitespace().collect();
                    let mut current_line = String::new();
                    for word in words {
                        if current_line.is_empty() {
                            current_line = word.to_string();
                        } else if current_line.len() + 1 + word.len() > max_chars {
                            lines.push(current_line);
                            current_line = word.to_string();
                        } else {
                            current_line.push(' ');
                            current_line.push_str(word);
                        }
                    }
                    if !current_line.is_empty() {
                        lines.push(current_line);
                    }
                }
                TextWrapping::CharWrap => {
                    let chars: Vec<char> = paragraph.chars().collect();
                    for chunk in chars.chunks(max_chars) {
                        lines.push(chunk.iter().collect());
                    }
                }
                TextWrapping::NoWrap => {
                    lines.push(paragraph.to_string());
                }
            }
        }

        // Apply max_lines with ellipsis.
        if let Some(max) = self.max_lines {
            if lines.len() > max && self.ellipsis {
                lines.truncate(max);
                if let Some(last) = lines.last_mut() {
                    if last.len() > 3 {
                        let l = last.len();
                        last.replace_range(l - 3.., "...");
                    }
                }
            }
        }

        lines
    }

    /// Convert our alignment to draw-command alignment.
    fn text_align(&self) -> TextAlign {
        match self.h_align {
            HAlign::Left => TextAlign::Left,
            HAlign::Center => TextAlign::Center,
            HAlign::Right => TextAlign::Right,
        }
    }
}

impl SlateWidget for TextBlock {
    fn compute_desired_size(&self, max_width: Option<f32>) -> Vec2 {
        let lines = self.wrap_lines(max_width);
        let line_h = self.line_height();
        let total_h = lines.len() as f32 * line_h;

        let max_line_w = lines
            .iter()
            .map(|l| l.len() as f32 * self.char_width())
            .fold(0.0f32, f32::max);

        let w = max_line_w.max(self.min_desired_width);
        Vec2::new(w, total_h)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible || self.opacity <= 0.0 {
            return;
        }

        let lines = self.wrap_lines(Some(rect.width()));
        let line_h = self.line_height();
        let total_h = lines.len() as f32 * line_h;

        // Vertical offset for alignment.
        let y_start = match self.v_align {
            VAlign::Top => rect.min.y,
            VAlign::Center => rect.min.y + (rect.height() - total_h) * 0.5,
            VAlign::Bottom => rect.max.y - total_h,
        };

        let text_color = self.color.with_alpha(self.color.a * self.opacity);

        for (i, line) in lines.iter().enumerate() {
            let y = y_start + i as f32 * line_h;

            let x = match self.h_align {
                HAlign::Left => rect.min.x,
                HAlign::Center => {
                    let lw = line.len() as f32 * self.char_width();
                    rect.min.x + (rect.width() - lw) * 0.5
                }
                HAlign::Right => {
                    let lw = line.len() as f32 * self.char_width();
                    rect.max.x - lw
                }
            };

            // Shadow pass.
            if self.shadow_color.a > 0.0 {
                draw_list.push(DrawCommand::Text {
                    text: line.clone(),
                    position: Vec2::new(x + self.shadow_offset.x, y + self.shadow_offset.y),
                    font_size: self.font_size,
                    color: self.shadow_color.with_alpha(self.shadow_color.a * self.opacity),
                    font_id: self.font_id,
                    max_width: Some(rect.width()),
                    align: self.text_align(),
                    vertical_align: TextVerticalAlign::Top,
                });
            }

            // Outline passes (4 offsets for cheap outline).
            if self.outline_width > 0.0 && self.outline_color.a > 0.0 {
                let ow = self.outline_width;
                let oc = self.outline_color.with_alpha(self.outline_color.a * self.opacity);
                for &(dx, dy) in &[(-ow, 0.0), (ow, 0.0), (0.0, -ow), (0.0, ow)] {
                    draw_list.push(DrawCommand::Text {
                        text: line.clone(),
                        position: Vec2::new(x + dx, y + dy),
                        font_size: self.font_size,
                        color: oc,
                        font_id: self.font_id,
                        max_width: Some(rect.width()),
                        align: self.text_align(),
                        vertical_align: TextVerticalAlign::Top,
                    });
                }
            }

            // Main text pass.
            draw_list.push(DrawCommand::Text {
                text: line.clone(),
                position: Vec2::new(x, y),
                font_size: self.font_size,
                color: text_color,
                font_id: self.font_id,
                max_width: Some(rect.width()),
                align: self.text_align(),
                vertical_align: TextVerticalAlign::Top,
            });
        }
    }

    fn handle_event(&mut self, _event: &UIEvent, _rect: Rect) -> EventReply {
        // TextBlock is display-only; it does not consume events.
        EventReply::Unhandled
    }
}

// =========================================================================
//  2. RichTextBlock
// =========================================================================

/// Inline text decoration type.
#[derive(Debug, Clone, PartialEq)]
pub enum TextDecoration {
    Bold,
    Italic,
    Underline,
    Strikethrough,
    Color(Color),
    FontSize(f32),
    Link(String),
    InlineImage {
        texture: TextureId,
        size: Vec2,
    },
}

/// A span of text with decorations.
#[derive(Debug, Clone)]
pub struct RichTextSpan {
    pub text: String,
    pub decorations: Vec<TextDecoration>,
}

impl RichTextSpan {
    pub fn plain(text: &str) -> Self {
        Self {
            text: text.to_string(),
            decorations: Vec::new(),
        }
    }

    pub fn bold(text: &str) -> Self {
        Self {
            text: text.to_string(),
            decorations: vec![TextDecoration::Bold],
        }
    }

    pub fn italic(text: &str) -> Self {
        Self {
            text: text.to_string(),
            decorations: vec![TextDecoration::Italic],
        }
    }

    pub fn colored(text: &str, color: Color) -> Self {
        Self {
            text: text.to_string(),
            decorations: vec![TextDecoration::Color(color)],
        }
    }

    pub fn link(text: &str, url: &str) -> Self {
        Self {
            text: text.to_string(),
            decorations: vec![
                TextDecoration::Link(url.to_string()),
                TextDecoration::Color(Color::from_hex("#4488FF")),
                TextDecoration::Underline,
            ],
        }
    }

    pub fn inline_image(texture: TextureId, size: Vec2) -> Self {
        Self {
            text: String::new(),
            decorations: vec![TextDecoration::InlineImage { texture, size }],
        }
    }

    pub fn with_decoration(mut self, dec: TextDecoration) -> Self {
        self.decorations.push(dec);
        self
    }

    /// Effective font size for this span.
    fn effective_font_size(&self, base: f32) -> f32 {
        for dec in &self.decorations {
            if let TextDecoration::FontSize(s) = dec {
                return *s;
            }
        }
        base
    }

    /// Effective colour for this span.
    fn effective_color(&self, base: Color) -> Color {
        for dec in &self.decorations {
            if let TextDecoration::Color(c) = dec {
                return *c;
            }
        }
        base
    }

    fn is_bold(&self) -> bool {
        self.decorations.iter().any(|d| matches!(d, TextDecoration::Bold))
    }

    fn is_italic(&self) -> bool {
        self.decorations.iter().any(|d| matches!(d, TextDecoration::Italic))
    }

    fn is_underline(&self) -> bool {
        self.decorations.iter().any(|d| matches!(d, TextDecoration::Underline))
    }

    fn is_strikethrough(&self) -> bool {
        self.decorations
            .iter()
            .any(|d| matches!(d, TextDecoration::Strikethrough))
    }

    fn link_url(&self) -> Option<&str> {
        for dec in &self.decorations {
            if let TextDecoration::Link(url) = dec {
                return Some(url.as_str());
            }
        }
        None
    }

    fn inline_image_info(&self) -> Option<(TextureId, Vec2)> {
        for dec in &self.decorations {
            if let TextDecoration::InlineImage { texture, size } = dec {
                return Some((*texture, *size));
            }
        }
        None
    }
}

/// Trait for custom rich-text decorators (user-defined inline rendering).
pub trait RichTextDecorator: std::fmt::Debug {
    /// Name of the tag this decorator handles (e.g. "b", "color").
    fn tag_name(&self) -> &str;

    /// Apply the decoration by modifying span draw properties.
    fn apply(&self, span: &mut RichTextSpan);

    /// Optional: render additional visuals over a span (e.g. wavy underline).
    fn paint_overlay(&self, _draw_list: &mut DrawList, _span_rect: Rect) {}
}

/// Built-in bold decorator.
#[derive(Debug)]
pub struct BoldDecorator;
impl RichTextDecorator for BoldDecorator {
    fn tag_name(&self) -> &str { "b" }
    fn apply(&self, span: &mut RichTextSpan) {
        if !span.is_bold() {
            span.decorations.push(TextDecoration::Bold);
        }
    }
}

/// Built-in italic decorator.
#[derive(Debug)]
pub struct ItalicDecorator;
impl RichTextDecorator for ItalicDecorator {
    fn tag_name(&self) -> &str { "i" }
    fn apply(&self, span: &mut RichTextSpan) {
        if !span.is_italic() {
            span.decorations.push(TextDecoration::Italic);
        }
    }
}

/// Built-in colour decorator (expects attribute).
#[derive(Debug)]
pub struct ColorDecorator {
    pub color: Color,
}
impl ColorDecorator {
    pub fn new(color: Color) -> Self { Self { color } }
}
impl RichTextDecorator for ColorDecorator {
    fn tag_name(&self) -> &str { "color" }
    fn apply(&self, span: &mut RichTextSpan) {
        span.decorations.push(TextDecoration::Color(self.color));
    }
}

/// Link decorator.
#[derive(Debug)]
pub struct LinkDecorator {
    pub url: String,
}
impl LinkDecorator {
    pub fn new(url: &str) -> Self { Self { url: url.to_string() } }
}
impl RichTextDecorator for LinkDecorator {
    fn tag_name(&self) -> &str { "a" }
    fn apply(&self, span: &mut RichTextSpan) {
        span.decorations.push(TextDecoration::Link(self.url.clone()));
        span.decorations.push(TextDecoration::Underline);
        span.decorations.push(TextDecoration::Color(Color::from_hex("#4488FF")));
    }
}

/// Image decorator for inline images.
#[derive(Debug)]
pub struct ImageDecorator {
    pub texture: TextureId,
    pub size: Vec2,
}
impl ImageDecorator {
    pub fn new(texture: TextureId, size: Vec2) -> Self { Self { texture, size } }
}
impl RichTextDecorator for ImageDecorator {
    fn tag_name(&self) -> &str { "img" }
    fn apply(&self, span: &mut RichTextSpan) {
        span.decorations
            .push(TextDecoration::InlineImage {
                texture: self.texture,
                size: self.size,
            });
    }
}

/// Rich text block supporting multiple styled spans with inline decorations.
#[derive(Debug, Clone)]
pub struct RichTextBlock {
    pub id: UIId,
    pub spans: Vec<RichTextSpan>,
    pub base_font_size: f32,
    pub base_color: Color,
    pub font_id: u32,
    pub line_height_multiplier: f32,
    pub wrapping: TextWrapping,
    pub h_align: HAlign,
    pub v_align: VAlign,
    pub max_width: Option<f32>,
    pub visible: bool,
    pub opacity: f32,
    /// Markup source -- set this and call `parse_markup()` to populate spans.
    pub markup_source: String,
}

impl RichTextBlock {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            spans: Vec::new(),
            base_font_size: 14.0,
            base_color: Color::WHITE,
            font_id: 0,
            line_height_multiplier: 1.2,
            wrapping: TextWrapping::WordWrap,
            h_align: HAlign::Left,
            v_align: VAlign::Top,
            max_width: None,
            visible: true,
            opacity: 1.0,
            markup_source: String::new(),
        }
    }

    pub fn with_spans(mut self, spans: Vec<RichTextSpan>) -> Self {
        self.spans = spans;
        self
    }

    pub fn add_span(&mut self, span: RichTextSpan) {
        self.spans.push(span);
    }

    pub fn clear(&mut self) {
        self.spans.clear();
    }

    /// Parse simple markup into spans.
    /// Supports: `<b>`, `<i>`, `<color=#HEX>`, `<a href="">`, `<img src="" w="" h="">`.
    pub fn parse_markup(&mut self, markup: &str) {
        self.spans.clear();
        let mut remaining = markup;
        let mut current_text = String::new();
        let mut decoration_stack: Vec<TextDecoration> = Vec::new();

        while !remaining.is_empty() {
            if let Some(tag_start) = remaining.find('<') {
                // Text before the tag.
                if tag_start > 0 {
                    let text = &remaining[..tag_start];
                    current_text.push_str(text);
                }

                if let Some(tag_end) = remaining[tag_start..].find('>') {
                    let tag_content = &remaining[tag_start + 1..tag_start + tag_end];
                    remaining = &remaining[tag_start + tag_end + 1..];

                    if tag_content.starts_with('/') {
                        // Closing tag -- flush current text with current decorations.
                        if !current_text.is_empty() {
                            self.spans.push(RichTextSpan {
                                text: std::mem::take(&mut current_text),
                                decorations: decoration_stack.clone(),
                            });
                        }
                        let tag_name = &tag_content[1..].trim();
                        // Pop matching decoration.
                        if let Some(pos) = decoration_stack.iter().rposition(|d| {
                            match (d, *tag_name) {
                                (TextDecoration::Bold, "b") => true,
                                (TextDecoration::Italic, "i") => true,
                                (TextDecoration::Color(_), "color") => true,
                                (TextDecoration::Link(_), "a") => true,
                                _ => false,
                            }
                        }) {
                            decoration_stack.remove(pos);
                            // Also remove associated decorations for links.
                            if *tag_name == "a" {
                                decoration_stack.retain(|d| {
                                    !matches!(d, TextDecoration::Underline)
                                });
                            }
                        }
                    } else if tag_content == "b" {
                        if !current_text.is_empty() {
                            self.spans.push(RichTextSpan {
                                text: std::mem::take(&mut current_text),
                                decorations: decoration_stack.clone(),
                            });
                        }
                        decoration_stack.push(TextDecoration::Bold);
                    } else if tag_content == "i" {
                        if !current_text.is_empty() {
                            self.spans.push(RichTextSpan {
                                text: std::mem::take(&mut current_text),
                                decorations: decoration_stack.clone(),
                            });
                        }
                        decoration_stack.push(TextDecoration::Italic);
                    } else if tag_content.starts_with("color=") {
                        if !current_text.is_empty() {
                            self.spans.push(RichTextSpan {
                                text: std::mem::take(&mut current_text),
                                decorations: decoration_stack.clone(),
                            });
                        }
                        let hex = tag_content[6..].trim_matches('#').trim_matches('"');
                        let color_hex = if hex.starts_with('#') {
                            hex.to_string()
                        } else {
                            format!("#{}", hex)
                        };
                        decoration_stack.push(TextDecoration::Color(Color::from_hex(&color_hex)));
                    } else if tag_content.starts_with("a ") {
                        if !current_text.is_empty() {
                            self.spans.push(RichTextSpan {
                                text: std::mem::take(&mut current_text),
                                decorations: decoration_stack.clone(),
                            });
                        }
                        // Extract href.
                        let href = Self::extract_attr(tag_content, "href")
                            .unwrap_or_default();
                        decoration_stack.push(TextDecoration::Link(href));
                        decoration_stack.push(TextDecoration::Underline);
                        decoration_stack.push(TextDecoration::Color(Color::from_hex("#4488FF")));
                    } else if tag_content.starts_with("img ") {
                        // Inline image -- self-closing.
                        if !current_text.is_empty() {
                            self.spans.push(RichTextSpan {
                                text: std::mem::take(&mut current_text),
                                decorations: decoration_stack.clone(),
                            });
                        }
                        let _src = Self::extract_attr(tag_content, "src")
                            .unwrap_or_default();
                        let w: f32 = Self::extract_attr(tag_content, "w")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(16.0);
                        let h: f32 = Self::extract_attr(tag_content, "h")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(16.0);
                        self.spans.push(RichTextSpan {
                            text: String::new(),
                            decorations: vec![TextDecoration::InlineImage {
                                texture: TextureId::INVALID,
                                size: Vec2::new(w, h),
                            }],
                        });
                    }
                } else {
                    // No closing '>' found; treat the rest as plain text.
                    current_text.push_str(remaining);
                    remaining = "";
                }
            } else {
                current_text.push_str(remaining);
                remaining = "";
            }
        }

        if !current_text.is_empty() {
            self.spans.push(RichTextSpan {
                text: current_text,
                decorations: decoration_stack,
            });
        }
    }

    fn extract_attr(tag: &str, attr_name: &str) -> Option<String> {
        let pattern = format!("{}=\"", attr_name);
        if let Some(start) = tag.find(&pattern) {
            let value_start = start + pattern.len();
            if let Some(end) = tag[value_start..].find('"') {
                return Some(tag[value_start..value_start + end].to_string());
            }
        }
        // Try without quotes.
        let pattern2 = format!("{}=", attr_name);
        if let Some(start) = tag.find(&pattern2) {
            let value_start = start + pattern2.len();
            let end = tag[value_start..]
                .find(|c: char| c.is_whitespace() || c == '/' || c == '>')
                .unwrap_or(tag.len() - value_start);
            return Some(tag[value_start..value_start + end].to_string());
        }
        None
    }

    fn char_width(&self, font_size: f32) -> f32 {
        font_size * 0.6
    }

    fn line_height(&self) -> f32 {
        self.base_font_size * self.line_height_multiplier
    }
}

impl SlateWidget for RichTextBlock {
    fn compute_desired_size(&self, max_width: Option<f32>) -> Vec2 {
        let mw = max_width.or(self.max_width).unwrap_or(f32::MAX);
        let line_h = self.line_height();

        let mut x = 0.0f32;
        let mut max_x = 0.0f32;
        let mut lines = 1usize;

        for span in &self.spans {
            if let Some((_tex, size)) = span.inline_image_info() {
                if x + size.x > mw && x > 0.0 {
                    lines += 1;
                    x = 0.0;
                }
                x += size.x;
                max_x = max_x.max(x);
            } else {
                let fs = span.effective_font_size(self.base_font_size);
                let cw = self.char_width(fs);
                for ch in span.text.chars() {
                    if ch == '\n' {
                        max_x = max_x.max(x);
                        x = 0.0;
                        lines += 1;
                        continue;
                    }
                    let char_w = cw;
                    if x + char_w > mw && x > 0.0 {
                        max_x = max_x.max(x);
                        x = 0.0;
                        lines += 1;
                    }
                    x += char_w;
                }
            }
        }
        max_x = max_x.max(x);
        Vec2::new(max_x, lines as f32 * line_h)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible || self.opacity <= 0.0 {
            return;
        }

        let mw = rect.width();
        let line_h = self.line_height();
        let mut cursor_x = rect.min.x;
        let mut cursor_y = rect.min.y;

        for span in &self.spans {
            // Inline image.
            if let Some((tex, size)) = span.inline_image_info() {
                if cursor_x + size.x > rect.max.x && cursor_x > rect.min.x {
                    cursor_x = rect.min.x;
                    cursor_y += line_h;
                }
                draw_list.push(DrawCommand::Image {
                    rect: Rect::new(
                        Vec2::new(cursor_x, cursor_y),
                        Vec2::new(cursor_x + size.x, cursor_y + size.y),
                    ),
                    texture: tex,
                    tint: Color::WHITE.with_alpha(self.opacity),
                    corner_radii: CornerRadii::ZERO,
                    scale_mode: ImageScaleMode::Fit,
                    uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
                });
                cursor_x += size.x;
                continue;
            }

            let fs = span.effective_font_size(self.base_font_size);
            let color = span.effective_color(self.base_color).with_alpha(
                span.effective_color(self.base_color).a * self.opacity,
            );
            let cw = self.char_width(fs);
            let bold = span.is_bold();
            let underline = span.is_underline();
            let strikethrough = span.is_strikethrough();

            // Accumulate a contiguous run of characters on the same line.
            let mut run_start_x = cursor_x;
            let mut run_text = String::new();

            let flush_run = |dl: &mut DrawList,
                             txt: &str,
                             sx: f32,
                             sy: f32,
                             fs: f32,
                             col: Color,
                             font_id: u32,
                             ul: bool,
                             st: bool,
                             cw: f32| {
                if txt.is_empty() {
                    return;
                }
                dl.push(DrawCommand::Text {
                    text: txt.to_string(),
                    position: Vec2::new(sx, sy),
                    font_size: fs,
                    color: col,
                    font_id,
                    max_width: None,
                    align: TextAlign::Left,
                    vertical_align: TextVerticalAlign::Top,
                });

                let run_w = txt.len() as f32 * cw;
                if ul {
                    dl.push(DrawCommand::Line {
                        start: Vec2::new(sx, sy + fs + 1.0),
                        end: Vec2::new(sx + run_w, sy + fs + 1.0),
                        color: col,
                        thickness: 1.0,
                    });
                }
                if st {
                    let mid_y = sy + fs * 0.5;
                    dl.push(DrawCommand::Line {
                        start: Vec2::new(sx, mid_y),
                        end: Vec2::new(sx + run_w, mid_y),
                        color: col,
                        thickness: 1.0,
                    });
                }
            };

            for ch in span.text.chars() {
                if ch == '\n' {
                    flush_run(
                        draw_list,
                        &run_text,
                        run_start_x,
                        cursor_y,
                        fs,
                        color,
                        self.font_id,
                        underline,
                        strikethrough,
                        cw,
                    );
                    run_text.clear();
                    cursor_x = rect.min.x;
                    cursor_y += line_h;
                    run_start_x = cursor_x;
                    continue;
                }

                if cursor_x + cw > rect.max.x && cursor_x > rect.min.x {
                    flush_run(
                        draw_list,
                        &run_text,
                        run_start_x,
                        cursor_y,
                        fs,
                        color,
                        self.font_id,
                        underline,
                        strikethrough,
                        cw,
                    );
                    run_text.clear();
                    cursor_x = rect.min.x;
                    cursor_y += line_h;
                    run_start_x = cursor_x;
                }

                run_text.push(ch);
                cursor_x += cw;
            }

            flush_run(
                draw_list,
                &run_text,
                run_start_x,
                cursor_y,
                fs,
                color,
                self.font_id,
                underline,
                strikethrough,
                cw,
            );
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        // Rich text blocks can handle clicks on link spans.
        match event {
            UIEvent::Click { position, .. } => {
                if !rect.contains(*position) {
                    return EventReply::Unhandled;
                }
                // Check if click is on a link span.
                let line_h = self.line_height();
                let mut cx = rect.min.x;
                let mut cy = rect.min.y;
                for span in &self.spans {
                    if let Some(url) = span.link_url() {
                        let fs = span.effective_font_size(self.base_font_size);
                        let cw = self.char_width(fs);
                        let span_w = span.text.len() as f32 * cw;
                        let span_rect = Rect::new(
                            Vec2::new(cx, cy),
                            Vec2::new(cx + span_w, cy + line_h),
                        );
                        if span_rect.contains(*position) {
                            // Link was clicked.
                            return EventReply::Handled;
                        }
                        cx += span_w;
                    } else if let Some((_, size)) = span.inline_image_info() {
                        cx += size.x;
                    } else {
                        let fs = span.effective_font_size(self.base_font_size);
                        let cw = self.char_width(fs);
                        for ch in span.text.chars() {
                            if ch == '\n' {
                                cx = rect.min.x;
                                cy += line_h;
                            } else {
                                cx += cw;
                                if cx > rect.max.x {
                                    cx = rect.min.x;
                                    cy += line_h;
                                }
                            }
                        }
                    }
                }
                EventReply::Unhandled
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
//  3. ImageWidget
// =========================================================================

/// Stretch mode for images (mirrors UE5 naming).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StretchMode {
    /// Scale to fill the allocation exactly (may distort).
    Stretch,
    /// Scale uniformly to fit inside the allocation (letterbox).
    Fit,
    /// Scale uniformly to fill the allocation (may crop).
    Fill,
    /// Display at natural size, centered.
    None,
}

impl Default for StretchMode {
    fn default() -> Self {
        StretchMode::Stretch
    }
}

/// Displays a textured image with tint and stretch control.
#[derive(Debug, Clone)]
pub struct ImageWidget {
    pub id: UIId,
    pub texture: TextureId,
    pub tint: Color,
    pub stretch: StretchMode,
    pub desired_size: Vec2,
    pub corner_radii: CornerRadii,
    pub opacity: f32,
    pub visible: bool,
    pub uv_min: Vec2,
    pub uv_max: Vec2,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
}

impl ImageWidget {
    pub fn new(texture: TextureId) -> Self {
        Self {
            id: UIId::INVALID,
            texture,
            tint: Color::WHITE,
            stretch: StretchMode::Stretch,
            desired_size: Vec2::new(64.0, 64.0),
            corner_radii: CornerRadii::ZERO,
            opacity: 1.0,
            visible: true,
            uv_min: Vec2::ZERO,
            uv_max: Vec2::ONE,
            flip_horizontal: false,
            flip_vertical: false,
        }
    }

    pub fn with_size(mut self, size: Vec2) -> Self {
        self.desired_size = size;
        self
    }

    pub fn with_tint(mut self, tint: Color) -> Self {
        self.tint = tint;
        self
    }

    pub fn with_stretch(mut self, mode: StretchMode) -> Self {
        self.stretch = mode;
        self
    }

    pub fn with_corner_radii(mut self, radii: CornerRadii) -> Self {
        self.corner_radii = radii;
        self
    }

    /// Compute the image rect given the allocation rect and stretch mode.
    fn compute_image_rect(&self, alloc: Rect) -> Rect {
        let alloc_w = alloc.width();
        let alloc_h = alloc.height();
        let img_w = self.desired_size.x;
        let img_h = self.desired_size.y;

        match self.stretch {
            StretchMode::Stretch => alloc,
            StretchMode::Fit => {
                if img_w <= 0.0 || img_h <= 0.0 {
                    return alloc;
                }
                let scale = (alloc_w / img_w).min(alloc_h / img_h);
                let w = img_w * scale;
                let h = img_h * scale;
                let cx = alloc.min.x + (alloc_w - w) * 0.5;
                let cy = alloc.min.y + (alloc_h - h) * 0.5;
                Rect::new(Vec2::new(cx, cy), Vec2::new(cx + w, cy + h))
            }
            StretchMode::Fill => {
                if img_w <= 0.0 || img_h <= 0.0 {
                    return alloc;
                }
                let scale = (alloc_w / img_w).max(alloc_h / img_h);
                let w = img_w * scale;
                let h = img_h * scale;
                let cx = alloc.min.x + (alloc_w - w) * 0.5;
                let cy = alloc.min.y + (alloc_h - h) * 0.5;
                Rect::new(Vec2::new(cx, cy), Vec2::new(cx + w, cy + h))
            }
            StretchMode::None => {
                let cx = alloc.min.x + (alloc_w - img_w) * 0.5;
                let cy = alloc.min.y + (alloc_h - img_h) * 0.5;
                Rect::new(Vec2::new(cx, cy), Vec2::new(cx + img_w, cy + img_h))
            }
        }
    }
}

impl SlateWidget for ImageWidget {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        self.desired_size
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible || self.opacity <= 0.0 {
            return;
        }

        let img_rect = self.compute_image_rect(rect);
        let mut uv_min = self.uv_min;
        let mut uv_max = self.uv_max;
        if self.flip_horizontal {
            std::mem::swap(&mut uv_min.x, &mut uv_max.x);
        }
        if self.flip_vertical {
            std::mem::swap(&mut uv_min.y, &mut uv_max.y);
        }

        draw_list.push(DrawCommand::Image {
            rect: img_rect,
            texture: self.texture,
            tint: self.tint.with_alpha(self.tint.a * self.opacity),
            corner_radii: self.corner_radii,
            scale_mode: match self.stretch {
                StretchMode::Stretch => ImageScaleMode::Stretch,
                StretchMode::Fit => ImageScaleMode::Fit,
                StretchMode::Fill => ImageScaleMode::Fill,
                StretchMode::None => ImageScaleMode::Center,
            },
            uv_rect: Rect::new(uv_min, uv_max),
        });
    }

    fn handle_event(&mut self, _event: &UIEvent, _rect: Rect) -> EventReply {
        EventReply::Unhandled
    }
}

// =========================================================================
//  4. BorderWidget (single child container)
// =========================================================================

/// A single-child container that draws a background brush and applies padding
/// around its child.
#[derive(Debug, Clone)]
pub struct BorderWidget {
    pub id: UIId,
    pub background: Brush,
    pub border: BorderSpec,
    pub corner_radii: CornerRadii,
    pub padding: Padding,
    pub shadow: Option<Shadow>,
    pub foreground_color: Color,
    pub child_desired_size: Vec2,
    pub visible: bool,
    pub opacity: f32,
}

impl BorderWidget {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            background: Brush::None,
            border: BorderSpec::default(),
            corner_radii: CornerRadii::ZERO,
            padding: Padding::ZERO,
            shadow: None,
            foreground_color: Color::WHITE,
            child_desired_size: Vec2::ZERO,
            visible: true,
            opacity: 1.0,
        }
    }

    pub fn with_background(mut self, brush: Brush) -> Self {
        self.background = brush;
        self
    }

    pub fn with_background_color(mut self, color: Color) -> Self {
        self.background = Brush::SolidColor(color);
        self
    }

    pub fn with_border(mut self, border: BorderSpec) -> Self {
        self.border = border;
        self
    }

    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_corner_radii(mut self, radii: CornerRadii) -> Self {
        self.corner_radii = radii;
        self
    }

    pub fn with_shadow(mut self, shadow: Shadow) -> Self {
        self.shadow = Some(shadow);
        self
    }

    pub fn set_child_size(&mut self, size: Vec2) {
        self.child_desired_size = size;
    }

    /// Returns the content rect (inside padding) for a given outer rect.
    pub fn content_rect(&self, outer: Rect) -> Rect {
        Rect::new(
            Vec2::new(
                outer.min.x + self.padding.left,
                outer.min.y + self.padding.top,
            ),
            Vec2::new(
                outer.max.x - self.padding.right,
                outer.max.y - self.padding.bottom,
            ),
        )
    }
}

impl SlateWidget for BorderWidget {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        Vec2::new(
            self.child_desired_size.x + self.padding.left + self.padding.right,
            self.child_desired_size.y + self.padding.top + self.padding.bottom,
        )
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible || self.opacity <= 0.0 {
            return;
        }

        // Shadow.
        if let Some(ref shadow) = self.shadow {
            let shadow_rect = Rect::new(
                Vec2::new(
                    rect.min.x + shadow.offset.x - shadow.spread,
                    rect.min.y + shadow.offset.y - shadow.spread,
                ),
                Vec2::new(
                    rect.max.x + shadow.offset.x + shadow.spread,
                    rect.max.y + shadow.offset.y + shadow.spread,
                ),
            );
            draw_list.draw_rounded_rect(
                shadow_rect,
                shadow.color.with_alpha(shadow.color.a * self.opacity),
                self.corner_radii,
                BorderSpec::default(),
            );
        }

        // Background brush.
        self.background.paint_rect(draw_list, rect, self.corner_radii);

        // Border outline.
        if self.border.width > 0.0 && self.border.color.a > 0.0 {
            draw_list.draw_rounded_rect(
                rect,
                Color::TRANSPARENT,
                self.corner_radii,
                BorderSpec::new(
                    self.border.color.with_alpha(self.border.color.a * self.opacity),
                    self.border.width,
                ),
            );
        }
    }

    fn handle_event(&mut self, _event: &UIEvent, _rect: Rect) -> EventReply {
        EventReply::Unhandled
    }
}

// =========================================================================
//  5. Separator
// =========================================================================

/// Orientation for separator and slider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

impl Default for Orientation {
    fn default() -> Self {
        Orientation::Horizontal
    }
}

/// A thin line separator (horizontal or vertical).
#[derive(Debug, Clone)]
pub struct Separator {
    pub id: UIId,
    pub orientation: Orientation,
    pub color: Color,
    pub thickness: f32,
    pub length: f32,
    pub padding: f32,
    pub visible: bool,
}

impl Separator {
    pub fn horizontal() -> Self {
        Self {
            id: UIId::INVALID,
            orientation: Orientation::Horizontal,
            color: Color::from_rgba8(128, 128, 128, 128),
            thickness: 1.0,
            length: 0.0,
            padding: 4.0,
            visible: true,
        }
    }

    pub fn vertical() -> Self {
        Self {
            id: UIId::INVALID,
            orientation: Orientation::Vertical,
            color: Color::from_rgba8(128, 128, 128, 128),
            thickness: 1.0,
            length: 0.0,
            padding: 4.0,
            visible: true,
        }
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn with_thickness(mut self, t: f32) -> Self {
        self.thickness = t;
        self
    }
}

impl SlateWidget for Separator {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        match self.orientation {
            Orientation::Horizontal => {
                Vec2::new(self.length, self.thickness + self.padding * 2.0)
            }
            Orientation::Vertical => {
                Vec2::new(self.thickness + self.padding * 2.0, self.length)
            }
        }
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }
        match self.orientation {
            Orientation::Horizontal => {
                let y = (rect.min.y + rect.max.y) * 0.5;
                draw_list.draw_line(
                    Vec2::new(rect.min.x, y),
                    Vec2::new(rect.max.x, y),
                    self.color,
                    self.thickness,
                );
            }
            Orientation::Vertical => {
                let x = (rect.min.x + rect.max.x) * 0.5;
                draw_list.draw_line(
                    Vec2::new(x, rect.min.y),
                    Vec2::new(x, rect.max.y),
                    self.color,
                    self.thickness,
                );
            }
        }
    }

    fn handle_event(&mut self, _event: &UIEvent, _rect: Rect) -> EventReply {
        EventReply::Unhandled
    }
}

// =========================================================================
//  6. SlateButton (full Slate-style button)
// =========================================================================

/// Visual state of a button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonState {
    Normal,
    Hovered,
    Pressed,
    Disabled,
}

impl Default for ButtonState {
    fn default() -> Self {
        ButtonState::Normal
    }
}

/// A Slate-style button with per-state brushes, click delegate, and content
/// slot.
#[derive(Debug, Clone)]
pub struct SlateButton {
    pub id: UIId,
    pub state: ButtonState,
    pub normal_brush: Brush,
    pub hovered_brush: Brush,
    pub pressed_brush: Brush,
    pub disabled_brush: Brush,
    pub text: String,
    pub text_color: Color,
    pub font_size: f32,
    pub font_id: u32,
    pub padding: Padding,
    pub corner_radii: CornerRadii,
    pub border: BorderSpec,
    pub min_size: Vec2,
    pub enabled: bool,
    pub clicked: bool,
    pub click_count: u32,
    pub tooltip_text: String,
    pub visible: bool,
    pub opacity: f32,
    pub content_desired_size: Vec2,
    pub focus_brush: Brush,
    pub is_focused: bool,
}

impl SlateButton {
    pub fn new(text: &str) -> Self {
        Self {
            id: UIId::INVALID,
            state: ButtonState::Normal,
            normal_brush: Brush::SolidColor(Color::from_hex("#3A3A3A")),
            hovered_brush: Brush::SolidColor(Color::from_hex("#4A4A4A")),
            pressed_brush: Brush::SolidColor(Color::from_hex("#2A2A2A")),
            disabled_brush: Brush::SolidColor(Color::from_hex("#252525")),
            text: text.to_string(),
            text_color: Color::WHITE,
            font_size: 14.0,
            font_id: 0,
            padding: Padding::new(12.0, 6.0, 12.0, 6.0),
            corner_radii: CornerRadii::all(4.0),
            border: BorderSpec::new(Color::from_hex("#555555"), 1.0),
            min_size: Vec2::new(40.0, 24.0),
            enabled: true,
            clicked: false,
            click_count: 0,
            tooltip_text: String::new(),
            visible: true,
            opacity: 1.0,
            content_desired_size: Vec2::ZERO,
            focus_brush: Brush::SolidColor(Color::from_hex("#2266CC")),
            is_focused: false,
        }
    }

    pub fn with_brushes(
        mut self,
        normal: Brush,
        hovered: Brush,
        pressed: Brush,
        disabled: Brush,
    ) -> Self {
        self.normal_brush = normal;
        self.hovered_brush = hovered;
        self.pressed_brush = pressed;
        self.disabled_brush = disabled;
        self
    }

    pub fn with_text_color(mut self, color: Color) -> Self {
        self.text_color = color;
        self
    }

    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_corner_radii(mut self, radii: CornerRadii) -> Self {
        self.corner_radii = radii;
        self
    }

    pub fn with_min_size(mut self, min: Vec2) -> Self {
        self.min_size = min;
        self
    }

    pub fn with_tooltip(mut self, text: &str) -> Self {
        self.tooltip_text = text.to_string();
        self
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.state = ButtonState::Disabled;
        } else if self.state == ButtonState::Disabled {
            self.state = ButtonState::Normal;
        }
    }

    /// Consume and clear the clicked flag.
    pub fn take_click(&mut self) -> bool {
        let c = self.clicked;
        self.clicked = false;
        c
    }

    fn current_brush(&self) -> &Brush {
        match self.state {
            ButtonState::Normal => &self.normal_brush,
            ButtonState::Hovered => &self.hovered_brush,
            ButtonState::Pressed => &self.pressed_brush,
            ButtonState::Disabled => &self.disabled_brush,
        }
    }
}

impl SlateWidget for SlateButton {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        let text_w = self.text.len() as f32 * self.font_size * 0.6;
        let text_h = self.font_size;
        let content_w = text_w.max(self.content_desired_size.x);
        let content_h = text_h.max(self.content_desired_size.y);

        Vec2::new(
            (content_w + self.padding.left + self.padding.right).max(self.min_size.x),
            (content_h + self.padding.top + self.padding.bottom).max(self.min_size.y),
        )
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible || self.opacity <= 0.0 {
            return;
        }

        // Background.
        self.current_brush()
            .paint_rect(draw_list, rect, self.corner_radii);

        // Border.
        if self.border.width > 0.0 {
            draw_list.draw_rounded_rect(
                rect,
                Color::TRANSPARENT,
                self.corner_radii,
                BorderSpec::new(
                    self.border.color.with_alpha(self.border.color.a * self.opacity),
                    self.border.width,
                ),
            );
        }

        // Focus indicator.
        if self.is_focused {
            let focus_rect = Rect::new(
                Vec2::new(rect.min.x - 2.0, rect.min.y - 2.0),
                Vec2::new(rect.max.x + 2.0, rect.max.y + 2.0),
            );
            draw_list.draw_rounded_rect(
                focus_rect,
                Color::TRANSPARENT,
                CornerRadii::all(self.corner_radii.top_left + 2.0),
                BorderSpec::new(Color::from_hex("#4488FF"), 2.0),
            );
        }

        // Text.
        if !self.text.is_empty() {
            let tc = if self.enabled {
                self.text_color
            } else {
                Color::from_rgba8(128, 128, 128, 180)
            };
            let text_w = self.text.len() as f32 * self.font_size * 0.6;
            let tx = rect.min.x + (rect.width() - text_w) * 0.5;
            let ty = rect.min.y + (rect.height() - self.font_size) * 0.5;

            draw_list.push(DrawCommand::Text {
                text: self.text.clone(),
                position: Vec2::new(tx, ty),
                font_size: self.font_size,
                color: tc.with_alpha(tc.a * self.opacity),
                font_id: self.font_id,
                max_width: Some(rect.width()),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                if rect.contains(*position) {
                    if self.state != ButtonState::Pressed {
                        self.state = ButtonState::Hovered;
                    }
                    EventReply::Handled
                } else {
                    if self.state == ButtonState::Hovered {
                        self.state = ButtonState::Normal;
                    }
                    EventReply::Unhandled
                }
            }
            UIEvent::HoverEnd => {
                if self.state == ButtonState::Hovered {
                    self.state = ButtonState::Normal;
                }
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if rect.contains(*position) {
                    self.state = ButtonState::Pressed;
                    EventReply::Handled.then(EventReply::CaptureMouse)
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::MouseUp {
                position,
                button: MouseButton::Left,
            } => {
                if self.state == ButtonState::Pressed {
                    if rect.contains(*position) {
                        self.clicked = true;
                        self.click_count += 1;
                        self.state = ButtonState::Hovered;
                    } else {
                        self.state = ButtonState::Normal;
                    }
                    EventReply::Handled.then(EventReply::ReleaseMouse)
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::Focus => {
                self.is_focused = true;
                EventReply::Handled
            }
            UIEvent::Blur => {
                self.is_focused = false;
                EventReply::Handled
            }
            UIEvent::KeyInput {
                key: KeyCode::Enter,
                pressed: true,
                ..
            }
            | UIEvent::KeyInput {
                key: KeyCode::Space,
                pressed: true,
                ..
            } => {
                if self.is_focused {
                    self.clicked = true;
                    self.click_count += 1;
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
//  7. CheckBox
// =========================================================================

/// Tri-state check value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckState {
    Unchecked,
    Checked,
    Indeterminate,
}

impl CheckState {
    pub fn toggle(self) -> Self {
        match self {
            CheckState::Unchecked => CheckState::Checked,
            CheckState::Checked => CheckState::Unchecked,
            CheckState::Indeterminate => CheckState::Checked,
        }
    }

    pub fn is_checked(self) -> bool {
        self == CheckState::Checked
    }
}

impl Default for CheckState {
    fn default() -> Self {
        CheckState::Unchecked
    }
}

/// A checkbox with checked/unchecked/indeterminate states and per-state brushes.
#[derive(Debug, Clone)]
pub struct CheckBox {
    pub id: UIId,
    pub check_state: CheckState,
    pub label: String,
    pub font_size: f32,
    pub font_id: u32,
    pub text_color: Color,
    pub box_size: f32,
    pub unchecked_bg: Color,
    pub checked_bg: Color,
    pub indeterminate_bg: Color,
    pub unchecked_border: Color,
    pub checked_border: Color,
    pub check_mark_color: Color,
    pub hovered: bool,
    pub enabled: bool,
    pub corner_radii: CornerRadii,
    pub spacing: f32,
    pub state_changed: bool,
    pub visible: bool,
    pub is_focused: bool,
}

impl CheckBox {
    pub fn new(label: &str) -> Self {
        Self {
            id: UIId::INVALID,
            check_state: CheckState::Unchecked,
            label: label.to_string(),
            font_size: 14.0,
            font_id: 0,
            text_color: Color::WHITE,
            box_size: 18.0,
            unchecked_bg: Color::from_hex("#333333"),
            checked_bg: Color::from_hex("#2266CC"),
            indeterminate_bg: Color::from_hex("#555555"),
            unchecked_border: Color::from_hex("#666666"),
            checked_border: Color::from_hex("#3388FF"),
            check_mark_color: Color::WHITE,
            hovered: false,
            enabled: true,
            corner_radii: CornerRadii::all(3.0),
            spacing: 8.0,
            state_changed: false,
            visible: true,
            is_focused: false,
        }
    }

    pub fn with_state(mut self, state: CheckState) -> Self {
        self.check_state = state;
        self
    }

    pub fn with_checked(mut self, checked: bool) -> Self {
        self.check_state = if checked {
            CheckState::Checked
        } else {
            CheckState::Unchecked
        };
        self
    }

    pub fn is_checked(&self) -> bool {
        self.check_state.is_checked()
    }

    pub fn take_state_changed(&mut self) -> bool {
        let c = self.state_changed;
        self.state_changed = false;
        c
    }

    fn box_rect(&self, origin: Rect) -> Rect {
        let cy = origin.min.y + (origin.height() - self.box_size) * 0.5;
        Rect::new(
            Vec2::new(origin.min.x, cy),
            Vec2::new(origin.min.x + self.box_size, cy + self.box_size),
        )
    }
}

impl SlateWidget for CheckBox {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        let text_w = self.label.len() as f32 * self.font_size * 0.6;
        Vec2::new(
            self.box_size + self.spacing + text_w,
            self.box_size.max(self.font_size * 1.2),
        )
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        let box_rect = self.box_rect(rect);

        // Background.
        let bg = match self.check_state {
            CheckState::Unchecked => {
                if self.hovered {
                    self.unchecked_bg.lighten(0.1)
                } else {
                    self.unchecked_bg
                }
            }
            CheckState::Checked => {
                if self.hovered {
                    self.checked_bg.lighten(0.1)
                } else {
                    self.checked_bg
                }
            }
            CheckState::Indeterminate => {
                if self.hovered {
                    self.indeterminate_bg.lighten(0.1)
                } else {
                    self.indeterminate_bg
                }
            }
        };
        let border_color = match self.check_state {
            CheckState::Checked => self.checked_border,
            _ => self.unchecked_border,
        };

        draw_list.draw_rounded_rect(
            box_rect,
            bg,
            self.corner_radii,
            BorderSpec::new(border_color, 1.0),
        );

        // Check mark.
        match self.check_state {
            CheckState::Checked => {
                let cx = box_rect.min.x + self.box_size * 0.5;
                let cy = box_rect.min.y + self.box_size * 0.5;
                let s = self.box_size * 0.25;
                // Simple check mark as two lines.
                draw_list.draw_line(
                    Vec2::new(cx - s, cy),
                    Vec2::new(cx - s * 0.3, cy + s * 0.7),
                    self.check_mark_color,
                    2.0,
                );
                draw_list.draw_line(
                    Vec2::new(cx - s * 0.3, cy + s * 0.7),
                    Vec2::new(cx + s, cy - s * 0.5),
                    self.check_mark_color,
                    2.0,
                );
            }
            CheckState::Indeterminate => {
                let cx = box_rect.min.x + self.box_size * 0.5;
                let cy = box_rect.min.y + self.box_size * 0.5;
                let s = self.box_size * 0.3;
                draw_list.draw_line(
                    Vec2::new(cx - s, cy),
                    Vec2::new(cx + s, cy),
                    self.check_mark_color,
                    2.0,
                );
            }
            CheckState::Unchecked => {}
        }

        // Focus ring.
        if self.is_focused {
            let focus_rect = Rect::new(
                Vec2::new(box_rect.min.x - 2.0, box_rect.min.y - 2.0),
                Vec2::new(box_rect.max.x + 2.0, box_rect.max.y + 2.0),
            );
            draw_list.draw_rounded_rect(
                focus_rect,
                Color::TRANSPARENT,
                CornerRadii::all(self.corner_radii.top_left + 2.0),
                BorderSpec::new(Color::from_hex("#4488FF"), 2.0),
            );
        }

        // Label.
        if !self.label.is_empty() {
            let lx = box_rect.max.x + self.spacing;
            let ly = rect.min.y + (rect.height() - self.font_size) * 0.5;
            let tc = if self.enabled {
                self.text_color
            } else {
                Color::from_rgba8(128, 128, 128, 180)
            };
            draw_list.push(DrawCommand::Text {
                text: self.label.clone(),
                position: Vec2::new(lx, ly),
                font_size: self.font_size,
                color: tc,
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                self.hovered = rect.contains(*position);
                EventReply::Handled
            }
            UIEvent::HoverEnd => {
                self.hovered = false;
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if rect.contains(*position) {
                    self.check_state = self.check_state.toggle();
                    self.state_changed = true;
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::Focus => {
                self.is_focused = true;
                EventReply::Handled
            }
            UIEvent::Blur => {
                self.is_focused = false;
                EventReply::Handled
            }
            UIEvent::KeyInput {
                key: KeyCode::Space,
                pressed: true,
                ..
            } => {
                if self.is_focused {
                    self.check_state = self.check_state.toggle();
                    self.state_changed = true;
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
//  8. Slider
// =========================================================================

/// Horizontal or vertical slider with track and thumb.
#[derive(Debug, Clone)]
pub struct Slider {
    pub id: UIId,
    pub value: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub step: f32,
    pub orientation: Orientation,
    pub track_brush: Brush,
    pub filled_track_brush: Brush,
    pub thumb_brush: Brush,
    pub thumb_hovered_brush: Brush,
    pub track_height: f32,
    pub thumb_radius: f32,
    pub enabled: bool,
    pub dragging: bool,
    pub hovered: bool,
    pub thumb_hovered: bool,
    pub value_changed: bool,
    pub visible: bool,
    pub desired_length: f32,
    pub show_value_tooltip: bool,
    pub is_focused: bool,
}

impl Slider {
    pub fn new(min: f32, max: f32) -> Self {
        Self {
            id: UIId::INVALID,
            value: min,
            min_value: min,
            max_value: max,
            step: 0.0,
            orientation: Orientation::Horizontal,
            track_brush: Brush::SolidColor(Color::from_hex("#333333")),
            filled_track_brush: Brush::SolidColor(Color::from_hex("#2266CC")),
            thumb_brush: Brush::SolidColor(Color::WHITE),
            thumb_hovered_brush: Brush::SolidColor(Color::from_hex("#DDDDFF")),
            track_height: 4.0,
            thumb_radius: 8.0,
            enabled: true,
            dragging: false,
            hovered: false,
            thumb_hovered: false,
            value_changed: false,
            visible: true,
            desired_length: 200.0,
            show_value_tooltip: true,
            is_focused: false,
        }
    }

    pub fn with_value(mut self, v: f32) -> Self {
        self.value = v.clamp(self.min_value, self.max_value);
        self
    }

    pub fn with_step(mut self, s: f32) -> Self {
        self.step = s;
        self
    }

    pub fn with_orientation(mut self, o: Orientation) -> Self {
        self.orientation = o;
        self
    }

    pub fn normalized_value(&self) -> f32 {
        let range = self.max_value - self.min_value;
        if range.abs() < 1e-10 {
            0.0
        } else {
            ((self.value - self.min_value) / range).clamp(0.0, 1.0)
        }
    }

    pub fn set_value(&mut self, v: f32) {
        let old = self.value;
        let mut new_val = v.clamp(self.min_value, self.max_value);
        if self.step > 0.0 {
            new_val = ((new_val - self.min_value) / self.step).round() * self.step + self.min_value;
            new_val = new_val.clamp(self.min_value, self.max_value);
        }
        self.value = new_val;
        self.value_changed = (self.value - old).abs() > 1e-10;
    }

    pub fn take_value_changed(&mut self) -> bool {
        let c = self.value_changed;
        self.value_changed = false;
        c
    }

    fn value_from_position(&self, pos: Vec2, rect: Rect) -> f32 {
        let t = match self.orientation {
            Orientation::Horizontal => {
                let track_x = rect.min.x + self.thumb_radius;
                let track_w = rect.width() - self.thumb_radius * 2.0;
                if track_w <= 0.0 {
                    0.0
                } else {
                    ((pos.x - track_x) / track_w).clamp(0.0, 1.0)
                }
            }
            Orientation::Vertical => {
                let track_y = rect.min.y + self.thumb_radius;
                let track_h = rect.height() - self.thumb_radius * 2.0;
                if track_h <= 0.0 {
                    0.0
                } else {
                    1.0 - ((pos.y - track_y) / track_h).clamp(0.0, 1.0)
                }
            }
        };
        self.min_value + t * (self.max_value - self.min_value)
    }

    fn thumb_center(&self, rect: Rect) -> Vec2 {
        let t = self.normalized_value();
        match self.orientation {
            Orientation::Horizontal => {
                let track_x = rect.min.x + self.thumb_radius;
                let track_w = rect.width() - self.thumb_radius * 2.0;
                let cy = (rect.min.y + rect.max.y) * 0.5;
                Vec2::new(track_x + t * track_w, cy)
            }
            Orientation::Vertical => {
                let track_y = rect.min.y + self.thumb_radius;
                let track_h = rect.height() - self.thumb_radius * 2.0;
                let cx = (rect.min.x + rect.max.x) * 0.5;
                Vec2::new(cx, track_y + (1.0 - t) * track_h)
            }
        }
    }
}

impl SlateWidget for Slider {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        match self.orientation {
            Orientation::Horizontal => {
                Vec2::new(self.desired_length, self.thumb_radius * 2.0 + 4.0)
            }
            Orientation::Vertical => {
                Vec2::new(self.thumb_radius * 2.0 + 4.0, self.desired_length)
            }
        }
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        let t = self.normalized_value();
        let thumb_center = self.thumb_center(rect);

        match self.orientation {
            Orientation::Horizontal => {
                let cy = (rect.min.y + rect.max.y) * 0.5;
                let track_left = rect.min.x + self.thumb_radius;
                let track_right = rect.max.x - self.thumb_radius;
                let half_h = self.track_height * 0.5;

                // Full track background.
                let track_rect = Rect::new(
                    Vec2::new(track_left, cy - half_h),
                    Vec2::new(track_right, cy + half_h),
                );
                self.track_brush
                    .paint_rect(draw_list, track_rect, CornerRadii::all(half_h));

                // Filled portion.
                let filled_right = track_left + t * (track_right - track_left);
                if filled_right > track_left + 0.5 {
                    let filled_rect = Rect::new(
                        Vec2::new(track_left, cy - half_h),
                        Vec2::new(filled_right, cy + half_h),
                    );
                    self.filled_track_brush
                        .paint_rect(draw_list, filled_rect, CornerRadii::all(half_h));
                }
            }
            Orientation::Vertical => {
                let cx = (rect.min.x + rect.max.x) * 0.5;
                let track_top = rect.min.y + self.thumb_radius;
                let track_bottom = rect.max.y - self.thumb_radius;
                let half_w = self.track_height * 0.5;

                let track_rect = Rect::new(
                    Vec2::new(cx - half_w, track_top),
                    Vec2::new(cx + half_w, track_bottom),
                );
                self.track_brush
                    .paint_rect(draw_list, track_rect, CornerRadii::all(half_w));

                let filled_top = track_top + (1.0 - t) * (track_bottom - track_top);
                if track_bottom > filled_top + 0.5 {
                    let filled_rect = Rect::new(
                        Vec2::new(cx - half_w, filled_top),
                        Vec2::new(cx + half_w, track_bottom),
                    );
                    self.filled_track_brush
                        .paint_rect(draw_list, filled_rect, CornerRadii::all(half_w));
                }
            }
        }

        // Thumb.
        let thumb_color = if self.dragging || self.thumb_hovered {
            match &self.thumb_hovered_brush {
                Brush::SolidColor(c) => *c,
                _ => Color::WHITE,
            }
        } else {
            match &self.thumb_brush {
                Brush::SolidColor(c) => *c,
                _ => Color::WHITE,
            }
        };

        draw_list.push(DrawCommand::Circle {
            center: thumb_center,
            radius: self.thumb_radius,
            color: thumb_color,
            border: if self.is_focused {
                BorderSpec::new(Color::from_hex("#4488FF"), 2.0)
            } else {
                BorderSpec::default()
            },
        });

        // Value tooltip while dragging.
        if self.dragging && self.show_value_tooltip {
            let tooltip_text = format!("{:.2}", self.value);
            let tooltip_w = tooltip_text.len() as f32 * 8.0 + 8.0;
            let tooltip_h = 20.0;
            let tooltip_rect = Rect::new(
                Vec2::new(
                    thumb_center.x - tooltip_w * 0.5,
                    thumb_center.y - self.thumb_radius - tooltip_h - 4.0,
                ),
                Vec2::new(
                    thumb_center.x + tooltip_w * 0.5,
                    thumb_center.y - self.thumb_radius - 4.0,
                ),
            );
            draw_list.draw_rounded_rect(
                tooltip_rect,
                Color::from_hex("#222222"),
                CornerRadii::all(4.0),
                BorderSpec::new(Color::from_hex("#555555"), 1.0),
            );
            draw_list.push(DrawCommand::Text {
                text: tooltip_text,
                position: Vec2::new(tooltip_rect.min.x + 4.0, tooltip_rect.min.y + 2.0),
                font_size: 12.0,
                color: Color::WHITE,
                font_id: 0,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                self.hovered = rect.contains(*position);
                let tc = self.thumb_center(rect);
                let dist = (*position - tc).length();
                self.thumb_hovered = dist <= self.thumb_radius + 4.0;
                EventReply::Handled
            }
            UIEvent::HoverEnd => {
                self.hovered = false;
                self.thumb_hovered = false;
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if rect.contains(*position) {
                    self.dragging = true;
                    let v = self.value_from_position(*position, rect);
                    self.set_value(v);
                    EventReply::Handled.then(EventReply::CaptureMouse)
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::DragMove { position, .. } => {
                if self.dragging {
                    let v = self.value_from_position(*position, rect);
                    self.set_value(v);
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::MouseUp {
                button: MouseButton::Left,
                ..
            }
            | UIEvent::DragEnd { .. } => {
                if self.dragging {
                    self.dragging = false;
                    EventReply::Handled.then(EventReply::ReleaseMouse)
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::Focus => {
                self.is_focused = true;
                EventReply::Handled
            }
            UIEvent::Blur => {
                self.is_focused = false;
                EventReply::Handled
            }
            UIEvent::KeyInput {
                key: KeyCode::ArrowLeft,
                pressed: true,
                ..
            }
            | UIEvent::KeyInput {
                key: KeyCode::ArrowDown,
                pressed: true,
                ..
            } => {
                if self.is_focused {
                    let step = if self.step > 0.0 {
                        self.step
                    } else {
                        (self.max_value - self.min_value) * 0.01
                    };
                    self.set_value(self.value - step);
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::KeyInput {
                key: KeyCode::ArrowRight,
                pressed: true,
                ..
            }
            | UIEvent::KeyInput {
                key: KeyCode::ArrowUp,
                pressed: true,
                ..
            } => {
                if self.is_focused {
                    let step = if self.step > 0.0 {
                        self.step
                    } else {
                        (self.max_value - self.min_value) * 0.01
                    };
                    self.set_value(self.value + step);
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
//  9. SpinBox
// =========================================================================

/// Numeric spin box with drag-to-adjust and text input.
#[derive(Debug, Clone)]
pub struct SpinBox {
    pub id: UIId,
    pub value: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub step: f64,
    pub precision: usize,
    pub font_size: f32,
    pub font_id: u32,
    pub text_color: Color,
    pub background_color: Color,
    pub border_color: Color,
    pub hovered_border_color: Color,
    pub focused_border_color: Color,
    pub button_color: Color,
    pub button_hovered_color: Color,
    pub corner_radii: CornerRadii,
    pub enabled: bool,
    pub editing: bool,
    pub edit_text: String,
    pub dragging: bool,
    pub drag_start_value: f64,
    pub drag_start_y: f32,
    pub drag_sensitivity: f64,
    pub hovered: bool,
    pub value_changed: bool,
    pub width: f32,
    pub height: f32,
    pub visible: bool,
    pub is_focused: bool,
    pub cursor_pos: usize,
}

impl SpinBox {
    pub fn new(value: f64) -> Self {
        Self {
            id: UIId::INVALID,
            value,
            min_value: f64::MIN,
            max_value: f64::MAX,
            step: 1.0,
            precision: 2,
            font_size: 13.0,
            font_id: 0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#2A2A2A"),
            border_color: Color::from_hex("#555555"),
            hovered_border_color: Color::from_hex("#777777"),
            focused_border_color: Color::from_hex("#4488FF"),
            button_color: Color::from_hex("#444444"),
            button_hovered_color: Color::from_hex("#555555"),
            corner_radii: CornerRadii::all(3.0),
            enabled: true,
            editing: false,
            edit_text: String::new(),
            dragging: false,
            drag_start_value: 0.0,
            drag_start_y: 0.0,
            drag_sensitivity: 0.5,
            hovered: false,
            value_changed: false,
            width: 80.0,
            height: 24.0,
            visible: true,
            is_focused: false,
            cursor_pos: 0,
        }
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = min;
        self.max_value = max;
        self.value = self.value.clamp(min, max);
        self
    }

    pub fn with_step(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    pub fn with_precision(mut self, p: usize) -> Self {
        self.precision = p;
        self
    }

    pub fn set_value(&mut self, v: f64) {
        let old = self.value;
        self.value = v.clamp(self.min_value, self.max_value);
        self.value_changed = (self.value - old).abs() > 1e-15;
    }

    pub fn take_value_changed(&mut self) -> bool {
        let c = self.value_changed;
        self.value_changed = false;
        c
    }

    fn display_text(&self) -> String {
        if self.editing {
            self.edit_text.clone()
        } else {
            format!("{:.*}", self.precision, self.value)
        }
    }

    fn begin_editing(&mut self) {
        self.editing = true;
        self.edit_text = format!("{:.*}", self.precision, self.value);
        self.cursor_pos = self.edit_text.len();
    }

    fn commit_edit(&mut self) {
        if self.editing {
            if let Ok(v) = self.edit_text.parse::<f64>() {
                self.set_value(v);
            }
            self.editing = false;
            self.edit_text.clear();
        }
    }

    fn cancel_edit(&mut self) {
        self.editing = false;
        self.edit_text.clear();
    }

    fn increment_button_rect(&self, rect: Rect) -> Rect {
        let btn_w = 16.0;
        Rect::new(
            Vec2::new(rect.max.x - btn_w, rect.min.y),
            Vec2::new(rect.max.x, rect.min.y + rect.height() * 0.5),
        )
    }

    fn decrement_button_rect(&self, rect: Rect) -> Rect {
        let btn_w = 16.0;
        Rect::new(
            Vec2::new(rect.max.x - btn_w, rect.min.y + rect.height() * 0.5),
            Vec2::new(rect.max.x, rect.max.y),
        )
    }
}

impl SlateWidget for SpinBox {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        Vec2::new(self.width, self.height)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        let border_c = if self.is_focused || self.editing {
            self.focused_border_color
        } else if self.hovered {
            self.hovered_border_color
        } else {
            self.border_color
        };

        // Background.
        draw_list.draw_rounded_rect(
            rect,
            self.background_color,
            self.corner_radii,
            BorderSpec::new(border_c, 1.0),
        );

        // Text.
        let text = self.display_text();
        let text_x = rect.min.x + 4.0;
        let text_y = rect.min.y + (rect.height() - self.font_size) * 0.5;
        let tc = if self.enabled {
            self.text_color
        } else {
            Color::from_rgba8(128, 128, 128, 180)
        };
        draw_list.push(DrawCommand::Text {
            text: text.clone(),
            position: Vec2::new(text_x, text_y),
            font_size: self.font_size,
            color: tc,
            font_id: self.font_id,
            max_width: Some(rect.width() - 20.0),
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });

        // Cursor when editing.
        if self.editing {
            let cursor_x =
                text_x + self.cursor_pos.min(text.len()) as f32 * self.font_size * 0.6;
            draw_list.draw_line(
                Vec2::new(cursor_x, text_y),
                Vec2::new(cursor_x, text_y + self.font_size),
                Color::WHITE,
                1.0,
            );
        }

        // Increment/decrement buttons.
        let inc_rect = self.increment_button_rect(rect);
        let dec_rect = self.decrement_button_rect(rect);

        draw_list.draw_rect(inc_rect, self.button_color);
        draw_list.draw_rect(dec_rect, self.button_color);

        // Up arrow.
        let icx = (inc_rect.min.x + inc_rect.max.x) * 0.5;
        let icy = (inc_rect.min.y + inc_rect.max.y) * 0.5;
        draw_list.push(DrawCommand::Triangle {
            p0: Vec2::new(icx, icy - 3.0),
            p1: Vec2::new(icx - 4.0, icy + 2.0),
            p2: Vec2::new(icx + 4.0, icy + 2.0),
            color: tc,
        });

        // Down arrow.
        let dcx = (dec_rect.min.x + dec_rect.max.x) * 0.5;
        let dcy = (dec_rect.min.y + dec_rect.max.y) * 0.5;
        draw_list.push(DrawCommand::Triangle {
            p0: Vec2::new(dcx, dcy + 3.0),
            p1: Vec2::new(dcx - 4.0, dcy - 2.0),
            p2: Vec2::new(dcx + 4.0, dcy - 2.0),
            color: tc,
        });

        // Separator line between buttons.
        draw_list.draw_line(
            Vec2::new(inc_rect.min.x, inc_rect.max.y),
            Vec2::new(inc_rect.max.x, inc_rect.max.y),
            self.border_color,
            1.0,
        );
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                self.hovered = rect.contains(*position);
                EventReply::Handled
            }
            UIEvent::HoverEnd => {
                self.hovered = false;
                EventReply::Handled
            }
            UIEvent::DoubleClick {
                position,
                button: MouseButton::Left,
            } => {
                if rect.contains(*position) {
                    self.begin_editing();
                    return EventReply::Handled.then(EventReply::SetFocus);
                }
                EventReply::Unhandled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if !rect.contains(*position) {
                    if self.editing {
                        self.commit_edit();
                    }
                    return EventReply::Unhandled;
                }

                // Check increment/decrement buttons.
                let inc = self.increment_button_rect(rect);
                let dec = self.decrement_button_rect(rect);
                if inc.contains(*position) {
                    self.set_value(self.value + self.step);
                    return EventReply::Handled;
                }
                if dec.contains(*position) {
                    self.set_value(self.value - self.step);
                    return EventReply::Handled;
                }

                // Start drag-to-adjust.
                if !self.editing {
                    self.dragging = true;
                    self.drag_start_value = self.value;
                    self.drag_start_y = position.y;
                    return EventReply::Handled.then(EventReply::CaptureMouse);
                }

                EventReply::Handled
            }
            UIEvent::DragMove { position, delta } => {
                if self.dragging {
                    let dy = self.drag_start_y - position.y;
                    let new_val =
                        self.drag_start_value + dy as f64 * self.step * self.drag_sensitivity;
                    self.set_value(new_val);
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::MouseUp {
                button: MouseButton::Left,
                ..
            }
            | UIEvent::DragEnd { .. } => {
                if self.dragging {
                    self.dragging = false;
                    return EventReply::Handled.then(EventReply::ReleaseMouse);
                }
                EventReply::Unhandled
            }
            UIEvent::TextInput { character } => {
                if self.editing {
                    if character.is_ascii_digit()
                        || *character == '.'
                        || *character == '-'
                        || *character == 'e'
                        || *character == 'E'
                    {
                        self.edit_text.insert(self.cursor_pos, *character);
                        self.cursor_pos += 1;
                    }
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::KeyInput {
                key, pressed: true, ..
            } => {
                if self.editing {
                    match key {
                        KeyCode::Enter => {
                            self.commit_edit();
                            return EventReply::Handled.then(EventReply::ClearFocus);
                        }
                        KeyCode::Escape => {
                            self.cancel_edit();
                            return EventReply::Handled.then(EventReply::ClearFocus);
                        }
                        KeyCode::Backspace => {
                            if self.cursor_pos > 0 {
                                self.cursor_pos -= 1;
                                self.edit_text.remove(self.cursor_pos);
                            }
                            return EventReply::Handled;
                        }
                        KeyCode::Delete => {
                            if self.cursor_pos < self.edit_text.len() {
                                self.edit_text.remove(self.cursor_pos);
                            }
                            return EventReply::Handled;
                        }
                        KeyCode::ArrowLeft => {
                            if self.cursor_pos > 0 {
                                self.cursor_pos -= 1;
                            }
                            return EventReply::Handled;
                        }
                        KeyCode::ArrowRight => {
                            if self.cursor_pos < self.edit_text.len() {
                                self.cursor_pos += 1;
                            }
                            return EventReply::Handled;
                        }
                        KeyCode::Home => {
                            self.cursor_pos = 0;
                            return EventReply::Handled;
                        }
                        KeyCode::End => {
                            self.cursor_pos = self.edit_text.len();
                            return EventReply::Handled;
                        }
                        _ => {}
                    }
                }
                EventReply::Unhandled
            }
            UIEvent::Focus => {
                self.is_focused = true;
                EventReply::Handled
            }
            UIEvent::Blur => {
                self.is_focused = false;
                if self.editing {
                    self.commit_edit();
                }
                EventReply::Handled
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 10. EditableTextBox
// =========================================================================

/// Single-line text input with hint text, selection, and cursor blink.
#[derive(Debug, Clone)]
pub struct EditableTextBox {
    pub id: UIId,
    pub text: String,
    pub hint_text: String,
    pub font_size: f32,
    pub font_id: u32,
    pub text_color: Color,
    pub hint_color: Color,
    pub background_color: Color,
    pub border_color: Color,
    pub focused_border_color: Color,
    pub selection_color: Color,
    pub cursor_color: Color,
    pub corner_radii: CornerRadii,
    pub padding: Padding,
    pub enabled: bool,
    pub read_only: bool,
    pub is_focused: bool,
    pub cursor_pos: usize,
    pub selection_start: Option<usize>,
    pub cursor_blink_timer: f32,
    pub cursor_visible: bool,
    pub scroll_offset: f32,
    pub max_length: Option<usize>,
    pub password_mode: bool,
    pub password_char: char,
    pub text_changed: bool,
    pub committed: bool,
    pub desired_width: f32,
    pub desired_height: f32,
    pub visible: bool,
    pub hovered: bool,
    /// Validation: if set, input is rejected when this returns false.
    pub validation_error: Option<String>,
}

impl EditableTextBox {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            text: String::new(),
            hint_text: String::new(),
            font_size: 14.0,
            font_id: 0,
            text_color: Color::WHITE,
            hint_color: Color::from_rgba8(128, 128, 128, 180),
            background_color: Color::from_hex("#2A2A2A"),
            border_color: Color::from_hex("#555555"),
            focused_border_color: Color::from_hex("#4488FF"),
            selection_color: Color::from_hex("#2266CC80"),
            cursor_color: Color::WHITE,
            corner_radii: CornerRadii::all(3.0),
            padding: Padding::new(6.0, 4.0, 6.0, 4.0),
            enabled: true,
            read_only: false,
            is_focused: false,
            cursor_pos: 0,
            selection_start: None,
            cursor_blink_timer: 0.0,
            cursor_visible: true,
            scroll_offset: 0.0,
            max_length: None,
            password_mode: false,
            password_char: '*',
            text_changed: false,
            committed: false,
            desired_width: 200.0,
            desired_height: 28.0,
            visible: true,
            hovered: false,
            validation_error: None,
        }
    }

    pub fn with_text(mut self, text: &str) -> Self {
        self.text = text.to_string();
        self.cursor_pos = text.len();
        self
    }

    pub fn with_hint(mut self, hint: &str) -> Self {
        self.hint_text = hint.to_string();
        self
    }

    pub fn with_password(mut self) -> Self {
        self.password_mode = true;
        self
    }

    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = Some(max);
        self
    }

    pub fn with_width(mut self, w: f32) -> Self {
        self.desired_width = w;
        self
    }

    pub fn take_text_changed(&mut self) -> bool {
        let c = self.text_changed;
        self.text_changed = false;
        c
    }

    pub fn take_committed(&mut self) -> bool {
        let c = self.committed;
        self.committed = false;
        c
    }

    fn display_text(&self) -> String {
        if self.password_mode {
            self.password_char.to_string().repeat(self.text.len())
        } else {
            self.text.clone()
        }
    }

    fn char_width(&self) -> f32 {
        self.font_size * 0.6
    }

    fn content_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            Vec2::new(rect.min.x + self.padding.left, rect.min.y + self.padding.top),
            Vec2::new(
                rect.max.x - self.padding.right,
                rect.max.y - self.padding.bottom,
            ),
        )
    }

    fn selection_range(&self) -> Option<(usize, usize)> {
        self.selection_start.map(|start| {
            let s = start.min(self.cursor_pos);
            let e = start.max(self.cursor_pos);
            (s, e)
        })
    }

    fn selected_text(&self) -> Option<String> {
        self.selection_range().map(|(s, e)| {
            self.text.chars().skip(s).take(e - s).collect()
        })
    }

    fn delete_selection(&mut self) {
        if let Some((s, e)) = self.selection_range() {
            let before: String = self.text.chars().take(s).collect();
            let after: String = self.text.chars().skip(e).collect();
            self.text = format!("{}{}", before, after);
            self.cursor_pos = s;
            self.selection_start = None;
            self.text_changed = true;
        }
    }

    fn insert_text(&mut self, s: &str) {
        self.delete_selection();
        if let Some(max) = self.max_length {
            if self.text.len() + s.len() > max {
                return;
            }
        }
        let before: String = self.text.chars().take(self.cursor_pos).collect();
        let after: String = self.text.chars().skip(self.cursor_pos).collect();
        self.text = format!("{}{}{}", before, s, after);
        self.cursor_pos += s.chars().count();
        self.text_changed = true;
    }

    fn select_all(&mut self) {
        self.selection_start = Some(0);
        self.cursor_pos = self.text.chars().count();
    }

    fn select_word_at(&mut self, pos: usize) {
        let chars: Vec<char> = self.text.chars().collect();
        let len = chars.len();
        if pos >= len {
            return;
        }

        let mut start = pos;
        while start > 0 && chars[start - 1].is_alphanumeric() {
            start -= 1;
        }
        let mut end = pos;
        while end < len && chars[end].is_alphanumeric() {
            end += 1;
        }
        self.selection_start = Some(start);
        self.cursor_pos = end;
    }

    fn pos_from_x(&self, x: f32, rect: Rect) -> usize {
        let content = self.content_rect(rect);
        let local_x = x - content.min.x + self.scroll_offset;
        let cw = self.char_width();
        let pos = (local_x / cw).round() as usize;
        pos.min(self.text.chars().count())
    }

    /// Update cursor blink timer (call each frame with dt).
    pub fn tick(&mut self, dt: f32) {
        if self.is_focused {
            self.cursor_blink_timer += dt;
            // 530ms on, 530ms off.
            let cycle = 1.06;
            let phase = self.cursor_blink_timer % cycle;
            self.cursor_visible = phase < 0.53;
        }
    }

    fn ensure_cursor_visible(&mut self, rect: Rect) {
        let content = self.content_rect(rect);
        let cw = self.char_width();
        let cursor_x = self.cursor_pos as f32 * cw - self.scroll_offset;
        let content_w = content.width();

        if cursor_x < 0.0 {
            self.scroll_offset += cursor_x;
        } else if cursor_x > content_w {
            self.scroll_offset += cursor_x - content_w;
        }
        self.scroll_offset = self.scroll_offset.max(0.0);
    }
}

impl SlateWidget for EditableTextBox {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        Vec2::new(self.desired_width, self.desired_height)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        let border_c = if self.is_focused {
            self.focused_border_color
        } else if self.hovered {
            self.border_color.lighten(0.1)
        } else {
            self.border_color
        };

        // Validation error border.
        let border_c = if self.validation_error.is_some() {
            Color::from_hex("#FF4444")
        } else {
            border_c
        };

        draw_list.draw_rounded_rect(
            rect,
            self.background_color,
            self.corner_radii,
            BorderSpec::new(border_c, 1.0),
        );

        let content = self.content_rect(rect);
        draw_list.push_clip(content);

        let display = self.display_text();
        let cw = self.char_width();

        // Selection highlight.
        if let Some((sel_start, sel_end)) = self.selection_range() {
            let sx = content.min.x + sel_start as f32 * cw - self.scroll_offset;
            let ex = content.min.x + sel_end as f32 * cw - self.scroll_offset;
            let sel_rect = Rect::new(
                Vec2::new(sx, content.min.y),
                Vec2::new(ex, content.max.y),
            );
            draw_list.draw_rect(sel_rect, self.selection_color);
        }

        if display.is_empty() && !self.hint_text.is_empty() && !self.is_focused {
            // Hint text.
            draw_list.push(DrawCommand::Text {
                text: self.hint_text.clone(),
                position: Vec2::new(
                    content.min.x,
                    content.min.y + (content.height() - self.font_size) * 0.5,
                ),
                font_size: self.font_size,
                color: self.hint_color,
                font_id: self.font_id,
                max_width: Some(content.width()),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        } else {
            let ty = content.min.y + (content.height() - self.font_size) * 0.5;
            draw_list.push(DrawCommand::Text {
                text: display,
                position: Vec2::new(content.min.x - self.scroll_offset, ty),
                font_size: self.font_size,
                color: if self.enabled {
                    self.text_color
                } else {
                    Color::from_rgba8(128, 128, 128, 180)
                },
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }

        // Cursor.
        if self.is_focused && self.cursor_visible {
            let cursor_x = content.min.x + self.cursor_pos as f32 * cw - self.scroll_offset;
            let ty = content.min.y + (content.height() - self.font_size) * 0.5;
            draw_list.draw_line(
                Vec2::new(cursor_x, ty),
                Vec2::new(cursor_x, ty + self.font_size),
                self.cursor_color,
                1.0,
            );
        }

        draw_list.pop_clip();

        // Validation error text.
        if let Some(ref err) = self.validation_error {
            draw_list.push(DrawCommand::Text {
                text: err.clone(),
                position: Vec2::new(rect.min.x, rect.max.y + 2.0),
                font_size: 11.0,
                color: Color::from_hex("#FF4444"),
                font_id: self.font_id,
                max_width: Some(rect.width()),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                self.hovered = rect.contains(*position);
                EventReply::Handled
            }
            UIEvent::HoverEnd => {
                self.hovered = false;
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                modifiers,
            } => {
                if rect.contains(*position) {
                    let pos = self.pos_from_x(position.x, rect);
                    if modifiers.shift {
                        if self.selection_start.is_none() {
                            self.selection_start = Some(self.cursor_pos);
                        }
                        self.cursor_pos = pos;
                    } else {
                        self.cursor_pos = pos;
                        self.selection_start = None;
                    }
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                    self.ensure_cursor_visible(rect);
                    EventReply::Handled.then(EventReply::SetFocus)
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::DoubleClick {
                position,
                button: MouseButton::Left,
            } => {
                if rect.contains(*position) {
                    let pos = self.pos_from_x(position.x, rect);
                    self.select_word_at(pos);
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::Focus => {
                self.is_focused = true;
                self.cursor_blink_timer = 0.0;
                self.cursor_visible = true;
                EventReply::Handled
            }
            UIEvent::Blur => {
                self.is_focused = false;
                self.selection_start = None;
                EventReply::Handled
            }
            UIEvent::TextInput { character } => {
                if self.is_focused && !self.read_only {
                    if !character.is_control() {
                        self.insert_text(&character.to_string());
                        self.cursor_blink_timer = 0.0;
                        self.cursor_visible = true;
                        self.ensure_cursor_visible(rect);
                    }
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::KeyInput {
                key,
                pressed: true,
                modifiers,
            } => {
                if !self.is_focused {
                    return EventReply::Unhandled;
                }

                let text_len = self.text.chars().count();

                match key {
                    KeyCode::ArrowLeft => {
                        if modifiers.shift {
                            if self.selection_start.is_none() {
                                self.selection_start = Some(self.cursor_pos);
                            }
                        } else {
                            self.selection_start = None;
                        }
                        if modifiers.ctrl {
                            // Move word left.
                            let chars: Vec<char> = self.text.chars().collect();
                            let mut p = self.cursor_pos;
                            while p > 0 && !chars[p - 1].is_alphanumeric() {
                                p -= 1;
                            }
                            while p > 0 && chars[p - 1].is_alphanumeric() {
                                p -= 1;
                            }
                            self.cursor_pos = p;
                        } else if self.cursor_pos > 0 {
                            self.cursor_pos -= 1;
                        }
                        self.cursor_blink_timer = 0.0;
                        self.cursor_visible = true;
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::ArrowRight => {
                        if modifiers.shift {
                            if self.selection_start.is_none() {
                                self.selection_start = Some(self.cursor_pos);
                            }
                        } else {
                            self.selection_start = None;
                        }
                        if modifiers.ctrl {
                            let chars: Vec<char> = self.text.chars().collect();
                            let len = chars.len();
                            let mut p = self.cursor_pos;
                            while p < len && chars[p].is_alphanumeric() {
                                p += 1;
                            }
                            while p < len && !chars[p].is_alphanumeric() {
                                p += 1;
                            }
                            self.cursor_pos = p;
                        } else if self.cursor_pos < text_len {
                            self.cursor_pos += 1;
                        }
                        self.cursor_blink_timer = 0.0;
                        self.cursor_visible = true;
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::Home => {
                        if modifiers.shift {
                            if self.selection_start.is_none() {
                                self.selection_start = Some(self.cursor_pos);
                            }
                        } else {
                            self.selection_start = None;
                        }
                        self.cursor_pos = 0;
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::End => {
                        if modifiers.shift {
                            if self.selection_start.is_none() {
                                self.selection_start = Some(self.cursor_pos);
                            }
                        } else {
                            self.selection_start = None;
                        }
                        self.cursor_pos = text_len;
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::Backspace => {
                        if self.read_only {
                            return EventReply::Handled;
                        }
                        if self.selection_start.is_some() {
                            self.delete_selection();
                        } else if self.cursor_pos > 0 {
                            let before: String =
                                self.text.chars().take(self.cursor_pos - 1).collect();
                            let after: String =
                                self.text.chars().skip(self.cursor_pos).collect();
                            self.text = format!("{}{}", before, after);
                            self.cursor_pos -= 1;
                            self.text_changed = true;
                        }
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::Delete => {
                        if self.read_only {
                            return EventReply::Handled;
                        }
                        if self.selection_start.is_some() {
                            self.delete_selection();
                        } else if self.cursor_pos < text_len {
                            let before: String =
                                self.text.chars().take(self.cursor_pos).collect();
                            let after: String =
                                self.text.chars().skip(self.cursor_pos + 1).collect();
                            self.text = format!("{}{}", before, after);
                            self.text_changed = true;
                        }
                        EventReply::Handled
                    }
                    KeyCode::Enter => {
                        self.committed = true;
                        EventReply::Handled.then(EventReply::ClearFocus)
                    }
                    KeyCode::Escape => {
                        EventReply::Handled.then(EventReply::ClearFocus)
                    }
                    KeyCode::A if modifiers.ctrl => {
                        self.select_all();
                        EventReply::Handled
                    }
                    KeyCode::C if modifiers.ctrl => {
                        // Copy handled by clipboard system.
                        EventReply::Handled
                    }
                    KeyCode::V if modifiers.ctrl => {
                        // Paste handled by clipboard system.
                        EventReply::Handled
                    }
                    KeyCode::X if modifiers.ctrl => {
                        if !self.read_only {
                            self.delete_selection();
                        }
                        EventReply::Handled
                    }
                    _ => EventReply::Unhandled,
                }
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 11. MultiLineEditableText
// =========================================================================

/// Multi-line text editor with scrolling and optional line numbers.
#[derive(Debug, Clone)]
pub struct MultiLineEditableText {
    pub id: UIId,
    pub text: String,
    pub font_size: f32,
    pub font_id: u32,
    pub text_color: Color,
    pub background_color: Color,
    pub border_color: Color,
    pub focused_border_color: Color,
    pub selection_color: Color,
    pub cursor_color: Color,
    pub line_number_color: Color,
    pub line_number_bg: Color,
    pub corner_radii: CornerRadii,
    pub padding: Padding,
    pub enabled: bool,
    pub read_only: bool,
    pub is_focused: bool,
    pub cursor_line: usize,
    pub cursor_col: usize,
    pub selection_start: Option<(usize, usize)>,
    pub cursor_blink_timer: f32,
    pub cursor_visible: bool,
    pub scroll_offset_y: f32,
    pub scroll_offset_x: f32,
    pub show_line_numbers: bool,
    pub line_number_width: f32,
    pub tab_size: usize,
    pub text_changed: bool,
    pub desired_size: Vec2,
    pub visible: bool,
    pub hovered: bool,
    pub word_wrap: bool,
}

impl MultiLineEditableText {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            text: String::new(),
            font_size: 13.0,
            font_id: 0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#1E1E1E"),
            border_color: Color::from_hex("#444444"),
            focused_border_color: Color::from_hex("#4488FF"),
            selection_color: Color::from_hex("#264F78"),
            cursor_color: Color::WHITE,
            line_number_color: Color::from_hex("#858585"),
            line_number_bg: Color::from_hex("#1E1E1E"),
            corner_radii: CornerRadii::all(3.0),
            padding: Padding::new(4.0, 4.0, 4.0, 4.0),
            enabled: true,
            read_only: false,
            is_focused: false,
            cursor_line: 0,
            cursor_col: 0,
            selection_start: None,
            cursor_blink_timer: 0.0,
            cursor_visible: true,
            scroll_offset_y: 0.0,
            scroll_offset_x: 0.0,
            show_line_numbers: true,
            line_number_width: 40.0,
            tab_size: 4,
            text_changed: false,
            desired_size: Vec2::new(400.0, 300.0),
            visible: true,
            hovered: false,
            word_wrap: false,
        }
    }

    pub fn with_text(mut self, text: &str) -> Self {
        self.text = text.to_string();
        self
    }

    pub fn with_show_line_numbers(mut self, show: bool) -> Self {
        self.show_line_numbers = show;
        self
    }

    fn lines(&self) -> Vec<&str> {
        self.text.lines().collect()
    }

    fn line_count(&self) -> usize {
        self.text.lines().count().max(1)
    }

    fn char_width(&self) -> f32 {
        self.font_size * 0.6
    }

    fn line_height(&self) -> f32 {
        self.font_size * 1.3
    }

    fn content_left(&self) -> f32 {
        if self.show_line_numbers {
            self.line_number_width
        } else {
            0.0
        }
    }

    pub fn tick(&mut self, dt: f32) {
        if self.is_focused {
            self.cursor_blink_timer += dt;
            let cycle = 1.06;
            let phase = self.cursor_blink_timer % cycle;
            self.cursor_visible = phase < 0.53;
        }
    }

    fn ensure_cursor_visible(&mut self, rect: Rect) {
        let lh = self.line_height();
        let content_h = rect.height() - self.padding.top - self.padding.bottom;
        let cursor_y = self.cursor_line as f32 * lh;

        if cursor_y < self.scroll_offset_y {
            self.scroll_offset_y = cursor_y;
        } else if cursor_y + lh > self.scroll_offset_y + content_h {
            self.scroll_offset_y = cursor_y + lh - content_h;
        }

        let cw = self.char_width();
        let content_w = rect.width()
            - self.padding.left
            - self.padding.right
            - self.content_left();
        let cursor_x = self.cursor_col as f32 * cw;

        if cursor_x < self.scroll_offset_x {
            self.scroll_offset_x = cursor_x;
        } else if cursor_x + cw > self.scroll_offset_x + content_w {
            self.scroll_offset_x = cursor_x + cw - content_w;
        }
    }

    fn current_line_text(&self) -> String {
        self.text
            .lines()
            .nth(self.cursor_line)
            .unwrap_or("")
            .to_string()
    }

    fn insert_char(&mut self, ch: char) {
        if self.read_only {
            return;
        }
        let mut lines: Vec<String> = self.text.lines().map(String::from).collect();
        if lines.is_empty() {
            lines.push(String::new());
        }

        if ch == '\n' {
            if self.cursor_line < lines.len() {
                let line = &lines[self.cursor_line];
                let col = self.cursor_col.min(line.len());
                let before = line[..col].to_string();
                let after = line[col..].to_string();
                lines[self.cursor_line] = before;
                lines.insert(self.cursor_line + 1, after);
                self.cursor_line += 1;
                self.cursor_col = 0;
            }
        } else if ch == '\t' {
            let spaces = " ".repeat(self.tab_size);
            if self.cursor_line < lines.len() {
                let col = self.cursor_col.min(lines[self.cursor_line].len());
                lines[self.cursor_line].insert_str(col, &spaces);
                self.cursor_col += self.tab_size;
            }
        } else {
            if self.cursor_line < lines.len() {
                let col = self.cursor_col.min(lines[self.cursor_line].len());
                lines[self.cursor_line].insert(col, ch);
                self.cursor_col += 1;
            }
        }

        self.text = lines.join("\n");
        self.text_changed = true;
    }

    fn delete_char_before(&mut self) {
        if self.read_only {
            return;
        }
        let mut lines: Vec<String> = self.text.lines().map(String::from).collect();
        if lines.is_empty() {
            return;
        }

        if self.cursor_col > 0 && self.cursor_line < lines.len() {
            let col = self.cursor_col.min(lines[self.cursor_line].len());
            if col > 0 {
                lines[self.cursor_line].remove(col - 1);
                self.cursor_col -= 1;
            }
        } else if self.cursor_line > 0 {
            let current = lines.remove(self.cursor_line);
            self.cursor_line -= 1;
            self.cursor_col = lines[self.cursor_line].len();
            lines[self.cursor_line].push_str(&current);
        }

        self.text = lines.join("\n");
        self.text_changed = true;
    }

    fn delete_char_after(&mut self) {
        if self.read_only {
            return;
        }
        let mut lines: Vec<String> = self.text.lines().map(String::from).collect();
        if lines.is_empty() {
            return;
        }

        if self.cursor_line < lines.len() {
            let line_len = lines[self.cursor_line].len();
            if self.cursor_col < line_len {
                lines[self.cursor_line].remove(self.cursor_col);
            } else if self.cursor_line + 1 < lines.len() {
                let next = lines.remove(self.cursor_line + 1);
                lines[self.cursor_line].push_str(&next);
            }
        }

        self.text = lines.join("\n");
        self.text_changed = true;
    }
}

impl SlateWidget for MultiLineEditableText {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        self.desired_size
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        let border_c = if self.is_focused {
            self.focused_border_color
        } else {
            self.border_color
        };

        draw_list.draw_rounded_rect(
            rect,
            self.background_color,
            self.corner_radii,
            BorderSpec::new(border_c, 1.0),
        );

        let inner = Rect::new(
            Vec2::new(rect.min.x + self.padding.left, rect.min.y + self.padding.top),
            Vec2::new(
                rect.max.x - self.padding.right,
                rect.max.y - self.padding.bottom,
            ),
        );
        draw_list.push_clip(inner);

        let lh = self.line_height();
        let cw = self.char_width();
        let lines = self.lines();
        let content_left = inner.min.x + self.content_left();

        let first_visible = (self.scroll_offset_y / lh).floor() as usize;
        let visible_count =
            ((inner.height() / lh).ceil() as usize + 1).min(lines.len() - first_visible);

        // Line numbers.
        if self.show_line_numbers {
            let ln_rect = Rect::new(
                Vec2::new(inner.min.x, inner.min.y),
                Vec2::new(inner.min.x + self.line_number_width - 4.0, inner.max.y),
            );
            draw_list.draw_rect(ln_rect, self.line_number_bg);

            for i in 0..visible_count {
                let line_idx = first_visible + i;
                if line_idx >= lines.len() {
                    break;
                }
                let y = inner.min.y + i as f32 * lh - (self.scroll_offset_y % lh);
                let num = format!("{}", line_idx + 1);
                let num_w = num.len() as f32 * cw;
                draw_list.push(DrawCommand::Text {
                    text: num,
                    position: Vec2::new(
                        inner.min.x + self.line_number_width - 8.0 - num_w,
                        y,
                    ),
                    font_size: self.font_size,
                    color: self.line_number_color,
                    font_id: self.font_id,
                    max_width: None,
                    align: TextAlign::Right,
                    vertical_align: TextVerticalAlign::Top,
                });
            }
        }

        // Text lines.
        for i in 0..visible_count {
            let line_idx = first_visible + i;
            if line_idx >= lines.len() {
                break;
            }
            let y = inner.min.y + i as f32 * lh - (self.scroll_offset_y % lh);
            let line = lines[line_idx];

            // Current line highlight.
            if line_idx == self.cursor_line && self.is_focused {
                draw_list.draw_rect(
                    Rect::new(
                        Vec2::new(content_left, y),
                        Vec2::new(inner.max.x, y + lh),
                    ),
                    Color::from_hex("#282828"),
                );
            }

            draw_list.push(DrawCommand::Text {
                text: line.to_string(),
                position: Vec2::new(content_left - self.scroll_offset_x, y),
                font_size: self.font_size,
                color: self.text_color,
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }

        // Cursor.
        if self.is_focused && self.cursor_visible {
            let visible_line = self.cursor_line as i32 - first_visible as i32;
            if visible_line >= 0 {
                let y =
                    inner.min.y + visible_line as f32 * lh - (self.scroll_offset_y % lh);
                let x = content_left + self.cursor_col as f32 * cw - self.scroll_offset_x;
                draw_list.draw_line(
                    Vec2::new(x, y),
                    Vec2::new(x, y + lh),
                    self.cursor_color,
                    1.0,
                );
            }
        }

        draw_list.pop_clip();
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                self.hovered = rect.contains(*position);
                EventReply::Handled
            }
            UIEvent::HoverEnd => {
                self.hovered = false;
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if rect.contains(*position) {
                    let inner_y = position.y - rect.min.y - self.padding.top + self.scroll_offset_y;
                    let lh = self.line_height();
                    self.cursor_line = (inner_y / lh).floor() as usize;
                    self.cursor_line = self.cursor_line.min(self.line_count().saturating_sub(1));

                    let content_left = rect.min.x + self.padding.left + self.content_left();
                    let inner_x = position.x - content_left + self.scroll_offset_x;
                    let cw = self.char_width();
                    self.cursor_col = (inner_x / cw).round().max(0.0) as usize;
                    let line_len = self.current_line_text().len();
                    self.cursor_col = self.cursor_col.min(line_len);

                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                    EventReply::Handled.then(EventReply::SetFocus)
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::Focus => {
                self.is_focused = true;
                self.cursor_blink_timer = 0.0;
                self.cursor_visible = true;
                EventReply::Handled
            }
            UIEvent::Blur => {
                self.is_focused = false;
                EventReply::Handled
            }
            UIEvent::TextInput { character } => {
                if self.is_focused && !self.read_only {
                    self.insert_char(*character);
                    self.ensure_cursor_visible(rect);
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::KeyInput {
                key,
                pressed: true,
                modifiers,
            } => {
                if !self.is_focused {
                    return EventReply::Unhandled;
                }

                match key {
                    KeyCode::Enter => {
                        self.insert_char('\n');
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::Tab => {
                        self.insert_char('\t');
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::Backspace => {
                        self.delete_char_before();
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::Delete => {
                        self.delete_char_after();
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::ArrowLeft => {
                        if self.cursor_col > 0 {
                            self.cursor_col -= 1;
                        } else if self.cursor_line > 0 {
                            self.cursor_line -= 1;
                            self.cursor_col = self.current_line_text().len();
                        }
                        self.cursor_blink_timer = 0.0;
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::ArrowRight => {
                        let line_len = self.current_line_text().len();
                        if self.cursor_col < line_len {
                            self.cursor_col += 1;
                        } else if self.cursor_line + 1 < self.line_count() {
                            self.cursor_line += 1;
                            self.cursor_col = 0;
                        }
                        self.cursor_blink_timer = 0.0;
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::ArrowUp => {
                        if self.cursor_line > 0 {
                            self.cursor_line -= 1;
                            let line_len = self.current_line_text().len();
                            self.cursor_col = self.cursor_col.min(line_len);
                        }
                        self.cursor_blink_timer = 0.0;
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::ArrowDown => {
                        if self.cursor_line + 1 < self.line_count() {
                            self.cursor_line += 1;
                            let line_len = self.current_line_text().len();
                            self.cursor_col = self.cursor_col.min(line_len);
                        }
                        self.cursor_blink_timer = 0.0;
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::Home => {
                        self.cursor_col = 0;
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::End => {
                        self.cursor_col = self.current_line_text().len();
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::PageUp => {
                        let page = (rect.height() / self.line_height()) as usize;
                        self.cursor_line = self.cursor_line.saturating_sub(page);
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    KeyCode::PageDown => {
                        let page = (rect.height() / self.line_height()) as usize;
                        self.cursor_line =
                            (self.cursor_line + page).min(self.line_count().saturating_sub(1));
                        self.ensure_cursor_visible(rect);
                        EventReply::Handled
                    }
                    _ => EventReply::Unhandled,
                }
            }
            UIEvent::Scroll { delta, .. } => {
                self.scroll_offset_y =
                    (self.scroll_offset_y - delta.y * self.line_height() * 3.0).max(0.0);
                EventReply::Handled
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 12. SearchBox
// =========================================================================

/// Text input with a search icon and clear button.
#[derive(Debug, Clone)]
pub struct SearchBox {
    pub id: UIId,
    pub inner: EditableTextBox,
    pub search_icon_color: Color,
    pub clear_button_color: Color,
    pub clear_button_hovered: bool,
    pub on_search_text_changed: bool,
    pub icon_padding: f32,
}

impl SearchBox {
    pub fn new() -> Self {
        let mut inner = EditableTextBox::new();
        inner.hint_text = "Search...".to_string();
        inner.padding.left = 28.0;
        inner.padding.right = 28.0;

        Self {
            id: UIId::INVALID,
            inner,
            search_icon_color: Color::from_rgba8(160, 160, 160, 200),
            clear_button_color: Color::from_rgba8(160, 160, 160, 200),
            clear_button_hovered: false,
            on_search_text_changed: false,
            icon_padding: 8.0,
        }
    }

    pub fn with_hint(mut self, hint: &str) -> Self {
        self.inner.hint_text = hint.to_string();
        self
    }

    pub fn text(&self) -> &str {
        &self.inner.text
    }

    pub fn set_text(&mut self, text: &str) {
        self.inner.text = text.to_string();
        self.inner.cursor_pos = text.len();
    }

    pub fn clear(&mut self) {
        self.inner.text.clear();
        self.inner.cursor_pos = 0;
        self.on_search_text_changed = true;
    }

    fn clear_button_rect(&self, rect: Rect) -> Rect {
        let size = 16.0;
        let cx = rect.max.x - self.icon_padding - size * 0.5;
        let cy = (rect.min.y + rect.max.y) * 0.5;
        Rect::new(
            Vec2::new(cx - size * 0.5, cy - size * 0.5),
            Vec2::new(cx + size * 0.5, cy + size * 0.5),
        )
    }

    fn search_icon_rect(&self, rect: Rect) -> Rect {
        let size = 14.0;
        let cx = rect.min.x + self.icon_padding + size * 0.5;
        let cy = (rect.min.y + rect.max.y) * 0.5;
        Rect::new(
            Vec2::new(cx - size * 0.5, cy - size * 0.5),
            Vec2::new(cx + size * 0.5, cy + size * 0.5),
        )
    }
}

impl SlateWidget for SearchBox {
    fn compute_desired_size(&self, max_width: Option<f32>) -> Vec2 {
        self.inner.compute_desired_size(max_width)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        self.inner.paint(draw_list, rect);

        // Search icon (magnifying glass - circle + line).
        let icon_rect = self.search_icon_rect(rect);
        let icx = (icon_rect.min.x + icon_rect.max.x) * 0.5;
        let icy = (icon_rect.min.y + icon_rect.max.y) * 0.5;
        let r = icon_rect.width() * 0.35;
        draw_list.push(DrawCommand::Circle {
            center: Vec2::new(icx - 1.0, icy - 1.0),
            radius: r,
            color: Color::TRANSPARENT,
            border: BorderSpec::new(self.search_icon_color, 1.5),
        });
        draw_list.draw_line(
            Vec2::new(icx + r * 0.5, icy + r * 0.5),
            Vec2::new(icx + r * 1.2, icy + r * 1.2),
            self.search_icon_color,
            1.5,
        );

        // Clear button (X) -- only shown when there is text.
        if !self.inner.text.is_empty() {
            let cr = self.clear_button_rect(rect);
            let ccx = (cr.min.x + cr.max.x) * 0.5;
            let ccy = (cr.min.y + cr.max.y) * 0.5;
            let s = 4.0;
            let color = if self.clear_button_hovered {
                Color::WHITE
            } else {
                self.clear_button_color
            };
            draw_list.draw_line(
                Vec2::new(ccx - s, ccy - s),
                Vec2::new(ccx + s, ccy + s),
                color,
                1.5,
            );
            draw_list.draw_line(
                Vec2::new(ccx + s, ccy - s),
                Vec2::new(ccx - s, ccy + s),
                color,
                1.5,
            );
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        // Check clear button click.
        if let UIEvent::Click {
            position,
            button: MouseButton::Left,
            ..
        } = event
        {
            if !self.inner.text.is_empty() {
                let cr = self.clear_button_rect(rect);
                if cr.contains(*position) {
                    self.clear();
                    return EventReply::Handled;
                }
            }
        }

        // Check hover on clear button.
        if let UIEvent::Hover { position } = event {
            let cr = self.clear_button_rect(rect);
            self.clear_button_hovered = cr.contains(*position);
        }

        let reply = self.inner.handle_event(event, rect);
        if self.inner.text_changed {
            self.on_search_text_changed = true;
        }
        reply
    }
}

// =========================================================================
// 13. ComboBox
// =========================================================================

/// Dropdown selector with a list of items.
#[derive(Debug, Clone)]
pub struct ComboBox {
    pub id: UIId,
    pub items: Vec<String>,
    pub selected_index: Option<usize>,
    pub is_open: bool,
    pub hovered_index: Option<usize>,
    pub font_size: f32,
    pub font_id: u32,
    pub text_color: Color,
    pub background_color: Color,
    pub hovered_bg: Color,
    pub selected_bg: Color,
    pub dropdown_bg: Color,
    pub border_color: Color,
    pub corner_radii: CornerRadii,
    pub item_height: f32,
    pub desired_width: f32,
    pub enabled: bool,
    pub visible: bool,
    pub hovered: bool,
    pub selection_changed: bool,
    pub max_dropdown_items: usize,
    pub dropdown_scroll_offset: f32,
}

impl ComboBox {
    pub fn new(items: Vec<String>) -> Self {
        Self {
            id: UIId::INVALID,
            items,
            selected_index: None,
            is_open: false,
            hovered_index: None,
            font_size: 14.0,
            font_id: 0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#3A3A3A"),
            hovered_bg: Color::from_hex("#4A4A4A"),
            selected_bg: Color::from_hex("#2266CC"),
            dropdown_bg: Color::from_hex("#2A2A2A"),
            border_color: Color::from_hex("#555555"),
            corner_radii: CornerRadii::all(3.0),
            item_height: 24.0,
            desired_width: 150.0,
            enabled: true,
            visible: true,
            hovered: false,
            selection_changed: false,
            max_dropdown_items: 8,
            dropdown_scroll_offset: 0.0,
        }
    }

    pub fn selected_text(&self) -> Option<&str> {
        self.selected_index
            .and_then(|i| self.items.get(i))
            .map(|s| s.as_str())
    }

    pub fn take_selection_changed(&mut self) -> bool {
        let c = self.selection_changed;
        self.selection_changed = false;
        c
    }

    fn dropdown_rect(&self, rect: Rect) -> Rect {
        let visible_items = self.items.len().min(self.max_dropdown_items);
        let height = visible_items as f32 * self.item_height;
        Rect::new(
            Vec2::new(rect.min.x, rect.max.y),
            Vec2::new(rect.max.x, rect.max.y + height),
        )
    }

    fn arrow_rect(&self, rect: Rect) -> Rect {
        let size = 16.0;
        Rect::new(
            Vec2::new(rect.max.x - size - 4.0, rect.min.y),
            Vec2::new(rect.max.x, rect.max.y),
        )
    }
}

impl SlateWidget for ComboBox {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        Vec2::new(self.desired_width, self.item_height + 4.0)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        // Main button area.
        let bg = if self.hovered || self.is_open {
            self.hovered_bg
        } else {
            self.background_color
        };
        draw_list.draw_rounded_rect(
            rect,
            bg,
            self.corner_radii,
            BorderSpec::new(self.border_color, 1.0),
        );

        // Selected text.
        let display = self.selected_text().unwrap_or("Select...");
        let ty = rect.min.y + (rect.height() - self.font_size) * 0.5;
        draw_list.push(DrawCommand::Text {
            text: display.to_string(),
            position: Vec2::new(rect.min.x + 8.0, ty),
            font_size: self.font_size,
            color: if self.selected_index.is_some() {
                self.text_color
            } else {
                Color::from_rgba8(128, 128, 128, 180)
            },
            font_id: self.font_id,
            max_width: Some(rect.width() - 24.0),
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });

        // Dropdown arrow.
        let arrow = self.arrow_rect(rect);
        let acx = (arrow.min.x + arrow.max.x) * 0.5;
        let acy = (arrow.min.y + arrow.max.y) * 0.5;
        if self.is_open {
            draw_list.push(DrawCommand::Triangle {
                p0: Vec2::new(acx, acy - 3.0),
                p1: Vec2::new(acx - 5.0, acy + 3.0),
                p2: Vec2::new(acx + 5.0, acy + 3.0),
                color: self.text_color,
            });
        } else {
            draw_list.push(DrawCommand::Triangle {
                p0: Vec2::new(acx, acy + 3.0),
                p1: Vec2::new(acx - 5.0, acy - 3.0),
                p2: Vec2::new(acx + 5.0, acy - 3.0),
                color: self.text_color,
            });
        }

        // Dropdown list.
        if self.is_open {
            let dd_rect = self.dropdown_rect(rect);
            draw_list.draw_rounded_rect(
                dd_rect,
                self.dropdown_bg,
                CornerRadii::new(0.0, 0.0, 4.0, 4.0),
                BorderSpec::new(self.border_color, 1.0),
            );

            let visible_start =
                (self.dropdown_scroll_offset / self.item_height).floor() as usize;
            let visible_count = self.max_dropdown_items.min(self.items.len());

            for i in 0..visible_count {
                let item_idx = visible_start + i;
                if item_idx >= self.items.len() {
                    break;
                }
                let iy = dd_rect.min.y + i as f32 * self.item_height;
                let item_rect = Rect::new(
                    Vec2::new(dd_rect.min.x, iy),
                    Vec2::new(dd_rect.max.x, iy + self.item_height),
                );

                // Highlight.
                if self.hovered_index == Some(item_idx) {
                    draw_list.draw_rect(item_rect, self.selected_bg);
                } else if self.selected_index == Some(item_idx) {
                    draw_list.draw_rect(item_rect, self.selected_bg.with_alpha(0.3));
                }

                let tiy = iy + (self.item_height - self.font_size) * 0.5;
                draw_list.push(DrawCommand::Text {
                    text: self.items[item_idx].clone(),
                    position: Vec2::new(dd_rect.min.x + 8.0, tiy),
                    font_size: self.font_size,
                    color: self.text_color,
                    font_id: self.font_id,
                    max_width: Some(dd_rect.width() - 16.0),
                    align: TextAlign::Left,
                    vertical_align: TextVerticalAlign::Top,
                });
            }
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                self.hovered = rect.contains(*position);
                if self.is_open {
                    let dd_rect = self.dropdown_rect(rect);
                    if dd_rect.contains(*position) {
                        let local_y = position.y - dd_rect.min.y + self.dropdown_scroll_offset;
                        let idx = (local_y / self.item_height).floor() as usize;
                        self.hovered_index = if idx < self.items.len() {
                            Some(idx)
                        } else {
                            None
                        };
                    } else {
                        self.hovered_index = None;
                    }
                }
                EventReply::Handled
            }
            UIEvent::HoverEnd => {
                self.hovered = false;
                self.hovered_index = None;
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if self.is_open {
                    let dd_rect = self.dropdown_rect(rect);
                    if dd_rect.contains(*position) {
                        let local_y = position.y - dd_rect.min.y + self.dropdown_scroll_offset;
                        let idx = (local_y / self.item_height).floor() as usize;
                        if idx < self.items.len() {
                            self.selected_index = Some(idx);
                            self.selection_changed = true;
                        }
                    }
                    self.is_open = false;
                    EventReply::Handled
                } else if rect.contains(*position) {
                    self.is_open = true;
                    EventReply::Handled
                } else {
                    self.is_open = false;
                    EventReply::Unhandled
                }
            }
            UIEvent::Scroll { delta, .. } => {
                if self.is_open {
                    let max_scroll =
                        (self.items.len() as f32 - self.max_dropdown_items as f32)
                            * self.item_height;
                    self.dropdown_scroll_offset =
                        (self.dropdown_scroll_offset - delta.y * self.item_height)
                            .clamp(0.0, max_scroll.max(0.0));
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::KeyInput {
                key: KeyCode::Escape,
                pressed: true,
                ..
            } => {
                if self.is_open {
                    self.is_open = false;
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 14. Hyperlink
// =========================================================================

/// Clickable text link.
#[derive(Debug, Clone)]
pub struct Hyperlink {
    pub id: UIId,
    pub text: String,
    pub url: String,
    pub font_size: f32,
    pub font_id: u32,
    pub color: Color,
    pub hovered_color: Color,
    pub visited_color: Color,
    pub hovered: bool,
    pub visited: bool,
    pub clicked: bool,
    pub underline: bool,
    pub visible: bool,
}

impl Hyperlink {
    pub fn new(text: &str, url: &str) -> Self {
        Self {
            id: UIId::INVALID,
            text: text.to_string(),
            url: url.to_string(),
            font_size: 14.0,
            font_id: 0,
            color: Color::from_hex("#4488FF"),
            hovered_color: Color::from_hex("#66AAFF"),
            visited_color: Color::from_hex("#9966CC"),
            hovered: false,
            visited: false,
            clicked: false,
            underline: true,
            visible: true,
        }
    }

    pub fn take_click(&mut self) -> bool {
        let c = self.clicked;
        self.clicked = false;
        c
    }
}

impl SlateWidget for Hyperlink {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        let w = self.text.len() as f32 * self.font_size * 0.6;
        Vec2::new(w, self.font_size * 1.2)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }
        let color = if self.hovered {
            self.hovered_color
        } else if self.visited {
            self.visited_color
        } else {
            self.color
        };

        let ty = rect.min.y + (rect.height() - self.font_size) * 0.5;
        draw_list.push(DrawCommand::Text {
            text: self.text.clone(),
            position: Vec2::new(rect.min.x, ty),
            font_size: self.font_size,
            color,
            font_id: self.font_id,
            max_width: Some(rect.width()),
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });

        if self.underline {
            let uy = ty + self.font_size + 1.0;
            let uw = self.text.len() as f32 * self.font_size * 0.6;
            draw_list.draw_line(
                Vec2::new(rect.min.x, uy),
                Vec2::new(rect.min.x + uw, uy),
                color,
                1.0,
            );
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        match event {
            UIEvent::Hover { position } => {
                self.hovered = rect.contains(*position);
                EventReply::Handled
            }
            UIEvent::HoverEnd => {
                self.hovered = false;
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if rect.contains(*position) {
                    self.clicked = true;
                    self.visited = true;
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 15. ProgressBar
// =========================================================================

/// Progress bar mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressMode {
    Determinate,
    Indeterminate,
}

/// A progress bar with determinate and indeterminate modes.
#[derive(Debug, Clone)]
pub struct ProgressBar {
    pub id: UIId,
    pub progress: f32,
    pub mode: ProgressMode,
    pub background_brush: Brush,
    pub fill_brush: Brush,
    pub border: BorderSpec,
    pub corner_radii: CornerRadii,
    pub height: f32,
    pub desired_width: f32,
    pub show_percentage: bool,
    pub text_color: Color,
    pub font_size: f32,
    pub font_id: u32,
    pub visible: bool,
    pub indeterminate_offset: f32,
    pub indeterminate_width: f32,
    pub indeterminate_speed: f32,
}

impl ProgressBar {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            progress: 0.0,
            mode: ProgressMode::Determinate,
            background_brush: Brush::SolidColor(Color::from_hex("#333333")),
            fill_brush: Brush::SolidColor(Color::from_hex("#2266CC")),
            border: BorderSpec::default(),
            corner_radii: CornerRadii::all(4.0),
            height: 20.0,
            desired_width: 200.0,
            show_percentage: true,
            text_color: Color::WHITE,
            font_size: 12.0,
            font_id: 0,
            visible: true,
            indeterminate_offset: 0.0,
            indeterminate_width: 0.3,
            indeterminate_speed: 1.5,
        }
    }

    pub fn with_progress(mut self, p: f32) -> Self {
        self.progress = p.clamp(0.0, 1.0);
        self
    }

    pub fn with_indeterminate(mut self) -> Self {
        self.mode = ProgressMode::Indeterminate;
        self
    }

    pub fn set_progress(&mut self, p: f32) {
        self.progress = p.clamp(0.0, 1.0);
    }

    /// Call each frame with delta time to animate indeterminate mode.
    pub fn tick(&mut self, dt: f32) {
        if self.mode == ProgressMode::Indeterminate {
            self.indeterminate_offset += dt * self.indeterminate_speed;
            if self.indeterminate_offset > 1.0 + self.indeterminate_width {
                self.indeterminate_offset = -self.indeterminate_width;
            }
        }
    }
}

impl SlateWidget for ProgressBar {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        Vec2::new(self.desired_width, self.height)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        // Background.
        self.background_brush
            .paint_rect(draw_list, rect, self.corner_radii);

        match self.mode {
            ProgressMode::Determinate => {
                let fill_w = rect.width() * self.progress;
                if fill_w > 0.5 {
                    let fill_rect = Rect::new(
                        rect.min,
                        Vec2::new(rect.min.x + fill_w, rect.max.y),
                    );
                    self.fill_brush
                        .paint_rect(draw_list, fill_rect, self.corner_radii);
                }

                if self.show_percentage {
                    let text = format!("{}%", (self.progress * 100.0) as u32);
                    let tw = text.len() as f32 * self.font_size * 0.6;
                    let tx = rect.min.x + (rect.width() - tw) * 0.5;
                    let ty = rect.min.y + (rect.height() - self.font_size) * 0.5;
                    draw_list.push(DrawCommand::Text {
                        text,
                        position: Vec2::new(tx, ty),
                        font_size: self.font_size,
                        color: self.text_color,
                        font_id: self.font_id,
                        max_width: None,
                        align: TextAlign::Left,
                        vertical_align: TextVerticalAlign::Top,
                    });
                }
            }
            ProgressMode::Indeterminate => {
                let bar_w = rect.width() * self.indeterminate_width;
                let bar_x = rect.min.x + self.indeterminate_offset * rect.width();
                let clamped_left = bar_x.max(rect.min.x);
                let clamped_right = (bar_x + bar_w).min(rect.max.x);
                if clamped_right > clamped_left {
                    let fill_rect = Rect::new(
                        Vec2::new(clamped_left, rect.min.y),
                        Vec2::new(clamped_right, rect.max.y),
                    );
                    self.fill_brush
                        .paint_rect(draw_list, fill_rect, self.corner_radii);
                }
            }
        }

        // Border.
        if self.border.width > 0.0 {
            draw_list.draw_rounded_rect(
                rect,
                Color::TRANSPARENT,
                self.corner_radii,
                self.border,
            );
        }
    }

    fn handle_event(&mut self, _event: &UIEvent, _rect: Rect) -> EventReply {
        EventReply::Unhandled
    }
}

// =========================================================================
// 16. ExpandableArea
// =========================================================================

/// Collapsible section with animated expand/collapse.
#[derive(Debug, Clone)]
pub struct ExpandableArea {
    pub id: UIId,
    pub header_text: String,
    pub is_expanded: bool,
    pub animation_progress: f32,
    pub animation_speed: f32,
    pub header_height: f32,
    pub content_height: f32,
    pub header_bg: Color,
    pub header_hovered_bg: Color,
    pub header_text_color: Color,
    pub border_color: Color,
    pub font_size: f32,
    pub font_id: u32,
    pub corner_radii: CornerRadii,
    pub hovered: bool,
    pub visible: bool,
    pub toggle_changed: bool,
}

impl ExpandableArea {
    pub fn new(header: &str) -> Self {
        Self {
            id: UIId::INVALID,
            header_text: header.to_string(),
            is_expanded: false,
            animation_progress: 0.0,
            animation_speed: 6.0,
            header_height: 28.0,
            content_height: 100.0,
            header_bg: Color::from_hex("#333333"),
            header_hovered_bg: Color::from_hex("#3E3E3E"),
            header_text_color: Color::WHITE,
            border_color: Color::from_hex("#555555"),
            font_size: 14.0,
            font_id: 0,
            corner_radii: CornerRadii::all(3.0),
            hovered: false,
            visible: true,
            toggle_changed: false,
        }
    }

    pub fn with_expanded(mut self, expanded: bool) -> Self {
        self.is_expanded = expanded;
        self.animation_progress = if expanded { 1.0 } else { 0.0 };
        self
    }

    pub fn with_content_height(mut self, h: f32) -> Self {
        self.content_height = h;
        self
    }

    /// Animate the expand/collapse. Call each frame with dt.
    pub fn tick(&mut self, dt: f32) {
        let target = if self.is_expanded { 1.0 } else { 0.0 };
        let diff = target - self.animation_progress;
        if diff.abs() > 0.001 {
            self.animation_progress += diff.signum() * self.animation_speed * dt;
            self.animation_progress = self.animation_progress.clamp(0.0, 1.0);
        } else {
            self.animation_progress = target;
        }
    }

    /// Current visible height of the content area (animated).
    pub fn current_content_height(&self) -> f32 {
        self.content_height * self.animation_progress
    }

    /// Total height including header.
    pub fn total_height(&self) -> f32 {
        self.header_height + self.current_content_height()
    }

    /// Rect for the content area.
    pub fn content_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            Vec2::new(rect.min.x, rect.min.y + self.header_height),
            Vec2::new(
                rect.max.x,
                rect.min.y + self.header_height + self.current_content_height(),
            ),
        )
    }

    pub fn take_toggle_changed(&mut self) -> bool {
        let c = self.toggle_changed;
        self.toggle_changed = false;
        c
    }
}

impl SlateWidget for ExpandableArea {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        Vec2::new(200.0, self.total_height())
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        let header_rect = Rect::new(
            rect.min,
            Vec2::new(rect.max.x, rect.min.y + self.header_height),
        );

        // Header background.
        let bg = if self.hovered {
            self.header_hovered_bg
        } else {
            self.header_bg
        };
        draw_list.draw_rounded_rect(
            header_rect,
            bg,
            if self.animation_progress > 0.0 {
                CornerRadii::new(
                    self.corner_radii.top_left,
                    self.corner_radii.top_right,
                    0.0,
                    0.0,
                )
            } else {
                self.corner_radii
            },
            BorderSpec::new(self.border_color, 1.0),
        );

        // Arrow icon.
        let arrow_x = header_rect.min.x + 12.0;
        let arrow_y = (header_rect.min.y + header_rect.max.y) * 0.5;
        let angle = self.animation_progress * std::f32::consts::FRAC_PI_2;
        let s = 4.0;
        // Rotate triangle based on expand state.
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let p0 = Vec2::new(
            arrow_x + s * cos_a,
            arrow_y + s * sin_a,
        );
        let p1 = Vec2::new(
            arrow_x - s * sin_a - s * 0.5 * cos_a,
            arrow_y + s * cos_a - s * 0.5 * sin_a,
        );
        let p2 = Vec2::new(
            arrow_x + s * sin_a - s * 0.5 * cos_a,
            arrow_y - s * cos_a - s * 0.5 * sin_a,
        );
        draw_list.push(DrawCommand::Triangle {
            p0,
            p1,
            p2,
            color: self.header_text_color,
        });

        // Header text.
        let tx = arrow_x + 16.0;
        let ty = header_rect.min.y + (self.header_height - self.font_size) * 0.5;
        draw_list.push(DrawCommand::Text {
            text: self.header_text.clone(),
            position: Vec2::new(tx, ty),
            font_size: self.font_size,
            color: self.header_text_color,
            font_id: self.font_id,
            max_width: Some(header_rect.width() - 32.0),
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });

        // Content area (clip to animated height).
        if self.animation_progress > 0.001 {
            let content_rect = self.content_rect(rect);
            draw_list.push_clip(content_rect);
            draw_list.draw_rounded_rect(
                Rect::new(
                    Vec2::new(rect.min.x, rect.min.y + self.header_height),
                    Vec2::new(rect.max.x, rect.min.y + self.header_height + self.content_height),
                ),
                Color::TRANSPARENT,
                CornerRadii::new(0.0, 0.0, self.corner_radii.bottom_right, self.corner_radii.bottom_left),
                BorderSpec::new(self.border_color, 1.0),
            );
            draw_list.pop_clip();
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        let header_rect = Rect::new(
            rect.min,
            Vec2::new(rect.max.x, rect.min.y + self.header_height),
        );

        match event {
            UIEvent::Hover { position } => {
                self.hovered = header_rect.contains(*position);
                EventReply::Handled
            }
            UIEvent::HoverEnd => {
                self.hovered = false;
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if header_rect.contains(*position) {
                    self.is_expanded = !self.is_expanded;
                    self.toggle_changed = true;
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 17. Throbber
// =========================================================================

/// Animated loading indicator (spinning dots / arc).
#[derive(Debug, Clone)]
pub struct Throbber {
    pub id: UIId,
    pub radius: f32,
    pub dot_count: usize,
    pub dot_radius: f32,
    pub color: Color,
    pub rotation: f32,
    pub speed: f32,
    pub visible: bool,
    pub style: ThrobberStyle,
}

/// Visual style for the throbber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrobberStyle {
    SpinningDots,
    SpinningArc,
    PulsingDots,
}

impl Throbber {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            radius: 16.0,
            dot_count: 8,
            dot_radius: 3.0,
            color: Color::WHITE,
            rotation: 0.0,
            speed: 4.0,
            visible: true,
            style: ThrobberStyle::SpinningDots,
        }
    }

    pub fn with_radius(mut self, r: f32) -> Self {
        self.radius = r;
        self
    }

    pub fn with_color(mut self, c: Color) -> Self {
        self.color = c;
        self
    }

    pub fn with_style(mut self, s: ThrobberStyle) -> Self {
        self.style = s;
        self
    }

    /// Animate the throbber. Call each frame with dt.
    pub fn tick(&mut self, dt: f32) {
        self.rotation += self.speed * dt;
        if self.rotation > std::f32::consts::TAU {
            self.rotation -= std::f32::consts::TAU;
        }
    }
}

impl SlateWidget for Throbber {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        let d = self.radius * 2.0 + self.dot_radius * 2.0;
        Vec2::new(d, d)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        let cx = (rect.min.x + rect.max.x) * 0.5;
        let cy = (rect.min.y + rect.max.y) * 0.5;

        match self.style {
            ThrobberStyle::SpinningDots => {
                for i in 0..self.dot_count {
                    let angle = self.rotation
                        + (i as f32 / self.dot_count as f32) * std::f32::consts::TAU;
                    let dx = cx + angle.cos() * self.radius;
                    let dy = cy + angle.sin() * self.radius;
                    let alpha = 1.0 - (i as f32 / self.dot_count as f32) * 0.8;
                    draw_list.push(DrawCommand::Circle {
                        center: Vec2::new(dx, dy),
                        radius: self.dot_radius,
                        color: self.color.with_alpha(alpha),
                        border: BorderSpec::default(),
                    });
                }
            }
            ThrobberStyle::SpinningArc => {
                let segments = 32;
                let arc_length = std::f32::consts::PI * 1.5;
                let mut points = Vec::with_capacity(segments + 1);
                for j in 0..=segments {
                    let t = j as f32 / segments as f32;
                    let angle = self.rotation + t * arc_length;
                    points.push(Vec2::new(
                        cx + angle.cos() * self.radius,
                        cy + angle.sin() * self.radius,
                    ));
                }
                draw_list.push(DrawCommand::Polyline {
                    points,
                    color: self.color,
                    thickness: self.dot_radius,
                    closed: false,
                });
            }
            ThrobberStyle::PulsingDots => {
                for i in 0..self.dot_count {
                    let angle = (i as f32 / self.dot_count as f32) * std::f32::consts::TAU;
                    let dx = cx + angle.cos() * self.radius;
                    let dy = cy + angle.sin() * self.radius;
                    let phase = self.rotation + i as f32 * 0.5;
                    let scale = 0.5 + 0.5 * phase.sin().abs();
                    draw_list.push(DrawCommand::Circle {
                        center: Vec2::new(dx, dy),
                        radius: self.dot_radius * scale,
                        color: self.color.with_alpha(0.4 + 0.6 * scale),
                        border: BorderSpec::default(),
                    });
                }
            }
        }
    }

    fn handle_event(&mut self, _event: &UIEvent, _rect: Rect) -> EventReply {
        EventReply::Unhandled
    }
}

// =========================================================================
// 18. InlineEditableTextBlock
// =========================================================================

/// Double-click-to-edit text block (rename pattern).
#[derive(Debug, Clone)]
pub struct InlineEditableTextBlock {
    pub id: UIId,
    pub text: String,
    pub editing: bool,
    pub edit_text: String,
    pub cursor_pos: usize,
    pub selection_start: Option<usize>,
    pub font_size: f32,
    pub font_id: u32,
    pub text_color: Color,
    pub editing_bg: Color,
    pub editing_border: Color,
    pub selection_color: Color,
    pub cursor_color: Color,
    pub hovered: bool,
    pub enabled: bool,
    pub visible: bool,
    pub text_changed: bool,
    pub cursor_blink_timer: f32,
    pub cursor_visible: bool,
    pub max_length: Option<usize>,
}

impl InlineEditableTextBlock {
    pub fn new(text: &str) -> Self {
        Self {
            id: UIId::INVALID,
            text: text.to_string(),
            editing: false,
            edit_text: String::new(),
            cursor_pos: 0,
            selection_start: None,
            font_size: 14.0,
            font_id: 0,
            text_color: Color::WHITE,
            editing_bg: Color::from_hex("#2A2A2A"),
            editing_border: Color::from_hex("#4488FF"),
            selection_color: Color::from_hex("#2266CC80"),
            cursor_color: Color::WHITE,
            hovered: false,
            enabled: true,
            visible: true,
            text_changed: false,
            cursor_blink_timer: 0.0,
            cursor_visible: true,
            max_length: None,
        }
    }

    pub fn begin_editing(&mut self) {
        self.editing = true;
        self.edit_text = self.text.clone();
        self.cursor_pos = self.edit_text.len();
        self.selection_start = Some(0);
        self.cursor_blink_timer = 0.0;
        self.cursor_visible = true;
    }

    pub fn commit(&mut self) {
        if self.editing {
            if !self.edit_text.is_empty() {
                self.text = self.edit_text.clone();
                self.text_changed = true;
            }
            self.editing = false;
            self.edit_text.clear();
            self.selection_start = None;
        }
    }

    pub fn cancel(&mut self) {
        self.editing = false;
        self.edit_text.clear();
        self.selection_start = None;
    }

    pub fn take_text_changed(&mut self) -> bool {
        let c = self.text_changed;
        self.text_changed = false;
        c
    }

    fn char_width(&self) -> f32 {
        self.font_size * 0.6
    }

    pub fn tick(&mut self, dt: f32) {
        if self.editing {
            self.cursor_blink_timer += dt;
            let cycle = 1.06;
            let phase = self.cursor_blink_timer % cycle;
            self.cursor_visible = phase < 0.53;
        }
    }
}

impl SlateWidget for InlineEditableTextBlock {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        let display = if self.editing {
            &self.edit_text
        } else {
            &self.text
        };
        let w = display.len() as f32 * self.char_width() + 8.0;
        Vec2::new(w.max(40.0), self.font_size * 1.4)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        let cw = self.char_width();

        if self.editing {
            // Editing mode: show text field.
            draw_list.draw_rounded_rect(
                rect,
                self.editing_bg,
                CornerRadii::all(2.0),
                BorderSpec::new(self.editing_border, 1.0),
            );

            let ty = rect.min.y + (rect.height() - self.font_size) * 0.5;

            // Selection.
            if let Some(sel_start) = self.selection_start {
                let s = sel_start.min(self.cursor_pos);
                let e = sel_start.max(self.cursor_pos);
                if s != e {
                    let sx = rect.min.x + 4.0 + s as f32 * cw;
                    let ex = rect.min.x + 4.0 + e as f32 * cw;
                    draw_list.draw_rect(
                        Rect::new(Vec2::new(sx, ty), Vec2::new(ex, ty + self.font_size)),
                        self.selection_color,
                    );
                }
            }

            draw_list.push(DrawCommand::Text {
                text: self.edit_text.clone(),
                position: Vec2::new(rect.min.x + 4.0, ty),
                font_size: self.font_size,
                color: self.text_color,
                font_id: self.font_id,
                max_width: Some(rect.width() - 8.0),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            if self.cursor_visible {
                let cx = rect.min.x + 4.0 + self.cursor_pos as f32 * cw;
                draw_list.draw_line(
                    Vec2::new(cx, ty),
                    Vec2::new(cx, ty + self.font_size),
                    self.cursor_color,
                    1.0,
                );
            }
        } else {
            // Display mode.
            let ty = rect.min.y + (rect.height() - self.font_size) * 0.5;
            draw_list.push(DrawCommand::Text {
                text: self.text.clone(),
                position: Vec2::new(rect.min.x, ty),
                font_size: self.font_size,
                color: self.text_color,
                font_id: self.font_id,
                max_width: Some(rect.width()),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                self.hovered = rect.contains(*position);
                EventReply::Handled
            }
            UIEvent::HoverEnd => {
                self.hovered = false;
                EventReply::Handled
            }
            UIEvent::DoubleClick {
                position,
                button: MouseButton::Left,
            } => {
                if rect.contains(*position) && !self.editing {
                    self.begin_editing();
                    return EventReply::Handled.then(EventReply::SetFocus);
                }
                EventReply::Unhandled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if self.editing && !rect.contains(*position) {
                    self.commit();
                    return EventReply::Handled.then(EventReply::ClearFocus);
                }
                if self.editing && rect.contains(*position) {
                    let cw = self.char_width();
                    let local_x = position.x - rect.min.x - 4.0;
                    self.cursor_pos =
                        (local_x / cw).round().max(0.0) as usize;
                    self.cursor_pos = self.cursor_pos.min(self.edit_text.len());
                    self.selection_start = None;
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::TextInput { character } => {
                if self.editing && !character.is_control() {
                    if let Some(max) = self.max_length {
                        if self.edit_text.len() >= max {
                            return EventReply::Handled;
                        }
                    }
                    // Delete selection first.
                    if let Some(sel_start) = self.selection_start {
                        let s = sel_start.min(self.cursor_pos);
                        let e = sel_start.max(self.cursor_pos);
                        if s != e {
                            let before: String = self.edit_text.chars().take(s).collect();
                            let after: String = self.edit_text.chars().skip(e).collect();
                            self.edit_text = format!("{}{}", before, after);
                            self.cursor_pos = s;
                        }
                        self.selection_start = None;
                    }
                    let before: String = self.edit_text.chars().take(self.cursor_pos).collect();
                    let after: String = self.edit_text.chars().skip(self.cursor_pos).collect();
                    self.edit_text = format!("{}{}{}", before, character, after);
                    self.cursor_pos += 1;
                    self.cursor_blink_timer = 0.0;
                    self.cursor_visible = true;
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::KeyInput {
                key, pressed: true, ..
            } => {
                if !self.editing {
                    return EventReply::Unhandled;
                }
                match key {
                    KeyCode::Enter => {
                        self.commit();
                        EventReply::Handled.then(EventReply::ClearFocus)
                    }
                    KeyCode::Escape => {
                        self.cancel();
                        EventReply::Handled.then(EventReply::ClearFocus)
                    }
                    KeyCode::Backspace => {
                        if self.cursor_pos > 0 {
                            let before: String =
                                self.edit_text.chars().take(self.cursor_pos - 1).collect();
                            let after: String =
                                self.edit_text.chars().skip(self.cursor_pos).collect();
                            self.edit_text = format!("{}{}", before, after);
                            self.cursor_pos -= 1;
                        }
                        EventReply::Handled
                    }
                    KeyCode::Delete => {
                        if self.cursor_pos < self.edit_text.len() {
                            let before: String =
                                self.edit_text.chars().take(self.cursor_pos).collect();
                            let after: String =
                                self.edit_text.chars().skip(self.cursor_pos + 1).collect();
                            self.edit_text = format!("{}{}", before, after);
                        }
                        EventReply::Handled
                    }
                    KeyCode::ArrowLeft => {
                        if self.cursor_pos > 0 {
                            self.cursor_pos -= 1;
                        }
                        self.cursor_blink_timer = 0.0;
                        EventReply::Handled
                    }
                    KeyCode::ArrowRight => {
                        if self.cursor_pos < self.edit_text.len() {
                            self.cursor_pos += 1;
                        }
                        self.cursor_blink_timer = 0.0;
                        EventReply::Handled
                    }
                    KeyCode::Home => {
                        self.cursor_pos = 0;
                        EventReply::Handled
                    }
                    KeyCode::End => {
                        self.cursor_pos = self.edit_text.len();
                        EventReply::Handled
                    }
                    KeyCode::A => {
                        // Ctrl+A: select all handled elsewhere.
                        EventReply::Unhandled
                    }
                    _ => EventReply::Unhandled,
                }
            }
            UIEvent::Blur => {
                if self.editing {
                    self.commit();
                }
                EventReply::Handled
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 19. Spacer
// =========================================================================

/// Configurable empty space widget.
#[derive(Debug, Clone)]
pub struct Spacer {
    pub id: UIId,
    pub size: Vec2,
    pub fill_remaining: bool,
}

impl Spacer {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            id: UIId::INVALID,
            size: Vec2::new(width, height),
            fill_remaining: false,
        }
    }

    pub fn horizontal(width: f32) -> Self {
        Self::new(width, 0.0)
    }

    pub fn vertical(height: f32) -> Self {
        Self::new(0.0, height)
    }

    /// A spacer that expands to fill remaining space in a flex layout.
    pub fn fill() -> Self {
        Self {
            id: UIId::INVALID,
            size: Vec2::ZERO,
            fill_remaining: true,
        }
    }
}

impl SlateWidget for Spacer {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        self.size
    }

    fn paint(&self, _draw_list: &mut DrawList, _rect: Rect) {
        // Spacers are invisible.
    }

    fn handle_event(&mut self, _event: &UIEvent, _rect: Rect) -> EventReply {
        EventReply::Unhandled
    }
}
