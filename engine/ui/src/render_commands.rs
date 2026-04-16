//! UI draw commands and rendering abstraction.
//!
//! The widget and layout systems emit [`DrawCommand`]s into a [`DrawList`].
//! A backend that implements [`UIRenderer`] consumes the draw list each frame
//! to produce actual GPU work.

use glam::{Mat3, Vec2, Vec4};
use serde::{Deserialize, Serialize};

use genovo_core::Rect;

// ---------------------------------------------------------------------------
// Color helper
// ---------------------------------------------------------------------------

/// RGBA colour in linear float space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const TRANSPARENT: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
    pub const RED: Self = Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Self = Self { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Self = Self { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const YELLOW: Self = Self { r: 1.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const CYAN: Self = Self { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const MAGENTA: Self = Self { r: 1.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const GRAY: Self = Self { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Create a colour from 0-255 byte values.
    pub fn from_rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    /// Create from a hex string like `"#FF8800"` or `"#FF880080"`.
    pub fn from_hex(hex: &str) -> Self {
        let hex = hex.trim_start_matches('#');
        let len = hex.len();
        if len == 6 || len == 8 {
            let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
            let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
            let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
            let a = if len == 8 {
                u8::from_str_radix(&hex[6..8], 16).unwrap_or(255)
            } else {
                255
            };
            Self::from_rgba8(r, g, b, a)
        } else {
            Self::BLACK
        }
    }

    /// Linearly interpolate between two colours.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// Returns the colour with modified alpha.
    pub fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }

    /// Convert to Vec4.
    pub fn to_vec4(self) -> Vec4 {
        Vec4::new(self.r, self.g, self.b, self.a)
    }

    /// Convert from HSV (h in 0..360, s and v in 0..1).
    pub fn from_hsv(h: f32, s: f32, v: f32) -> Self {
        let h = h % 360.0;
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };
        Self::new(r + m, g + m, b + m, 1.0)
    }

    /// Convert to HSV (h in 0..360, s and v in 0..1).
    pub fn to_hsv(&self) -> (f32, f32, f32) {
        let cmax = self.r.max(self.g).max(self.b);
        let cmin = self.r.min(self.g).min(self.b);
        let delta = cmax - cmin;

        let h = if delta < 1e-6 {
            0.0
        } else if (cmax - self.r).abs() < 1e-6 {
            60.0 * (((self.g - self.b) / delta) % 6.0)
        } else if (cmax - self.g).abs() < 1e-6 {
            60.0 * (((self.b - self.r) / delta) + 2.0)
        } else {
            60.0 * (((self.r - self.g) / delta) + 4.0)
        };
        let h = if h < 0.0 { h + 360.0 } else { h };
        let s = if cmax < 1e-6 { 0.0 } else { delta / cmax };
        (h, s, cmax)
    }

    /// Darken the colour by a factor (0 = no change, 1 = black).
    pub fn darken(self, amount: f32) -> Self {
        let factor = 1.0 - amount.clamp(0.0, 1.0);
        Self::new(self.r * factor, self.g * factor, self.b * factor, self.a)
    }

    /// Lighten the colour by a factor (0 = no change, 1 = white).
    pub fn lighten(self, amount: f32) -> Self {
        let t = amount.clamp(0.0, 1.0);
        Self::new(
            self.r + (1.0 - self.r) * t,
            self.g + (1.0 - self.g) * t,
            self.b + (1.0 - self.b) * t,
            self.a,
        )
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}

impl From<Color> for Vec4 {
    fn from(c: Color) -> Self {
        c.to_vec4()
    }
}

// ---------------------------------------------------------------------------
// TextureId
// ---------------------------------------------------------------------------

/// Opaque reference to a texture managed by the renderer backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextureId(pub u64);

impl TextureId {
    pub const INVALID: Self = Self(u64::MAX);
}

impl Default for TextureId {
    fn default() -> Self {
        Self::INVALID
    }
}

// ---------------------------------------------------------------------------
// DrawCommand
// ---------------------------------------------------------------------------

/// A border specification for rectangle drawing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Border {
    pub color: Color,
    pub width: f32,
}

impl Border {
    pub fn new(color: Color, width: f32) -> Self {
        Self { color, width }
    }
}

impl Default for Border {
    fn default() -> Self {
        Self {
            color: Color::TRANSPARENT,
            width: 0.0,
        }
    }
}

/// Shadow specification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Shadow {
    pub color: Color,
    pub offset: Vec2,
    pub blur_radius: f32,
    pub spread: f32,
}

impl Shadow {
    pub fn new(color: Color, offset: Vec2, blur_radius: f32, spread: f32) -> Self {
        Self {
            color,
            offset,
            blur_radius,
            spread,
        }
    }
}

impl Default for Shadow {
    fn default() -> Self {
        Self {
            color: Color::TRANSPARENT,
            offset: Vec2::ZERO,
            blur_radius: 0.0,
            spread: 0.0,
        }
    }
}

/// Corner radii (top-left, top-right, bottom-right, bottom-left).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CornerRadii {
    pub top_left: f32,
    pub top_right: f32,
    pub bottom_right: f32,
    pub bottom_left: f32,
}

impl CornerRadii {
    pub const ZERO: Self = Self {
        top_left: 0.0,
        top_right: 0.0,
        bottom_right: 0.0,
        bottom_left: 0.0,
    };

    pub fn all(radius: f32) -> Self {
        Self {
            top_left: radius,
            top_right: radius,
            bottom_right: radius,
            bottom_left: radius,
        }
    }

    pub fn new(tl: f32, tr: f32, br: f32, bl: f32) -> Self {
        Self {
            top_left: tl,
            top_right: tr,
            bottom_right: br,
            bottom_left: bl,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.top_left == 0.0
            && self.top_right == 0.0
            && self.bottom_right == 0.0
            && self.bottom_left == 0.0
    }
}

impl Default for CornerRadii {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Gradient stop.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GradientStop {
    pub offset: f32,
    pub color: Color,
}

/// Gradient types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Gradient {
    Linear {
        start: Vec2,
        end: Vec2,
        stops: Vec<GradientStop>,
    },
    Radial {
        center: Vec2,
        radius: f32,
        stops: Vec<GradientStop>,
    },
}

/// Image scaling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageScaleMode {
    /// Scale to fill the rect exactly (may distort).
    Stretch,
    /// Scale to fit inside the rect, preserving aspect ratio (may letterbox).
    Fit,
    /// Scale to fill the rect, preserving aspect ratio (may crop).
    Fill,
    /// Render at original size, centered.
    Center,
    /// Tile the image.
    Tile,
}

impl Default for ImageScaleMode {
    fn default() -> Self {
        Self::Stretch
    }
}

/// Text horizontal alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

impl Default for TextAlign {
    fn default() -> Self {
        Self::Left
    }
}

/// Text vertical alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextVerticalAlign {
    Top,
    Middle,
    Bottom,
}

impl Default for TextVerticalAlign {
    fn default() -> Self {
        Self::Top
    }
}

/// A single draw command emitted by the UI system.
#[derive(Debug, Clone)]
pub enum DrawCommand {
    /// Draw a filled rectangle.
    Rect {
        rect: Rect,
        color: Color,
        corner_radii: CornerRadii,
        border: Border,
        shadow: Option<Shadow>,
    },
    /// Draw a filled circle.
    Circle {
        center: Vec2,
        radius: f32,
        color: Color,
        border: Border,
    },
    /// Draw a line segment.
    Line {
        start: Vec2,
        end: Vec2,
        color: Color,
        thickness: f32,
    },
    /// Draw a polyline.
    Polyline {
        points: Vec<Vec2>,
        color: Color,
        thickness: f32,
        closed: bool,
    },
    /// Draw text.
    Text {
        text: String,
        position: Vec2,
        font_size: f32,
        color: Color,
        font_id: u32,
        max_width: Option<f32>,
        align: TextAlign,
        vertical_align: TextVerticalAlign,
    },
    /// Draw a textured image.
    Image {
        rect: Rect,
        texture: TextureId,
        tint: Color,
        corner_radii: CornerRadii,
        scale_mode: ImageScaleMode,
        uv_rect: Rect,
    },
    /// Push a clipping rectangle. Subsequent commands are clipped to this.
    PushClip {
        rect: Rect,
    },
    /// Pop the most recent clipping rectangle.
    PopClip,
    /// Push a 2-D transform matrix.
    PushTransform {
        transform: Mat3,
    },
    /// Pop the most recent transform.
    PopTransform,
    /// Fill with a gradient.
    GradientRect {
        rect: Rect,
        gradient: Gradient,
        corner_radii: CornerRadii,
    },
    /// Draw a triangle (for custom shapes).
    Triangle {
        p0: Vec2,
        p1: Vec2,
        p2: Vec2,
        color: Color,
    },
}

// ---------------------------------------------------------------------------
// DrawList
// ---------------------------------------------------------------------------

/// Collected draw commands for a single frame. The renderer backend iterates
/// this to produce GPU work.
#[derive(Debug, Clone)]
pub struct DrawList {
    pub commands: Vec<DrawCommand>,
    clip_stack: Vec<Rect>,
    transform_stack: Vec<Mat3>,
}

impl DrawList {
    pub fn new() -> Self {
        Self {
            commands: Vec::with_capacity(1024),
            clip_stack: Vec::new(),
            transform_stack: Vec::new(),
        }
    }

    /// Clear all commands for a new frame.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.clip_stack.clear();
        self.transform_stack.clear();
    }

    /// Number of draw commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Push a draw command.
    pub fn push(&mut self, cmd: DrawCommand) {
        self.commands.push(cmd);
    }

    // -- Convenience drawing methods ------------------------------------------

    /// Draw a solid rectangle.
    pub fn draw_rect(&mut self, rect: Rect, color: Color) {
        self.commands.push(DrawCommand::Rect {
            rect,
            color,
            corner_radii: CornerRadii::ZERO,
            border: Border::default(),
            shadow: None,
        });
    }

    /// Draw a rounded rectangle with optional border.
    pub fn draw_rounded_rect(
        &mut self,
        rect: Rect,
        color: Color,
        radii: CornerRadii,
        border: Border,
    ) {
        self.commands.push(DrawCommand::Rect {
            rect,
            color,
            corner_radii: radii,
            border,
            shadow: None,
        });
    }

    /// Draw a rounded rect with shadow.
    pub fn draw_rounded_rect_with_shadow(
        &mut self,
        rect: Rect,
        color: Color,
        radii: CornerRadii,
        border: Border,
        shadow: Shadow,
    ) {
        self.commands.push(DrawCommand::Rect {
            rect,
            color,
            corner_radii: radii,
            border,
            shadow: Some(shadow),
        });
    }

    /// Draw a filled circle.
    pub fn draw_circle(&mut self, center: Vec2, radius: f32, color: Color) {
        self.commands.push(DrawCommand::Circle {
            center,
            radius,
            color,
            border: Border::default(),
        });
    }

    /// Draw a line.
    pub fn draw_line(&mut self, start: Vec2, end: Vec2, color: Color, thickness: f32) {
        self.commands.push(DrawCommand::Line {
            start,
            end,
            color,
            thickness,
        });
    }

    /// Draw text at a position.
    pub fn draw_text(
        &mut self,
        text: &str,
        position: Vec2,
        font_size: f32,
        color: Color,
    ) {
        self.commands.push(DrawCommand::Text {
            text: text.to_string(),
            position,
            font_size,
            color,
            font_id: 0,
            max_width: None,
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });
    }

    /// Draw text with full parameters.
    pub fn draw_text_ex(
        &mut self,
        text: &str,
        position: Vec2,
        font_size: f32,
        color: Color,
        font_id: u32,
        max_width: Option<f32>,
        align: TextAlign,
        vertical_align: TextVerticalAlign,
    ) {
        self.commands.push(DrawCommand::Text {
            text: text.to_string(),
            position,
            font_size,
            color,
            font_id,
            max_width,
            align,
            vertical_align,
        });
    }

    /// Draw an image.
    pub fn draw_image(&mut self, rect: Rect, texture: TextureId, tint: Color) {
        self.commands.push(DrawCommand::Image {
            rect,
            texture,
            tint,
            corner_radii: CornerRadii::ZERO,
            scale_mode: ImageScaleMode::Stretch,
            uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
        });
    }

    /// Push a clip rect.
    pub fn push_clip(&mut self, rect: Rect) {
        // Intersect with current clip if one exists.
        let clipped = if let Some(current) = self.clip_stack.last() {
            Rect::new(
                Vec2::new(rect.min.x.max(current.min.x), rect.min.y.max(current.min.y)),
                Vec2::new(rect.max.x.min(current.max.x), rect.max.y.min(current.max.y)),
            )
        } else {
            rect
        };
        self.clip_stack.push(clipped);
        self.commands.push(DrawCommand::PushClip { rect: clipped });
    }

    /// Pop a clip rect.
    pub fn pop_clip(&mut self) {
        self.clip_stack.pop();
        self.commands.push(DrawCommand::PopClip);
    }

    /// Push a 2-D transform.
    pub fn push_transform(&mut self, transform: Mat3) {
        self.transform_stack.push(transform);
        self.commands.push(DrawCommand::PushTransform { transform });
    }

    /// Pop a 2-D transform.
    pub fn pop_transform(&mut self) {
        self.transform_stack.pop();
        self.commands.push(DrawCommand::PopTransform);
    }

    /// Sort commands by z-order. Stable sort preserves insertion order for
    /// equal z-values.
    pub fn sort_by_z(&mut self, z_values: &[i32]) {
        if z_values.len() != self.commands.len() {
            return;
        }
        let mut indexed: Vec<(usize, i32)> =
            z_values.iter().copied().enumerate().collect();
        indexed.sort_by_key(|(_, z)| *z);

        let old_commands = std::mem::take(&mut self.commands);
        self.commands = indexed
            .into_iter()
            .map(|(i, _)| old_commands[i].clone())
            .collect();
    }

    /// Attempt to batch adjacent rect commands that share the same texture
    /// and clipping state. Returns the number of batches.
    pub fn batch_commands(&self) -> Vec<DrawBatch> {
        let mut batches: Vec<DrawBatch> = Vec::new();

        for (i, cmd) in self.commands.iter().enumerate() {
            let kind = DrawBatchKind::from_command(cmd);
            if let Some(last) = batches.last_mut() {
                if last.kind == kind {
                    last.end = i + 1;
                    continue;
                }
            }
            batches.push(DrawBatch {
                kind,
                start: i,
                end: i + 1,
            });
        }

        batches
    }
}

impl Default for DrawList {
    fn default() -> Self {
        Self::new()
    }
}

/// A contiguous range of draw commands that share the same batch kind.
#[derive(Debug, Clone)]
pub struct DrawBatch {
    pub kind: DrawBatchKind,
    pub start: usize,
    pub end: usize,
}

impl DrawBatch {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// Categories for batching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawBatchKind {
    Geometry,
    Text,
    Image,
    Clip,
    Transform,
}

impl DrawBatchKind {
    fn from_command(cmd: &DrawCommand) -> Self {
        match cmd {
            DrawCommand::Rect { .. }
            | DrawCommand::Circle { .. }
            | DrawCommand::Line { .. }
            | DrawCommand::Polyline { .. }
            | DrawCommand::Triangle { .. }
            | DrawCommand::GradientRect { .. } => Self::Geometry,
            DrawCommand::Text { .. } => Self::Text,
            DrawCommand::Image { .. } => Self::Image,
            DrawCommand::PushClip { .. } | DrawCommand::PopClip => Self::Clip,
            DrawCommand::PushTransform { .. } | DrawCommand::PopTransform => Self::Transform,
        }
    }
}

// ---------------------------------------------------------------------------
// UIRenderer trait
// ---------------------------------------------------------------------------

/// Backend trait that consumes a [`DrawList`] and produces actual rendered
/// output. Implementations live in platform/renderer-specific crates.
pub trait UIRenderer {
    /// Called at the start of a frame.
    fn begin_frame(&mut self, screen_size: Vec2);

    /// Process the draw list and submit GPU work.
    fn render(&mut self, draw_list: &DrawList);

    /// Called at the end of a frame.
    fn end_frame(&mut self);

    /// Create or update a texture for UI use. Returns a [`TextureId`].
    fn create_texture(&mut self, width: u32, height: u32, data: &[u8]) -> TextureId;

    /// Destroy a previously created texture.
    fn destroy_texture(&mut self, id: TextureId);

    /// Returns the current screen/viewport size.
    fn screen_size(&self) -> Vec2;
}
