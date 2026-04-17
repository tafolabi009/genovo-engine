//! Slate-style brush system for the Genovo UI framework.
//!
//! A brush defines how a surface is filled: with a solid colour, a texture,
//! a nine-slice image, a gradient, a procedural rounded rectangle, or nothing
//! at all. This is the styling foundation for a professional editor UI.
//!
//! The system also provides composite style sets for common widget types
//! (buttons, sliders, checkboxes, etc.) so that a single `StyleSet` can
//! theme every widget in the editor consistently.
//!
//! # Nine-Slice Rendering
//!
//! Nine-slice (also called border-image) rendering divides a texture into a
//! 3x3 grid. The corners are drawn at their natural size, the edges are
//! stretched or tiled along one axis, and the centre is stretched/tiled in
//! both axes. This technique allows a single small texture to be used for
//! variable-size panels and buttons without distorting the corners.
//!
//! ```text
//!  ┌────┬──────────────────┬────┐
//!  │ TL │   Top Edge       │ TR │  <-- margin.top
//!  ├────┼──────────────────┼────┤
//!  │ L  │                  │ R  │
//!  │ E  │     Centre       │ E  │
//!  │    │                  │    │
//!  ├────┼──────────────────┼────┤
//!  │ BL │   Bottom Edge    │ BR │  <-- margin.bottom
//!  └────┴──────────────────┴────┘
//!  ^                        ^
//!  margin.left         margin.right
//! ```

use std::collections::HashMap;

use glam::Vec2;
use serde::{Deserialize, Serialize};

use genovo_core::Rect;

use crate::render_commands::{
    Border, Color, CornerRadii, DrawCommand, DrawList, ImageScaleMode, TextureId,
};

// ---------------------------------------------------------------------------
// Margin type (local to this module to avoid name collision)
// ---------------------------------------------------------------------------

/// Edge offsets used for nine-slice margins and padding.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BrushMargin {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl BrushMargin {
    pub const ZERO: Self = Self {
        left: 0.0,
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
    };

    pub fn all(value: f32) -> Self {
        Self {
            left: value,
            top: value,
            right: value,
            bottom: value,
        }
    }

    pub fn symmetric(horizontal: f32, vertical: f32) -> Self {
        Self {
            left: horizontal,
            top: vertical,
            right: horizontal,
            bottom: vertical,
        }
    }

    pub fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }

    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }

    /// Inset a rect by the margin values.
    pub fn inset_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            Vec2::new(rect.min.x + self.left, rect.min.y + self.top),
            Vec2::new(rect.max.x - self.right, rect.max.y - self.bottom),
        )
    }

    /// Expand a rect by the margin values.
    pub fn expand_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            Vec2::new(rect.min.x - self.left, rect.min.y - self.top),
            Vec2::new(rect.max.x + self.right, rect.max.y + self.bottom),
        )
    }
}

impl Default for BrushMargin {
    fn default() -> Self {
        Self::ZERO
    }
}

// ---------------------------------------------------------------------------
// CornerRadius — per-corner radius
// ---------------------------------------------------------------------------

/// Per-corner radius specification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CornerRadius {
    pub top_left: f32,
    pub top_right: f32,
    pub bottom_left: f32,
    pub bottom_right: f32,
}

impl CornerRadius {
    pub const ZERO: Self = Self {
        top_left: 0.0,
        top_right: 0.0,
        bottom_left: 0.0,
        bottom_right: 0.0,
    };

    pub fn all(radius: f32) -> Self {
        Self {
            top_left: radius,
            top_right: radius,
            bottom_left: radius,
            bottom_right: radius,
        }
    }

    pub fn new(tl: f32, tr: f32, bl: f32, br: f32) -> Self {
        Self {
            top_left: tl,
            top_right: tr,
            bottom_left: bl,
            bottom_right: br,
        }
    }

    /// Top corners only (e.g., for tab headers).
    pub fn top(radius: f32) -> Self {
        Self {
            top_left: radius,
            top_right: radius,
            bottom_left: 0.0,
            bottom_right: 0.0,
        }
    }

    /// Bottom corners only.
    pub fn bottom(radius: f32) -> Self {
        Self {
            top_left: 0.0,
            top_right: 0.0,
            bottom_left: radius,
            bottom_right: radius,
        }
    }

    /// Left corners only.
    pub fn left(radius: f32) -> Self {
        Self {
            top_left: radius,
            top_right: 0.0,
            bottom_left: radius,
            bottom_right: 0.0,
        }
    }

    /// Right corners only.
    pub fn right(radius: f32) -> Self {
        Self {
            top_left: 0.0,
            top_right: radius,
            bottom_left: 0.0,
            bottom_right: radius,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.top_left == 0.0
            && self.top_right == 0.0
            && self.bottom_left == 0.0
            && self.bottom_right == 0.0
    }

    /// Convert to the render_commands CornerRadii type.
    pub fn to_corner_radii(&self) -> CornerRadii {
        CornerRadii::new(
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        )
    }

    /// Clamp all radii to fit within a rect of the given size.
    pub fn clamped_to_size(&self, width: f32, height: f32) -> Self {
        let max_r = (width * 0.5).min(height * 0.5).max(0.0);
        Self {
            top_left: self.top_left.min(max_r),
            top_right: self.top_right.min(max_r),
            bottom_left: self.bottom_left.min(max_r),
            bottom_right: self.bottom_right.min(max_r),
        }
    }
}

impl Default for CornerRadius {
    fn default() -> Self {
        Self::ZERO
    }
}

// ---------------------------------------------------------------------------
// TilingMode
// ---------------------------------------------------------------------------

/// How a texture brush fills its target rect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TilingMode {
    /// Stretch the texture to fill the rect.
    Stretch,
    /// Tile (repeat) the texture.
    Tile,
    /// Use nine-slice rendering (texture has a 3x3 grid defined by margins).
    NineSlice,
}

impl Default for TilingMode {
    fn default() -> Self {
        Self::Stretch
    }
}

// ---------------------------------------------------------------------------
// GradientDir
// ---------------------------------------------------------------------------

/// Direction of a linear gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GradientDir {
    /// Left to right.
    Horizontal,
    /// Top to bottom.
    Vertical,
    /// Top-left to bottom-right.
    Diagonal,
    /// Top-right to bottom-left.
    DiagonalReverse,
}

impl Default for GradientDir {
    fn default() -> Self {
        Self::Vertical
    }
}

// ---------------------------------------------------------------------------
// TextShadow / TextOutline
// ---------------------------------------------------------------------------

/// Shadow applied to text.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextShadow {
    pub color: Color,
    pub offset: Vec2,
    pub blur_radius: f32,
}

impl TextShadow {
    pub fn new(color: Color, offset: Vec2, blur_radius: f32) -> Self {
        Self {
            color,
            offset,
            blur_radius,
        }
    }

    pub fn simple(color: Color) -> Self {
        Self {
            color,
            offset: Vec2::new(1.0, 1.0),
            blur_radius: 0.0,
        }
    }
}

impl Default for TextShadow {
    fn default() -> Self {
        Self {
            color: Color::new(0.0, 0.0, 0.0, 0.5),
            offset: Vec2::new(1.0, 1.0),
            blur_radius: 0.0,
        }
    }
}

/// Outline applied to text.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextOutline {
    pub color: Color,
    pub width: f32,
}

impl TextOutline {
    pub fn new(color: Color, width: f32) -> Self {
        Self { color, width }
    }
}

impl Default for TextOutline {
    fn default() -> Self {
        Self {
            color: Color::BLACK,
            width: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Brush enum — the core style primitive
// ---------------------------------------------------------------------------

/// A brush defines how a surface is filled. This is the fundamental styling
/// primitive, analogous to Unreal/Slate's FSlateBrush.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Brush {
    /// No fill at all (fully transparent).
    None,

    /// A flat solid colour fill.
    Solid {
        color: Color,
    },

    /// A texture fill, with tinting and tiling control.
    Image {
        texture_id: u32,
        tiling: TilingMode,
        tint: Color,
    },

    /// A nine-slice (box) texture: corners stay at natural size, edges and
    /// centre stretch. Used for panels and buttons.
    Box {
        texture_id: u32,
        margin: BrushMargin,
        tint: Color,
    },

    /// A nine-slice border: like Box but only the border region is drawn,
    /// the centre is transparent.
    Border {
        texture_id: u32,
        margin: BrushMargin,
        tint: Color,
    },

    /// A procedurally generated rounded rectangle.
    RoundedBox {
        color: Color,
        radius: CornerRadius,
        border_color: Color,
        border_width: f32,
    },

    /// A linear gradient between two colours.
    Gradient {
        start_color: Color,
        end_color: Color,
        direction: GradientDir,
    },
}

impl Brush {
    // -- Convenience constructors -------------------------------------------

    /// Solid colour brush.
    pub fn solid(color: Color) -> Self {
        Self::Solid { color }
    }

    /// Transparent / invisible brush.
    pub fn none() -> Self {
        Self::None
    }

    /// Image brush with default tiling.
    pub fn image(texture_id: u32) -> Self {
        Self::Image {
            texture_id,
            tiling: TilingMode::Stretch,
            tint: Color::WHITE,
        }
    }

    /// Image brush with tinting.
    pub fn image_tinted(texture_id: u32, tint: Color) -> Self {
        Self::Image {
            texture_id,
            tiling: TilingMode::Stretch,
            tint,
        }
    }

    /// Nine-slice box brush.
    pub fn nine_slice(texture_id: u32, margin: BrushMargin) -> Self {
        Self::Box {
            texture_id,
            margin,
            tint: Color::WHITE,
        }
    }

    /// Nine-slice border brush.
    pub fn nine_slice_border(texture_id: u32, margin: BrushMargin) -> Self {
        Self::Border {
            texture_id,
            margin,
            tint: Color::WHITE,
        }
    }

    /// Rounded box brush.
    pub fn rounded(color: Color, radius: f32) -> Self {
        Self::RoundedBox {
            color,
            radius: CornerRadius::all(radius),
            border_color: Color::TRANSPARENT,
            border_width: 0.0,
        }
    }

    /// Rounded box with border.
    pub fn rounded_with_border(
        color: Color,
        radius: f32,
        border_color: Color,
        border_width: f32,
    ) -> Self {
        Self::RoundedBox {
            color,
            radius: CornerRadius::all(radius),
            border_color,
            border_width,
        }
    }

    /// Horizontal gradient.
    pub fn gradient_h(left: Color, right: Color) -> Self {
        Self::Gradient {
            start_color: left,
            end_color: right,
            direction: GradientDir::Horizontal,
        }
    }

    /// Vertical gradient.
    pub fn gradient_v(top: Color, bottom: Color) -> Self {
        Self::Gradient {
            start_color: top,
            end_color: bottom,
            direction: GradientDir::Vertical,
        }
    }

    /// Returns `true` if this brush is None/invisible.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Returns the primary colour of this brush (solid, tint, start, etc.).
    pub fn primary_color(&self) -> Color {
        match self {
            Self::None => Color::TRANSPARENT,
            Self::Solid { color } => *color,
            Self::Image { tint, .. } => *tint,
            Self::Box { tint, .. } => *tint,
            Self::Border { tint, .. } => *tint,
            Self::RoundedBox { color, .. } => *color,
            Self::Gradient { start_color, .. } => *start_color,
        }
    }

    /// Tint the brush by multiplying with a colour.
    pub fn with_tint(&self, tint: Color) -> Self {
        match self {
            Self::None => Self::None,
            Self::Solid { color } => Self::Solid {
                color: multiply_colors(*color, tint),
            },
            Self::Image {
                texture_id,
                tiling,
                tint: current,
            } => Self::Image {
                texture_id: *texture_id,
                tiling: *tiling,
                tint: multiply_colors(*current, tint),
            },
            Self::Box {
                texture_id,
                margin,
                tint: current,
            } => Self::Box {
                texture_id: *texture_id,
                margin: *margin,
                tint: multiply_colors(*current, tint),
            },
            Self::Border {
                texture_id,
                margin,
                tint: current,
            } => Self::Border {
                texture_id: *texture_id,
                margin: *margin,
                tint: multiply_colors(*current, tint),
            },
            Self::RoundedBox {
                color,
                radius,
                border_color,
                border_width,
            } => Self::RoundedBox {
                color: multiply_colors(*color, tint),
                radius: *radius,
                border_color: *border_color,
                border_width: *border_width,
            },
            Self::Gradient {
                start_color,
                end_color,
                direction,
            } => Self::Gradient {
                start_color: multiply_colors(*start_color, tint),
                end_color: multiply_colors(*end_color, tint),
                direction: *direction,
            },
        }
    }

    /// Render this brush into a draw list at the given rect.
    pub fn draw(&self, draw_list: &mut DrawList, rect: Rect) {
        match self {
            Self::None => {}
            Self::Solid { color } => {
                draw_list.draw_rect(rect, *color);
            }
            Self::Image {
                texture_id,
                tiling,
                tint,
            } => match tiling {
                TilingMode::Stretch => {
                    draw_list.draw_image(rect, TextureId(*texture_id as u64), *tint);
                }
                TilingMode::Tile => {
                    // For tiled rendering, we emit the image with tile scale mode
                    draw_list.push(DrawCommand::Image {
                        rect,
                        texture: TextureId(*texture_id as u64),
                        tint: *tint,
                        corner_radii: CornerRadii::ZERO,
                        scale_mode: ImageScaleMode::Tile,
                        uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
                    });
                }
                TilingMode::NineSlice => {
                    // Treat as full nine-slice with uniform margin
                    let margin = BrushMargin::all(16.0);
                    draw_nine_slice(draw_list, rect, TextureId(*texture_id as u64), *tint, &margin, false);
                }
            },
            Self::Box {
                texture_id,
                margin,
                tint,
            } => {
                draw_nine_slice(
                    draw_list,
                    rect,
                    TextureId(*texture_id as u64),
                    *tint,
                    margin,
                    false, // draw centre
                );
            }
            Self::Border {
                texture_id,
                margin,
                tint,
            } => {
                draw_nine_slice(
                    draw_list,
                    rect,
                    TextureId(*texture_id as u64),
                    *tint,
                    margin,
                    true, // skip centre
                );
            }
            Self::RoundedBox {
                color,
                radius,
                border_color,
                border_width,
            } => {
                let clamped = radius.clamped_to_size(rect.width(), rect.height());
                let radii = clamped.to_corner_radii();
                if *border_width > 0.0 {
                    draw_list.draw_rounded_rect(
                        rect,
                        *color,
                        radii,
                        Border::new(*border_color, *border_width),
                    );
                } else {
                    draw_list.draw_rounded_rect(
                        rect,
                        *color,
                        radii,
                        Border::default(),
                    );
                }
            }
            Self::Gradient {
                start_color,
                end_color,
                direction,
            } => {
                draw_gradient(draw_list, rect, *start_color, *end_color, *direction);
            }
        }
    }
}

impl Default for Brush {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// Nine-slice rendering
// ---------------------------------------------------------------------------

/// Draw a nine-slice image. `skip_centre` is true for border-only rendering.
///
/// The texture is divided into a 3x3 grid by the margin. The UV coordinates
/// for each slice are computed from the margin relative to the assumed texture
/// size (1.0 x 1.0 in normalised UV space, with margin in pixels mapped to
/// a fraction of the texture).
fn draw_nine_slice(
    draw_list: &mut DrawList,
    rect: Rect,
    texture: TextureId,
    tint: Color,
    margin: &BrushMargin,
    skip_centre: bool,
) {
    let w = rect.width();
    let h = rect.height();

    // If the rect is smaller than the margins, fall back to stretch.
    if w < margin.horizontal() || h < margin.vertical() {
        if !skip_centre {
            draw_list.draw_image(rect, texture, tint);
        }
        return;
    }

    // Pixel positions for the grid cuts.
    let x0 = rect.min.x;
    let x1 = rect.min.x + margin.left;
    let x2 = rect.max.x - margin.right;
    let x3 = rect.max.x;

    let y0 = rect.min.y;
    let y1 = rect.min.y + margin.top;
    let y2 = rect.max.y - margin.bottom;
    let y3 = rect.max.y;

    // UV positions (assuming margin maps linearly to UV).
    // We need to know the texture size to compute exact UVs. Since we may
    // not know it, we estimate from the margin ratios. This is a common
    // heuristic: assume the margin is the same in pixels as in texels.
    let tex_w = margin.left + margin.right + 1.0; // minimum texture width
    let tex_h = margin.top + margin.bottom + 1.0;

    let u0 = 0.0_f32;
    let u1 = margin.left / tex_w;
    let u2 = 1.0 - margin.right / tex_w;
    let u3 = 1.0_f32;

    let v0 = 0.0_f32;
    let v1 = margin.top / tex_h;
    let v2 = 1.0 - margin.bottom / tex_h;
    let v3 = 1.0_f32;

    // Helper to emit one slice.
    let mut emit_slice = |px_rect: Rect, uv_rect: Rect| {
        if px_rect.width() <= 0.0 || px_rect.height() <= 0.0 {
            return;
        }
        draw_list.push(DrawCommand::Image {
            rect: px_rect,
            texture,
            tint,
            corner_radii: CornerRadii::ZERO,
            scale_mode: ImageScaleMode::Stretch,
            uv_rect,
        });
    };

    // Top-left corner
    emit_slice(
        Rect::new(Vec2::new(x0, y0), Vec2::new(x1, y1)),
        Rect::new(Vec2::new(u0, v0), Vec2::new(u1, v1)),
    );
    // Top edge
    emit_slice(
        Rect::new(Vec2::new(x1, y0), Vec2::new(x2, y1)),
        Rect::new(Vec2::new(u1, v0), Vec2::new(u2, v1)),
    );
    // Top-right corner
    emit_slice(
        Rect::new(Vec2::new(x2, y0), Vec2::new(x3, y1)),
        Rect::new(Vec2::new(u2, v0), Vec2::new(u3, v1)),
    );

    // Left edge
    emit_slice(
        Rect::new(Vec2::new(x0, y1), Vec2::new(x1, y2)),
        Rect::new(Vec2::new(u0, v1), Vec2::new(u1, v2)),
    );
    // Centre
    if !skip_centre {
        emit_slice(
            Rect::new(Vec2::new(x1, y1), Vec2::new(x2, y2)),
            Rect::new(Vec2::new(u1, v1), Vec2::new(u2, v2)),
        );
    }
    // Right edge
    emit_slice(
        Rect::new(Vec2::new(x2, y1), Vec2::new(x3, y2)),
        Rect::new(Vec2::new(u2, v1), Vec2::new(u3, v2)),
    );

    // Bottom-left corner
    emit_slice(
        Rect::new(Vec2::new(x0, y2), Vec2::new(x1, y3)),
        Rect::new(Vec2::new(u0, v2), Vec2::new(u1, v3)),
    );
    // Bottom edge
    emit_slice(
        Rect::new(Vec2::new(x1, y2), Vec2::new(x2, y3)),
        Rect::new(Vec2::new(u1, v2), Vec2::new(u2, v3)),
    );
    // Bottom-right corner
    emit_slice(
        Rect::new(Vec2::new(x2, y2), Vec2::new(x3, y3)),
        Rect::new(Vec2::new(u2, v2), Vec2::new(u3, v3)),
    );
}

/// Draw a gradient rectangle by splitting into vertical or horizontal strips.
fn draw_gradient(
    draw_list: &mut DrawList,
    rect: Rect,
    start: Color,
    end: Color,
    direction: GradientDir,
) {
    let steps = 32;
    let w = rect.width();
    let h = rect.height();

    match direction {
        GradientDir::Horizontal => {
            let strip_w = w / steps as f32;
            for i in 0..steps {
                let t0 = i as f32 / steps as f32;
                let t1 = (i + 1) as f32 / steps as f32;
                let c0 = start.lerp(end, t0);
                let c1 = start.lerp(end, t1);
                let avg_c = c0.lerp(c1, 0.5);
                let strip_rect = Rect::new(
                    Vec2::new(rect.min.x + strip_w * i as f32, rect.min.y),
                    Vec2::new(rect.min.x + strip_w * (i + 1) as f32, rect.max.y),
                );
                draw_list.draw_rect(strip_rect, avg_c);
            }
        }
        GradientDir::Vertical => {
            let strip_h = h / steps as f32;
            for i in 0..steps {
                let t0 = i as f32 / steps as f32;
                let t1 = (i + 1) as f32 / steps as f32;
                let c0 = start.lerp(end, t0);
                let c1 = start.lerp(end, t1);
                let avg_c = c0.lerp(c1, 0.5);
                let strip_rect = Rect::new(
                    Vec2::new(rect.min.x, rect.min.y + strip_h * i as f32),
                    Vec2::new(rect.max.x, rect.min.y + strip_h * (i + 1) as f32),
                );
                draw_list.draw_rect(strip_rect, avg_c);
            }
        }
        GradientDir::Diagonal | GradientDir::DiagonalReverse => {
            // Diagonal gradients are approximated by rows with shifted t
            let strip_h = h / steps as f32;
            for i in 0..steps {
                let base_t = i as f32 / steps as f32;
                // Split each row into sub-strips for diagonal effect
                let sub_strips = 8;
                let sub_w = w / sub_strips as f32;
                for j in 0..sub_strips {
                    let x_t = j as f32 / sub_strips as f32;
                    let combined_t = match direction {
                        GradientDir::Diagonal => (base_t + x_t) * 0.5,
                        GradientDir::DiagonalReverse => (base_t + (1.0 - x_t)) * 0.5,
                        _ => unreachable!(),
                    };
                    let c = start.lerp(end, combined_t.clamp(0.0, 1.0));
                    let sub_rect = Rect::new(
                        Vec2::new(
                            rect.min.x + sub_w * j as f32,
                            rect.min.y + strip_h * i as f32,
                        ),
                        Vec2::new(
                            rect.min.x + sub_w * (j + 1) as f32,
                            rect.min.y + strip_h * (i + 1) as f32,
                        ),
                    );
                    draw_list.draw_rect(sub_rect, c);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Colour math
// ---------------------------------------------------------------------------

/// Multiply two colours component-wise.
fn multiply_colors(a: Color, b: Color) -> Color {
    Color::new(a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a)
}

// ---------------------------------------------------------------------------
// Style structs — per-widget styling
// ---------------------------------------------------------------------------

/// Button visual style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ButtonStyle {
    pub normal: Brush,
    pub hovered: Brush,
    pub pressed: Brush,
    pub disabled: Brush,
    pub padding: BrushMargin,
    pub text_color: Color,
    pub disabled_text_color: Color,
    pub font_size: f32,
    pub corner_radius: CornerRadius,
}

impl Default for ButtonStyle {
    fn default() -> Self {
        Self {
            normal: Brush::rounded(Color::from_hex("#45475A"), 4.0),
            hovered: Brush::rounded(Color::from_hex("#585B70"), 4.0),
            pressed: Brush::rounded(Color::from_hex("#313244"), 4.0),
            disabled: Brush::rounded(Color::from_hex("#313244"), 4.0),
            padding: BrushMargin::symmetric(12.0, 6.0),
            text_color: Color::from_hex("#CDD6F4"),
            disabled_text_color: Color::from_hex("#6C7086"),
            font_size: 14.0,
            corner_radius: CornerRadius::all(4.0),
        }
    }
}

/// Text visual style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrushTextStyle {
    pub font_size: f32,
    pub color: Color,
    pub shadow: Option<TextShadow>,
    pub outline: Option<TextOutline>,
    pub font_id: u32,
}

impl Default for BrushTextStyle {
    fn default() -> Self {
        Self {
            font_size: 14.0,
            color: Color::from_hex("#CDD6F4"),
            shadow: None,
            outline: None,
            font_id: 0,
        }
    }
}

/// Slider visual style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliderStyle {
    pub track: Brush,
    pub fill: Brush,
    pub thumb: Brush,
    pub thumb_hovered: Brush,
    pub track_height: f32,
    pub thumb_radius: f32,
}

impl Default for SliderStyle {
    fn default() -> Self {
        Self {
            track: Brush::rounded(Color::from_hex("#313244"), 2.0),
            fill: Brush::rounded(Color::from_hex("#89B4FA"), 2.0),
            thumb: Brush::rounded(Color::from_hex("#CDD6F4"), 6.0),
            thumb_hovered: Brush::rounded(Color::from_hex("#F5E0DC"), 6.0),
            track_height: 4.0,
            thumb_radius: 6.0,
        }
    }
}

/// Checkbox visual style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckboxStyle {
    pub checked: Brush,
    pub unchecked: Brush,
    pub indeterminate: Brush,
    pub check_color: Color,
    pub size: f32,
    pub corner_radius: f32,
}

impl Default for CheckboxStyle {
    fn default() -> Self {
        Self {
            checked: Brush::rounded_with_border(
                Color::from_hex("#89B4FA"),
                3.0,
                Color::from_hex("#89B4FA"),
                1.0,
            ),
            unchecked: Brush::rounded_with_border(
                Color::TRANSPARENT,
                3.0,
                Color::from_hex("#6C7086"),
                1.0,
            ),
            indeterminate: Brush::rounded_with_border(
                Color::from_hex("#89B4FA"),
                3.0,
                Color::from_hex("#89B4FA"),
                1.0,
            ),
            check_color: Color::from_hex("#1E1E2E"),
            size: 18.0,
            corner_radius: 3.0,
        }
    }
}

/// Scrollbar visual style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollbarStyle {
    pub track: Brush,
    pub thumb: Brush,
    pub thumb_hovered: Brush,
    pub thumb_pressed: Brush,
    pub width: f32,
    pub min_thumb_length: f32,
    pub corner_radius: f32,
}

impl Default for ScrollbarStyle {
    fn default() -> Self {
        Self {
            track: Brush::solid(Color::from_hex("#181825")),
            thumb: Brush::rounded(Color::from_hex("#45475A"), 4.0),
            thumb_hovered: Brush::rounded(Color::from_hex("#585B70"), 4.0),
            thumb_pressed: Brush::rounded(Color::from_hex("#6C7086"), 4.0),
            width: 10.0,
            min_thumb_length: 30.0,
            corner_radius: 4.0,
        }
    }
}

/// Panel visual style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelStyle {
    pub background: Brush,
    pub border: Brush,
    pub title_bg: Brush,
    pub padding: BrushMargin,
    pub title_text: BrushTextStyle,
    pub corner_radius: CornerRadius,
}

impl Default for PanelStyle {
    fn default() -> Self {
        Self {
            background: Brush::solid(Color::from_hex("#1E1E2E")),
            border: Brush::rounded_with_border(
                Color::TRANSPARENT,
                0.0,
                Color::from_hex("#313244"),
                1.0,
            ),
            title_bg: Brush::solid(Color::from_hex("#181825")),
            padding: BrushMargin::all(4.0),
            title_text: BrushTextStyle {
                font_size: 13.0,
                color: Color::from_hex("#CDD6F4"),
                shadow: None,
                outline: None,
                font_id: 0,
            },
            corner_radius: CornerRadius::all(4.0),
        }
    }
}

/// Tab visual style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabStyle {
    pub active: Brush,
    pub inactive: Brush,
    pub hovered: Brush,
    pub close_button: Brush,
    pub close_button_hovered: Brush,
    pub active_text: BrushTextStyle,
    pub inactive_text: BrushTextStyle,
    pub height: f32,
    pub corner_radius: CornerRadius,
}

impl Default for TabStyle {
    fn default() -> Self {
        Self {
            active: Brush::solid(Color::from_hex("#1E1E2E")),
            inactive: Brush::solid(Color::from_hex("#14141F")),
            hovered: Brush::solid(Color::from_hex("#252535")),
            close_button: Brush::none(),
            close_button_hovered: Brush::rounded(Color::from_hex("#F38BA8"), 2.0),
            active_text: BrushTextStyle {
                font_size: 13.0,
                color: Color::from_hex("#CDD6F4"),
                ..Default::default()
            },
            inactive_text: BrushTextStyle {
                font_size: 13.0,
                color: Color::from_hex("#6C7086"),
                ..Default::default()
            },
            height: 28.0,
            corner_radius: CornerRadius::top(4.0),
        }
    }
}

/// Text input visual style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputStyle {
    pub background: Brush,
    pub focused: Brush,
    pub text: BrushTextStyle,
    pub hint: BrushTextStyle,
    pub cursor_color: Color,
    pub selection_color: Color,
    pub padding: BrushMargin,
    pub corner_radius: CornerRadius,
}

impl Default for InputStyle {
    fn default() -> Self {
        Self {
            background: Brush::rounded_with_border(
                Color::from_hex("#313244"),
                3.0,
                Color::from_hex("#45475A"),
                1.0,
            ),
            focused: Brush::rounded_with_border(
                Color::from_hex("#313244"),
                3.0,
                Color::from_hex("#89B4FA"),
                1.5,
            ),
            text: BrushTextStyle {
                font_size: 14.0,
                color: Color::from_hex("#CDD6F4"),
                ..Default::default()
            },
            hint: BrushTextStyle {
                font_size: 14.0,
                color: Color::from_hex("#6C7086"),
                ..Default::default()
            },
            cursor_color: Color::from_hex("#CDD6F4"),
            selection_color: Color::new(0.53, 0.71, 0.98, 0.3),
            padding: BrushMargin::symmetric(8.0, 4.0),
            corner_radius: CornerRadius::all(3.0),
        }
    }
}

/// Tree view visual style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeViewStyle {
    pub row_background: Brush,
    pub row_hovered: Brush,
    pub row_selected: Brush,
    pub row_height: f32,
    pub indent_size: f32,
    pub expand_icon_size: f32,
    pub text: BrushTextStyle,
    pub selected_text: BrushTextStyle,
}

impl Default for TreeViewStyle {
    fn default() -> Self {
        Self {
            row_background: Brush::none(),
            row_hovered: Brush::solid(Color::new(1.0, 1.0, 1.0, 0.05)),
            row_selected: Brush::solid(Color::from_hex("#45475A")),
            row_height: 22.0,
            indent_size: 18.0,
            expand_icon_size: 12.0,
            text: BrushTextStyle::default(),
            selected_text: BrushTextStyle {
                color: Color::WHITE,
                ..Default::default()
            },
        }
    }
}

/// Tooltip visual style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipStyle {
    pub background: Brush,
    pub text: BrushTextStyle,
    pub padding: BrushMargin,
    pub max_width: f32,
    pub delay_seconds: f32,
}

impl Default for TooltipStyle {
    fn default() -> Self {
        Self {
            background: Brush::rounded_with_border(
                Color::from_hex("#313244"),
                4.0,
                Color::from_hex("#585B70"),
                1.0,
            ),
            text: BrushTextStyle {
                font_size: 12.0,
                color: Color::from_hex("#CDD6F4"),
                ..Default::default()
            },
            padding: BrushMargin::symmetric(8.0, 4.0),
            max_width: 300.0,
            delay_seconds: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// StyleSet — named collection of styles
// ---------------------------------------------------------------------------

/// A named collection of widget styles. This is the top-level theming object
/// that provides consistent styling across all widgets.
///
/// Style sets support inheritance: a child set can override specific entries
/// from its parent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StyleSet {
    pub name: String,
    pub button: ButtonStyle,
    pub text: BrushTextStyle,
    pub slider: SliderStyle,
    pub checkbox: CheckboxStyle,
    pub scrollbar: ScrollbarStyle,
    pub panel: PanelStyle,
    pub tab: TabStyle,
    pub input: InputStyle,
    pub tree_view: TreeViewStyle,
    pub tooltip: TooltipStyle,
    /// Custom named brushes for application-specific use.
    pub custom_brushes: HashMap<String, Brush>,
    /// Custom named text styles.
    pub custom_text_styles: HashMap<String, BrushTextStyle>,
    /// Parent style set name for inheritance.
    pub parent: Option<String>,
}

impl StyleSet {
    /// Create a new empty style set with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Premium dark theme preset (Catppuccin Mocha inspired).
    pub fn dark_theme() -> Self {
        Self {
            name: "Dark".to_string(),
            button: ButtonStyle::default(),
            text: BrushTextStyle::default(),
            slider: SliderStyle::default(),
            checkbox: CheckboxStyle::default(),
            scrollbar: ScrollbarStyle::default(),
            panel: PanelStyle::default(),
            tab: TabStyle::default(),
            input: InputStyle::default(),
            tree_view: TreeViewStyle::default(),
            tooltip: TooltipStyle::default(),
            custom_brushes: HashMap::new(),
            custom_text_styles: HashMap::new(),
            parent: None,
        }
    }

    /// Light theme preset (Catppuccin Latte inspired).
    pub fn light_theme() -> Self {
        Self {
            name: "Light".to_string(),
            button: ButtonStyle {
                normal: Brush::rounded(Color::from_hex("#ACB0BE"), 4.0),
                hovered: Brush::rounded(Color::from_hex("#BCC0CC"), 4.0),
                pressed: Brush::rounded(Color::from_hex("#9CA0B0"), 4.0),
                disabled: Brush::rounded(Color::from_hex("#CCD0DA"), 4.0),
                text_color: Color::from_hex("#4C4F69"),
                disabled_text_color: Color::from_hex("#9CA0B0"),
                ..Default::default()
            },
            text: BrushTextStyle {
                color: Color::from_hex("#4C4F69"),
                ..Default::default()
            },
            slider: SliderStyle {
                track: Brush::rounded(Color::from_hex("#CCD0DA"), 2.0),
                fill: Brush::rounded(Color::from_hex("#1E66F5"), 2.0),
                thumb: Brush::rounded(Color::from_hex("#4C4F69"), 6.0),
                thumb_hovered: Brush::rounded(Color::from_hex("#5C5F77"), 6.0),
                ..Default::default()
            },
            checkbox: CheckboxStyle {
                checked: Brush::rounded_with_border(
                    Color::from_hex("#1E66F5"),
                    3.0,
                    Color::from_hex("#1E66F5"),
                    1.0,
                ),
                unchecked: Brush::rounded_with_border(
                    Color::TRANSPARENT,
                    3.0,
                    Color::from_hex("#9CA0B0"),
                    1.0,
                ),
                check_color: Color::WHITE,
                ..Default::default()
            },
            panel: PanelStyle {
                background: Brush::solid(Color::from_hex("#EFF1F5")),
                border: Brush::rounded_with_border(
                    Color::TRANSPARENT,
                    0.0,
                    Color::from_hex("#9CA0B0"),
                    1.0,
                ),
                title_bg: Brush::solid(Color::from_hex("#DCE0E8")),
                title_text: BrushTextStyle {
                    color: Color::from_hex("#4C4F69"),
                    ..Default::default()
                },
                ..Default::default()
            },
            input: InputStyle {
                background: Brush::rounded_with_border(
                    Color::from_hex("#E6E9EF"),
                    3.0,
                    Color::from_hex("#9CA0B0"),
                    1.0,
                ),
                focused: Brush::rounded_with_border(
                    Color::from_hex("#E6E9EF"),
                    3.0,
                    Color::from_hex("#1E66F5"),
                    1.5,
                ),
                text: BrushTextStyle {
                    color: Color::from_hex("#4C4F69"),
                    ..Default::default()
                },
                hint: BrushTextStyle {
                    color: Color::from_hex("#9CA0B0"),
                    ..Default::default()
                },
                cursor_color: Color::from_hex("#4C4F69"),
                selection_color: Color::new(0.12, 0.4, 0.96, 0.25),
                ..Default::default()
            },
            custom_brushes: HashMap::new(),
            custom_text_styles: HashMap::new(),
            parent: None,
            ..Default::default()
        }
    }

    /// Register a custom named brush.
    pub fn set_brush(&mut self, name: impl Into<String>, brush: Brush) {
        self.custom_brushes.insert(name.into(), brush);
    }

    /// Get a custom named brush. Returns `None` if not found.
    pub fn get_brush(&self, name: &str) -> Option<&Brush> {
        self.custom_brushes.get(name)
    }

    /// Register a custom named text style.
    pub fn set_text_style(&mut self, name: impl Into<String>, style: BrushTextStyle) {
        self.custom_text_styles.insert(name.into(), style);
    }

    /// Get a custom named text style.
    pub fn get_text_style(&self, name: &str) -> Option<&BrushTextStyle> {
        self.custom_text_styles.get(name)
    }
}

// ---------------------------------------------------------------------------
// StyleManager — manages theme inheritance
// ---------------------------------------------------------------------------

/// Manages multiple style sets with inheritance.
pub struct StyleManager {
    /// All registered style sets.
    sets: HashMap<String, StyleSet>,
    /// The active style set name.
    active: String,
}

impl StyleManager {
    /// Create a new style manager with a default dark theme.
    pub fn new() -> Self {
        let mut sets = HashMap::new();
        sets.insert("Dark".to_string(), StyleSet::dark_theme());
        sets.insert("Light".to_string(), StyleSet::light_theme());

        Self {
            sets,
            active: "Dark".to_string(),
        }
    }

    /// Register a style set.
    pub fn register(&mut self, set: StyleSet) {
        self.sets.insert(set.name.clone(), set);
    }

    /// Set the active style set by name.
    pub fn set_active(&mut self, name: &str) -> bool {
        if self.sets.contains_key(name) {
            self.active = name.to_string();
            true
        } else {
            false
        }
    }

    /// Get the active style set.
    pub fn active(&self) -> &StyleSet {
        self.sets
            .get(&self.active)
            .unwrap_or_else(|| self.sets.values().next().unwrap())
    }

    /// Get a style set by name.
    pub fn get(&self, name: &str) -> Option<&StyleSet> {
        self.sets.get(name)
    }

    /// Get a mutable style set by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut StyleSet> {
        self.sets.get_mut(name)
    }

    /// Returns a list of all registered style set names.
    pub fn names(&self) -> Vec<&str> {
        self.sets.keys().map(|s| s.as_str()).collect()
    }

    /// Resolve a brush from the active set, falling back to parent sets.
    pub fn resolve_brush(&self, name: &str) -> Option<&Brush> {
        let mut current_name = Some(self.active.as_str());
        while let Some(set_name) = current_name {
            if let Some(set) = self.sets.get(set_name) {
                if let Some(brush) = set.custom_brushes.get(name) {
                    return Some(brush);
                }
                current_name = set.parent.as_deref();
            } else {
                break;
            }
        }
        None
    }
}

impl Default for StyleManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brush_solid() {
        let brush = Brush::solid(Color::RED);
        assert_eq!(brush.primary_color(), Color::RED);
        assert!(!brush.is_none());
    }

    #[test]
    fn test_brush_none() {
        let brush = Brush::none();
        assert!(brush.is_none());
    }

    #[test]
    fn test_brush_tint() {
        let brush = Brush::solid(Color::WHITE);
        let tinted = brush.with_tint(Color::new(1.0, 0.0, 0.0, 1.0));
        match tinted {
            Brush::Solid { color } => {
                assert!((color.r - 1.0).abs() < 0.001);
                assert!(color.g.abs() < 0.001);
                assert!(color.b.abs() < 0.001);
            }
            _ => panic!("Expected solid brush"),
        }
    }

    #[test]
    fn test_corner_radius() {
        let r = CornerRadius::all(10.0);
        assert_eq!(r.top_left, 10.0);
        assert_eq!(r.top_right, 10.0);
        assert!(!r.is_zero());

        let clamped = r.clamped_to_size(8.0, 8.0);
        assert_eq!(clamped.top_left, 4.0);
    }

    #[test]
    fn test_brush_margin_inset() {
        let margin = BrushMargin::all(10.0);
        let rect = Rect::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        let inset = margin.inset_rect(rect);
        assert_eq!(inset.min.x, 10.0);
        assert_eq!(inset.min.y, 10.0);
        assert_eq!(inset.max.x, 90.0);
        assert_eq!(inset.max.y, 90.0);
    }

    #[test]
    fn test_brush_draw_solid() {
        let mut draw_list = DrawList::new();
        let brush = Brush::solid(Color::BLUE);
        let rect = Rect::new(Vec2::ZERO, Vec2::new(100.0, 50.0));
        brush.draw(&mut draw_list, rect);
        assert_eq!(draw_list.len(), 1);
    }

    #[test]
    fn test_brush_draw_rounded_box() {
        let mut draw_list = DrawList::new();
        let brush = Brush::rounded_with_border(
            Color::from_hex("#333333"),
            8.0,
            Color::from_hex("#666666"),
            2.0,
        );
        let rect = Rect::new(Vec2::ZERO, Vec2::new(200.0, 100.0));
        brush.draw(&mut draw_list, rect);
        assert_eq!(draw_list.len(), 1);
    }

    #[test]
    fn test_nine_slice_draw() {
        let mut draw_list = DrawList::new();
        let brush = Brush::nine_slice(1, BrushMargin::all(8.0));
        let rect = Rect::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        brush.draw(&mut draw_list, rect);
        // Nine-slice produces 9 image commands
        assert_eq!(draw_list.len(), 9);
    }

    #[test]
    fn test_nine_slice_border_only() {
        let mut draw_list = DrawList::new();
        let brush = Brush::nine_slice_border(1, BrushMargin::all(8.0));
        let rect = Rect::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        brush.draw(&mut draw_list, rect);
        // Nine-slice border skips centre = 8 commands
        assert_eq!(draw_list.len(), 8);
    }

    #[test]
    fn test_gradient_draw() {
        let mut draw_list = DrawList::new();
        let brush = Brush::gradient_v(Color::RED, Color::BLUE);
        let rect = Rect::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        brush.draw(&mut draw_list, rect);
        assert!(draw_list.len() > 0);
    }

    #[test]
    fn test_style_set_dark() {
        let theme = StyleSet::dark_theme();
        assert_eq!(theme.name, "Dark");
        assert!(theme.button.font_size > 0.0);
    }

    #[test]
    fn test_style_set_light() {
        let theme = StyleSet::light_theme();
        assert_eq!(theme.name, "Light");
    }

    #[test]
    fn test_style_set_custom_brush() {
        let mut theme = StyleSet::dark_theme();
        theme.set_brush("my_brush", Brush::solid(Color::GREEN));
        assert!(theme.get_brush("my_brush").is_some());
        assert!(theme.get_brush("nonexistent").is_none());
    }

    #[test]
    fn test_style_manager() {
        let mut mgr = StyleManager::new();
        assert_eq!(mgr.active().name, "Dark");

        mgr.set_active("Light");
        assert_eq!(mgr.active().name, "Light");

        assert!(!mgr.set_active("Nonexistent"));
        assert_eq!(mgr.active().name, "Light");
    }

    #[test]
    fn test_style_manager_resolve_brush() {
        let mut mgr = StyleManager::new();

        // Add a custom brush to the dark theme
        if let Some(dark) = mgr.get_mut("Dark") {
            dark.set_brush("accent", Brush::solid(Color::from_hex("#89B4FA")));
        }

        mgr.set_active("Dark");
        let brush = mgr.resolve_brush("accent");
        assert!(brush.is_some());
    }

    #[test]
    fn test_text_shadow() {
        let shadow = TextShadow::simple(Color::BLACK);
        assert_eq!(shadow.offset, Vec2::new(1.0, 1.0));
        assert_eq!(shadow.blur_radius, 0.0);
    }
}
