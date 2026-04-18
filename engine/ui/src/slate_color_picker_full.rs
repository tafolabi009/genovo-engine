//! Full colour picker for the Genovo Slate UI, inspired by Unreal Engine.
//!
//! Provides:
//! - HSV wheel (hue ring + saturation/value square)
//! - RGB sliders (0-255)
//! - HSV sliders
//! - Hex input (#RRGGBB)
//! - Alpha slider with checkerboard
//! - Eye-dropper mode (pick colour from screen)
//! - Recent colours palette (last 16)
//! - Preset colours (16 basic colours)
//! - sRGB / linear toggle
//! - Colour comparison (old vs new)

use glam::Vec2;
use genovo_core::Rect;

use crate::core::{KeyCode, KeyModifiers, MouseButton, Padding, UIEvent, UIId};
use crate::render_commands::{
    Border as BorderSpec, Color, CornerRadii, DrawCommand, DrawList, GradientStop, Gradient,
    TextAlign, TextVerticalAlign, TextureId,
};
use crate::slate_widgets::EventReply;

// =========================================================================
// Constants
// =========================================================================

const WHEEL_OUTER_RADIUS: f32 = 90.0;
const WHEEL_RING_WIDTH: f32 = 18.0;
const SV_SQUARE_SIZE: f32 = 100.0;
const SLIDER_HEIGHT: f32 = 18.0;
const SLIDER_LABEL_WIDTH: f32 = 28.0;
const SLIDER_VALUE_WIDTH: f32 = 42.0;
const SLIDER_SPACING: f32 = 4.0;
const SWATCH_SIZE: f32 = 20.0;
const SWATCH_SPACING: f32 = 3.0;
const SWATCH_COLUMNS: usize = 8;
const RECENT_COUNT: usize = 16;
const HEX_INPUT_WIDTH: f32 = 80.0;
const HEX_INPUT_HEIGHT: f32 = 24.0;
const CHECKER_SIZE: f32 = 6.0;
const COMPARISON_HEIGHT: f32 = 32.0;
const SECTION_SPACING: f32 = 8.0;
const WIDGET_PADDING: f32 = 12.0;

// =========================================================================
// ColorSpace
// =========================================================================

/// Colour space for display/editing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    SRGB,
    Linear,
}

impl Default for ColorSpace {
    fn default() -> Self {
        ColorSpace::SRGB
    }
}

/// Convert linear to sRGB (per component).
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert sRGB to linear (per component).
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn color_linear_to_srgb(c: Color) -> Color {
    Color::new(
        linear_to_srgb(c.r),
        linear_to_srgb(c.g),
        linear_to_srgb(c.b),
        c.a,
    )
}

fn color_srgb_to_linear(c: Color) -> Color {
    Color::new(
        srgb_to_linear(c.r),
        srgb_to_linear(c.g),
        srgb_to_linear(c.b),
        c.a,
    )
}

// =========================================================================
// InteractionTarget
// =========================================================================

/// Which part of the colour picker is being interacted with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InteractionTarget {
    None,
    HueRing,
    SVSquare,
    RedSlider,
    GreenSlider,
    BlueSlider,
    HueSlider,
    SatSlider,
    ValSlider,
    AlphaSlider,
    HexInput,
    EyeDropper,
}

// =========================================================================
// ColorSliderMode
// =========================================================================

/// Which set of sliders to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSliderMode {
    RGB,
    HSV,
    Both,
}

impl Default for ColorSliderMode {
    fn default() -> Self {
        ColorSliderMode::Both
    }
}

// =========================================================================
// ColorPickerLayout
// =========================================================================

/// Layout mode for the colour picker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPickerLayout {
    /// Compact: wheel + sliders stacked.
    Compact,
    /// Full: wheel on left, sliders on right.
    Full,
}

impl Default for ColorPickerLayout {
    fn default() -> Self {
        ColorPickerLayout::Full
    }
}

// =========================================================================
// FullColorPicker
// =========================================================================

/// A full-featured colour picker widget.
#[derive(Debug, Clone)]
pub struct FullColorPicker {
    /// Widget ID.
    pub id: UIId,

    // --- Colour state ---
    /// Current colour (always stored in linear space internally).
    pub color: Color,
    /// Original colour (for comparison).
    pub original_color: Color,
    /// HSV representation (hue 0-360, sat 0-1, val 0-1).
    pub hsv: (f32, f32, f32),
    /// Alpha (0-1).
    pub alpha: f32,

    // --- Mode ---
    /// Colour space for editing/display.
    pub color_space: ColorSpace,
    /// Slider mode.
    pub slider_mode: ColorSliderMode,
    /// Layout mode.
    pub layout: ColorPickerLayout,
    /// Whether alpha editing is enabled.
    pub show_alpha: bool,
    /// Whether the comparison swatch is shown.
    pub show_comparison: bool,
    /// Whether the hex input is shown.
    pub show_hex: bool,
    /// Whether the eye dropper button is shown.
    pub show_eye_dropper: bool,
    /// Whether recent colours are shown.
    pub show_recent: bool,
    /// Whether preset colours are shown.
    pub show_presets: bool,

    // --- Recent / presets ---
    /// Recently used colours (most recent first).
    pub recent_colors: Vec<Color>,
    /// Preset colours.
    pub preset_colors: Vec<Color>,

    // --- Interaction ---
    /// Current interaction target.
    interaction: InteractionTarget,
    /// Whether the eye dropper is active.
    pub eye_dropper_active: bool,
    /// Hex input text.
    pub hex_text: String,
    /// Hex input focused.
    pub hex_focused: bool,
    /// Hex input cursor position.
    pub hex_cursor: usize,

    // --- Visual ---
    /// Background colour.
    pub background: Color,
    /// Border colour.
    pub border_color: Color,
    /// Text colour.
    pub text_color: Color,
    /// Label colour.
    pub label_color: Color,
    /// Font size.
    pub font_size: f32,
    /// Font ID.
    pub font_id: u32,

    // --- State ---
    /// Whether the colour changed this frame.
    pub color_changed: bool,
    /// Whether the widget is enabled.
    pub enabled: bool,
    /// Whether the widget is visible.
    pub visible: bool,
    /// Computed total size.
    cached_size: Vec2,
}

impl FullColorPicker {
    /// Create a new colour picker with the given initial colour.
    pub fn new(color: Color) -> Self {
        let (h, s, v) = color.to_hsv();
        let mut picker = Self {
            id: UIId::INVALID,
            color,
            original_color: color,
            hsv: (h, s, v),
            alpha: color.a,
            color_space: ColorSpace::SRGB,
            slider_mode: ColorSliderMode::Both,
            layout: ColorPickerLayout::Full,
            show_alpha: true,
            show_comparison: true,
            show_hex: true,
            show_eye_dropper: true,
            show_recent: true,
            show_presets: true,
            recent_colors: Vec::new(),
            preset_colors: Self::default_presets(),
            interaction: InteractionTarget::None,
            eye_dropper_active: false,
            hex_text: String::new(),
            hex_focused: false,
            hex_cursor: 0,
            background: Color::from_hex("#2D2D30"),
            border_color: Color::from_hex("#3F3F46"),
            text_color: Color::from_hex("#CCCCCC"),
            label_color: Color::from_hex("#999999"),
            font_size: 12.0,
            font_id: 0,
            color_changed: false,
            enabled: true,
            visible: true,
            cached_size: Vec2::ZERO,
        };
        picker.sync_hex_from_color();
        picker
    }

    /// Default preset colours.
    fn default_presets() -> Vec<Color> {
        vec![
            Color::from_hex("#FF0000"), // Red
            Color::from_hex("#FF8800"), // Orange
            Color::from_hex("#FFFF00"), // Yellow
            Color::from_hex("#88FF00"), // Yellow-green
            Color::from_hex("#00FF00"), // Green
            Color::from_hex("#00FF88"), // Spring green
            Color::from_hex("#00FFFF"), // Cyan
            Color::from_hex("#0088FF"), // Sky blue
            Color::from_hex("#0000FF"), // Blue
            Color::from_hex("#8800FF"), // Purple
            Color::from_hex("#FF00FF"), // Magenta
            Color::from_hex("#FF0088"), // Pink
            Color::from_hex("#FFFFFF"), // White
            Color::from_hex("#AAAAAA"), // Light grey
            Color::from_hex("#555555"), // Dark grey
            Color::from_hex("#000000"), // Black
        ]
    }

    /// Set the colour.
    pub fn set_color(&mut self, color: Color) {
        self.color = color;
        self.alpha = color.a;
        let (h, s, v) = color.to_hsv();
        // Preserve hue if saturation or value is zero.
        if s > 0.001 && v > 0.001 {
            self.hsv = (h, s, v);
        } else {
            self.hsv.1 = s;
            self.hsv.2 = v;
        }
        self.sync_hex_from_color();
    }

    /// Get the current colour.
    pub fn get_color(&self) -> Color {
        self.color.with_alpha(self.alpha)
    }

    /// Get display colour (converted to display colour space).
    pub fn display_color(&self) -> Color {
        match self.color_space {
            ColorSpace::SRGB => color_linear_to_srgb(self.color),
            ColorSpace::Linear => self.color,
        }
    }

    /// Set the original colour for comparison.
    pub fn set_original(&mut self, color: Color) {
        self.original_color = color;
    }

    /// Push the current colour to the recent list.
    pub fn push_recent(&mut self) {
        let c = self.get_color();
        // Remove duplicates.
        self.recent_colors.retain(|rc| {
            (rc.r - c.r).abs() > 0.01
                || (rc.g - c.g).abs() > 0.01
                || (rc.b - c.b).abs() > 0.01
                || (rc.a - c.a).abs() > 0.01
        });
        self.recent_colors.insert(0, c);
        if self.recent_colors.len() > RECENT_COUNT {
            self.recent_colors.truncate(RECENT_COUNT);
        }
    }

    fn update_color_from_hsv(&mut self) {
        let c = Color::from_hsv(self.hsv.0, self.hsv.1, self.hsv.2);
        self.color = c.with_alpha(self.alpha);
        self.sync_hex_from_color();
        self.color_changed = true;
    }

    fn update_hsv_from_color(&mut self) {
        let (h, s, v) = self.color.to_hsv();
        if s > 0.001 && v > 0.001 {
            self.hsv = (h, s, v);
        } else {
            self.hsv.1 = s;
            self.hsv.2 = v;
        }
    }

    fn sync_hex_from_color(&mut self) {
        let c = self.get_color();
        let r = (c.r.clamp(0.0, 1.0) * 255.0) as u8;
        let g = (c.g.clamp(0.0, 1.0) * 255.0) as u8;
        let b = (c.b.clamp(0.0, 1.0) * 255.0) as u8;
        if self.show_alpha {
            let a = (c.a.clamp(0.0, 1.0) * 255.0) as u8;
            self.hex_text = format!("{:02X}{:02X}{:02X}{:02X}", r, g, b, a);
        } else {
            self.hex_text = format!("{:02X}{:02X}{:02X}", r, g, b);
        }
    }

    fn apply_hex(&mut self) {
        let c = Color::from_hex(&self.hex_text);
        self.color = c;
        self.alpha = c.a;
        self.update_hsv_from_color();
        self.color_changed = true;
    }

    /// RGB values as 0-255.
    pub fn rgb_u8(&self) -> (u8, u8, u8) {
        let c = self.color;
        (
            (c.r.clamp(0.0, 1.0) * 255.0) as u8,
            (c.g.clamp(0.0, 1.0) * 255.0) as u8,
            (c.b.clamp(0.0, 1.0) * 255.0) as u8,
        )
    }

    /// Set from RGB 0-255.
    pub fn set_rgb_u8(&mut self, r: u8, g: u8, b: u8) {
        self.color = Color::from_rgba8(r, g, b, (self.alpha * 255.0) as u8);
        self.update_hsv_from_color();
        self.sync_hex_from_color();
        self.color_changed = true;
    }

    /// Toggle colour space.
    pub fn toggle_color_space(&mut self) {
        self.color_space = match self.color_space {
            ColorSpace::SRGB => ColorSpace::Linear,
            ColorSpace::Linear => ColorSpace::SRGB,
        };
    }

    // =====================================================================
    // Layout helpers
    // =====================================================================

    /// Center of the hue wheel.
    fn wheel_center(&self, rect: Rect) -> Vec2 {
        Vec2::new(
            rect.min.x + WIDGET_PADDING + WHEEL_OUTER_RADIUS,
            rect.min.y + WIDGET_PADDING + WHEEL_OUTER_RADIUS,
        )
    }

    /// The SV square rect (inscribed in the hue ring).
    fn sv_square_rect(&self, rect: Rect) -> Rect {
        let center = self.wheel_center(rect);
        let inner_r = WHEEL_OUTER_RADIUS - WHEEL_RING_WIDTH;
        let half = inner_r * 0.707; // cos(45) for inscribed square
        Rect::new(
            Vec2::new(center.x - half, center.y - half),
            Vec2::new(center.x + half, center.y + half),
        )
    }

    /// Slider area rect (right of wheel in Full layout, below in Compact).
    fn slider_area_rect(&self, rect: Rect) -> Rect {
        match self.layout {
            ColorPickerLayout::Full => {
                let x = rect.min.x + WIDGET_PADDING + WHEEL_OUTER_RADIUS * 2.0 + SECTION_SPACING * 2.0;
                let y = rect.min.y + WIDGET_PADDING;
                let w = rect.width() - (x - rect.min.x) - WIDGET_PADDING;
                Rect::new(
                    Vec2::new(x, y),
                    Vec2::new(x + w.max(150.0), rect.max.y - WIDGET_PADDING),
                )
            }
            ColorPickerLayout::Compact => {
                let y = rect.min.y + WIDGET_PADDING + WHEEL_OUTER_RADIUS * 2.0 + SECTION_SPACING;
                Rect::new(
                    Vec2::new(rect.min.x + WIDGET_PADDING, y),
                    Vec2::new(rect.max.x - WIDGET_PADDING, rect.max.y - WIDGET_PADDING),
                )
            }
        }
    }

    /// Individual slider rect.
    fn slider_rect(&self, base: Rect, index: usize) -> Rect {
        let y = base.min.y + index as f32 * (SLIDER_HEIGHT + SLIDER_SPACING);
        Rect::new(
            Vec2::new(base.min.x + SLIDER_LABEL_WIDTH, y),
            Vec2::new(base.max.x - SLIDER_VALUE_WIDTH, y + SLIDER_HEIGHT),
        )
    }

    /// Compute desired size.
    pub fn compute_desired_size(&self) -> Vec2 {
        let wheel_w = WHEEL_OUTER_RADIUS * 2.0;
        let wheel_h = WHEEL_OUTER_RADIUS * 2.0;

        let mut slider_count: usize = 0;
        match self.slider_mode {
            ColorSliderMode::RGB => slider_count += 3,
            ColorSliderMode::HSV => slider_count += 3,
            ColorSliderMode::Both => slider_count += 6,
        }
        if self.show_alpha {
            slider_count += 1;
        }
        let slider_area_h = slider_count as f32 * (SLIDER_HEIGHT + SLIDER_SPACING);

        let bottom_section = {
            let mut h: f32 = 0.0;
            if self.show_comparison {
                h += COMPARISON_HEIGHT + SECTION_SPACING;
            }
            if self.show_hex || self.show_eye_dropper {
                h += HEX_INPUT_HEIGHT + SECTION_SPACING;
            }
            if self.show_presets {
                let rows = (self.preset_colors.len() + SWATCH_COLUMNS - 1) / SWATCH_COLUMNS;
                h += rows as f32 * (SWATCH_SIZE + SWATCH_SPACING) + SECTION_SPACING;
            }
            if self.show_recent && !self.recent_colors.is_empty() {
                let rows = (self.recent_colors.len() + SWATCH_COLUMNS - 1) / SWATCH_COLUMNS;
                h += rows as f32 * (SWATCH_SIZE + SWATCH_SPACING) + SECTION_SPACING + 16.0;
            }
            h
        };

        match self.layout {
            ColorPickerLayout::Full => {
                let w = wheel_w + 200.0 + WIDGET_PADDING * 2.0 + SECTION_SPACING;
                let h = wheel_h.max(slider_area_h) + bottom_section + WIDGET_PADDING * 2.0;
                Vec2::new(w, h)
            }
            ColorPickerLayout::Compact => {
                let w = wheel_w + WIDGET_PADDING * 2.0;
                let h = wheel_h + slider_area_h + bottom_section + WIDGET_PADDING * 2.0;
                Vec2::new(w.max(250.0), h)
            }
        }
    }

    // =====================================================================
    // Paint
    // =====================================================================

    /// Paint the colour picker.
    pub fn paint(&self, rect: Rect, draw: &mut DrawList) {
        if !self.visible {
            return;
        }

        // Background.
        draw.commands.push(DrawCommand::Rect {
            rect,
            color: self.background,
            corner_radii: CornerRadii::all(4.0),
            border: BorderSpec::new(self.border_color, 1.0),
            shadow: None,
        });

        self.paint_hue_wheel(rect, draw);
        self.paint_sv_square(rect, draw);
        self.paint_sliders(rect, draw);
        self.paint_bottom_section(rect, draw);
    }

    fn paint_hue_wheel(&self, rect: Rect, draw: &mut DrawList) {
        let center = self.wheel_center(rect);
        let outer_r = WHEEL_OUTER_RADIUS;
        let inner_r = outer_r - WHEEL_RING_WIDTH;
        let segments = 64;

        // Draw hue ring as coloured segments.
        for i in 0..segments {
            let angle0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let angle1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            let mid_angle = (angle0 + angle1) * 0.5;
            let hue = mid_angle.to_degrees();
            let color = Color::from_hsv(if hue < 0.0 { hue + 360.0 } else { hue }, 1.0, 1.0);

            let cos0 = angle0.cos();
            let sin0 = angle0.sin();
            let cos1 = angle1.cos();
            let sin1 = angle1.sin();

            // Outer arc segment as two triangles.
            let p_outer0 = center + Vec2::new(cos0 * outer_r, sin0 * outer_r);
            let p_outer1 = center + Vec2::new(cos1 * outer_r, sin1 * outer_r);
            let p_inner0 = center + Vec2::new(cos0 * inner_r, sin0 * inner_r);
            let p_inner1 = center + Vec2::new(cos1 * inner_r, sin1 * inner_r);

            draw.commands.push(DrawCommand::Triangle {
                p0: p_outer0,
                p1: p_outer1,
                p2: p_inner0,
                color,
            });
            draw.commands.push(DrawCommand::Triangle {
                p0: p_inner0,
                p1: p_outer1,
                p2: p_inner1,
                color,
            });
        }

        // Hue indicator on the ring.
        let hue_angle = self.hsv.0.to_radians();
        let mid_r = (outer_r + inner_r) * 0.5;
        let indicator_pos = center + Vec2::new(hue_angle.cos() * mid_r, hue_angle.sin() * mid_r);
        draw.commands.push(DrawCommand::Circle {
            center: indicator_pos,
            radius: WHEEL_RING_WIDTH * 0.4,
            color: Color::WHITE,
            border: BorderSpec::new(Color::BLACK, 2.0),
        });
    }

    fn paint_sv_square(&self, rect: Rect, draw: &mut DrawList) {
        let sv_rect = self.sv_square_rect(rect);
        let steps = 16;

        // Draw as a grid of coloured quads.
        let step_w = sv_rect.width() / steps as f32;
        let step_h = sv_rect.height() / steps as f32;

        for yi in 0..steps {
            for xi in 0..steps {
                let s = (xi as f32 + 0.5) / steps as f32;
                let v = 1.0 - (yi as f32 + 0.5) / steps as f32;
                let color = Color::from_hsv(self.hsv.0, s, v);

                let cell_rect = Rect::new(
                    Vec2::new(
                        sv_rect.min.x + xi as f32 * step_w,
                        sv_rect.min.y + yi as f32 * step_h,
                    ),
                    Vec2::new(
                        sv_rect.min.x + (xi + 1) as f32 * step_w,
                        sv_rect.min.y + (yi + 1) as f32 * step_h,
                    ),
                );
                draw.commands.push(DrawCommand::Rect {
                    rect: cell_rect,
                    color,
                    corner_radii: CornerRadii::ZERO,
                    border: BorderSpec::default(),
                    shadow: None,
                });
            }
        }

        // SV indicator.
        let sx = sv_rect.min.x + self.hsv.1 * sv_rect.width();
        let sy = sv_rect.min.y + (1.0 - self.hsv.2) * sv_rect.height();
        draw.commands.push(DrawCommand::Circle {
            center: Vec2::new(sx, sy),
            radius: 5.0,
            color: Color::WHITE,
            border: BorderSpec::new(Color::BLACK, 2.0),
        });

        // Border.
        draw.commands.push(DrawCommand::Rect {
            rect: sv_rect,
            color: Color::TRANSPARENT,
            corner_radii: CornerRadii::ZERO,
            border: BorderSpec::new(Color::new(0.3, 0.3, 0.3, 1.0), 1.0),
            shadow: None,
        });
    }

    fn paint_sliders(&self, rect: Rect, draw: &mut DrawList) {
        let area = self.slider_area_rect(rect);
        let mut idx: usize = 0;

        if self.slider_mode == ColorSliderMode::RGB || self.slider_mode == ColorSliderMode::Both {
            let (r, g, b) = self.rgb_u8();
            self.paint_slider(area, idx, "R", r as f32, 255.0, Color::RED, draw);
            idx += 1;
            self.paint_slider(area, idx, "G", g as f32, 255.0, Color::GREEN, draw);
            idx += 1;
            self.paint_slider(area, idx, "B", b as f32, 255.0, Color::BLUE, draw);
            idx += 1;
        }

        if self.slider_mode == ColorSliderMode::HSV || self.slider_mode == ColorSliderMode::Both {
            let (h, s, v) = self.hsv;
            self.paint_slider(area, idx, "H", h, 360.0, Color::from_hex("#FF6600"), draw);
            idx += 1;
            self.paint_slider(area, idx, "S", s * 100.0, 100.0, Color::from_hex("#44CC44"), draw);
            idx += 1;
            self.paint_slider(area, idx, "V", v * 100.0, 100.0, Color::from_hex("#4488FF"), draw);
            idx += 1;
        }

        if self.show_alpha {
            // Alpha slider with checkerboard.
            let sr = self.slider_rect(area, idx);

            // Label.
            draw.commands.push(DrawCommand::Text {
                text: "A".to_string(),
                position: Vec2::new(area.min.x, sr.min.y + 2.0),
                font_size: self.font_size,
                color: self.label_color,
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            // Checkerboard background.
            self.paint_checkerboard(sr, draw);

            // Alpha gradient overlay.
            let c = self.color;
            draw.commands.push(DrawCommand::Rect {
                rect: Rect::new(
                    sr.min,
                    Vec2::new(sr.min.x + sr.width() * self.alpha, sr.max.y),
                ),
                color: c.with_alpha(1.0),
                corner_radii: CornerRadii::ZERO,
                border: BorderSpec::default(),
                shadow: None,
            });

            // Border.
            draw.commands.push(DrawCommand::Rect {
                rect: sr,
                color: Color::TRANSPARENT,
                corner_radii: CornerRadii::all(2.0),
                border: BorderSpec::new(self.border_color, 1.0),
                shadow: None,
            });

            // Thumb.
            let thumb_x = sr.min.x + self.alpha * sr.width();
            draw.commands.push(DrawCommand::Rect {
                rect: Rect::new(
                    Vec2::new(thumb_x - 2.0, sr.min.y - 1.0),
                    Vec2::new(thumb_x + 2.0, sr.max.y + 1.0),
                ),
                color: Color::WHITE,
                corner_radii: CornerRadii::ZERO,
                border: BorderSpec::new(Color::BLACK, 1.0),
                shadow: None,
            });

            // Value text.
            draw.commands.push(DrawCommand::Text {
                text: format!("{}", (self.alpha * 255.0) as u8),
                position: Vec2::new(sr.max.x + 6.0, sr.min.y + 2.0),
                font_size: self.font_size,
                color: self.text_color,
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }
    }

    fn paint_slider(
        &self,
        area: Rect,
        index: usize,
        label: &str,
        value: f32,
        max: f32,
        accent: Color,
        draw: &mut DrawList,
    ) {
        let sr = self.slider_rect(area, index);

        // Label.
        draw.commands.push(DrawCommand::Text {
            text: label.to_string(),
            position: Vec2::new(area.min.x, sr.min.y + 2.0),
            font_size: self.font_size,
            color: self.label_color,
            font_id: self.font_id,
            max_width: None,
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });

        // Track background.
        draw.commands.push(DrawCommand::Rect {
            rect: sr,
            color: Color::from_hex("#1E1E1E"),
            corner_radii: CornerRadii::all(2.0),
            border: BorderSpec::new(self.border_color, 1.0),
            shadow: None,
        });

        // Filled portion.
        let fill_w = (value / max).clamp(0.0, 1.0) * sr.width();
        draw.commands.push(DrawCommand::Rect {
            rect: Rect::new(sr.min, Vec2::new(sr.min.x + fill_w, sr.max.y)),
            color: accent.with_alpha(0.6),
            corner_radii: CornerRadii::new(2.0, 0.0, 0.0, 2.0),
            border: BorderSpec::default(),
            shadow: None,
        });

        // Thumb.
        let thumb_x = sr.min.x + fill_w;
        draw.commands.push(DrawCommand::Rect {
            rect: Rect::new(
                Vec2::new(thumb_x - 2.0, sr.min.y - 1.0),
                Vec2::new(thumb_x + 2.0, sr.max.y + 1.0),
            ),
            color: Color::WHITE,
            corner_radii: CornerRadii::ZERO,
            border: BorderSpec::new(Color::BLACK, 1.0),
            shadow: None,
        });

        // Value text.
        draw.commands.push(DrawCommand::Text {
            text: format!("{}", value as i32),
            position: Vec2::new(sr.max.x + 6.0, sr.min.y + 2.0),
            font_size: self.font_size,
            color: self.text_color,
            font_id: self.font_id,
            max_width: None,
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });
    }

    fn paint_checkerboard(&self, rect: Rect, draw: &mut DrawList) {
        let cols = (rect.width() / CHECKER_SIZE).ceil() as i32;
        let rows = (rect.height() / CHECKER_SIZE).ceil() as i32;

        draw.commands.push(DrawCommand::PushClip { rect });

        for cy in 0..rows {
            for cx in 0..cols {
                let dark = (cx + cy) % 2 == 0;
                let color = if dark {
                    Color::from_hex("#666666")
                } else {
                    Color::from_hex("#999999")
                };
                draw.commands.push(DrawCommand::Rect {
                    rect: Rect::new(
                        Vec2::new(
                            rect.min.x + cx as f32 * CHECKER_SIZE,
                            rect.min.y + cy as f32 * CHECKER_SIZE,
                        ),
                        Vec2::new(
                            rect.min.x + (cx + 1) as f32 * CHECKER_SIZE,
                            rect.min.y + (cy + 1) as f32 * CHECKER_SIZE,
                        ),
                    ),
                    color,
                    corner_radii: CornerRadii::ZERO,
                    border: BorderSpec::default(),
                    shadow: None,
                });
            }
        }

        draw.commands.push(DrawCommand::PopClip);
    }

    fn paint_bottom_section(&self, rect: Rect, draw: &mut DrawList) {
        let mut y = match self.layout {
            ColorPickerLayout::Full => {
                rect.min.y + WIDGET_PADDING + WHEEL_OUTER_RADIUS * 2.0 + SECTION_SPACING
            }
            ColorPickerLayout::Compact => {
                // Below sliders.
                let area = self.slider_area_rect(rect);
                let mut slider_count: usize = match self.slider_mode {
                    ColorSliderMode::RGB => 3,
                    ColorSliderMode::HSV => 3,
                    ColorSliderMode::Both => 6,
                };
                if self.show_alpha {
                    slider_count += 1;
                }
                area.min.y + slider_count as f32 * (SLIDER_HEIGHT + SLIDER_SPACING) + SECTION_SPACING
            }
        };

        let left = rect.min.x + WIDGET_PADDING;
        let right = rect.max.x - WIDGET_PADDING;

        // Comparison swatch (old vs new).
        if self.show_comparison {
            let half = (right - left) * 0.5;

            // Old colour.
            draw.commands.push(DrawCommand::Rect {
                rect: Rect::new(
                    Vec2::new(left, y),
                    Vec2::new(left + half, y + COMPARISON_HEIGHT),
                ),
                color: self.original_color,
                corner_radii: CornerRadii::new(4.0, 0.0, 0.0, 4.0),
                border: BorderSpec::new(self.border_color, 1.0),
                shadow: None,
            });

            // New colour.
            draw.commands.push(DrawCommand::Rect {
                rect: Rect::new(
                    Vec2::new(left + half, y),
                    Vec2::new(right, y + COMPARISON_HEIGHT),
                ),
                color: self.get_color(),
                corner_radii: CornerRadii::new(0.0, 4.0, 4.0, 0.0),
                border: BorderSpec::new(self.border_color, 1.0),
                shadow: None,
            });

            // Labels.
            draw.commands.push(DrawCommand::Text {
                text: "Old".to_string(),
                position: Vec2::new(left + 4.0, y + 9.0),
                font_size: self.font_size - 1.0,
                color: Color::WHITE.with_alpha(0.7),
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
            draw.commands.push(DrawCommand::Text {
                text: "New".to_string(),
                position: Vec2::new(left + half + 4.0, y + 9.0),
                font_size: self.font_size - 1.0,
                color: Color::WHITE.with_alpha(0.7),
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            y += COMPARISON_HEIGHT + SECTION_SPACING;
        }

        // Hex input + eye dropper + sRGB toggle.
        if self.show_hex || self.show_eye_dropper {
            // Hex input.
            if self.show_hex {
                let hex_rect = Rect::new(
                    Vec2::new(left, y),
                    Vec2::new(left + HEX_INPUT_WIDTH, y + HEX_INPUT_HEIGHT),
                );
                draw.commands.push(DrawCommand::Rect {
                    rect: hex_rect,
                    color: Color::from_hex("#1E1E1E"),
                    corner_radii: CornerRadii::all(3.0),
                    border: BorderSpec::new(
                        if self.hex_focused {
                            Color::from_hex("#007ACC")
                        } else {
                            self.border_color
                        },
                        1.0,
                    ),
                    shadow: None,
                });

                draw.commands.push(DrawCommand::Text {
                    text: "#".to_string(),
                    position: Vec2::new(hex_rect.min.x + 4.0, hex_rect.min.y + 5.0),
                    font_size: self.font_size,
                    color: self.label_color,
                    font_id: self.font_id,
                    max_width: None,
                    align: TextAlign::Left,
                    vertical_align: TextVerticalAlign::Top,
                });

                draw.commands.push(DrawCommand::Text {
                    text: self.hex_text.clone(),
                    position: Vec2::new(hex_rect.min.x + 16.0, hex_rect.min.y + 5.0),
                    font_size: self.font_size,
                    color: self.text_color,
                    font_id: self.font_id,
                    max_width: Some(HEX_INPUT_WIDTH - 24.0),
                    align: TextAlign::Left,
                    vertical_align: TextVerticalAlign::Top,
                });
            }

            // sRGB / Linear toggle.
            let toggle_x = left + HEX_INPUT_WIDTH + 8.0;
            let label = match self.color_space {
                ColorSpace::SRGB => "sRGB",
                ColorSpace::Linear => "Linear",
            };
            draw.commands.push(DrawCommand::Rect {
                rect: Rect::new(
                    Vec2::new(toggle_x, y),
                    Vec2::new(toggle_x + 50.0, y + HEX_INPUT_HEIGHT),
                ),
                color: Color::from_hex("#3C3C3C"),
                corner_radii: CornerRadii::all(3.0),
                border: BorderSpec::new(self.border_color, 1.0),
                shadow: None,
            });
            draw.commands.push(DrawCommand::Text {
                text: label.to_string(),
                position: Vec2::new(toggle_x + 6.0, y + 5.0),
                font_size: self.font_size - 1.0,
                color: self.text_color,
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            // Eye dropper button.
            if self.show_eye_dropper {
                let ed_x = toggle_x + 58.0;
                let ed_rect = Rect::new(
                    Vec2::new(ed_x, y),
                    Vec2::new(ed_x + HEX_INPUT_HEIGHT, y + HEX_INPUT_HEIGHT),
                );
                let ed_color = if self.eye_dropper_active {
                    Color::from_hex("#007ACC")
                } else {
                    Color::from_hex("#3C3C3C")
                };
                draw.commands.push(DrawCommand::Rect {
                    rect: ed_rect,
                    color: ed_color,
                    corner_radii: CornerRadii::all(3.0),
                    border: BorderSpec::new(self.border_color, 1.0),
                    shadow: None,
                });
                // Dropper icon (simplified).
                let cx = (ed_rect.min.x + ed_rect.max.x) * 0.5;
                let cy = (ed_rect.min.y + ed_rect.max.y) * 0.5;
                draw.commands.push(DrawCommand::Circle {
                    center: Vec2::new(cx, cy - 2.0),
                    radius: 4.0,
                    color: Color::TRANSPARENT,
                    border: BorderSpec::new(Color::WHITE, 1.5),
                });
                draw.commands.push(DrawCommand::Line {
                    start: Vec2::new(cx, cy + 2.0),
                    end: Vec2::new(cx, cy + 7.0),
                    color: Color::WHITE,
                    thickness: 1.5,
                });
            }

            y += HEX_INPUT_HEIGHT + SECTION_SPACING;
        }

        // Preset colours.
        if self.show_presets {
            draw.commands.push(DrawCommand::Text {
                text: "Presets".to_string(),
                position: Vec2::new(left, y),
                font_size: self.font_size - 1.0,
                color: self.label_color,
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
            y += 14.0;

            self.paint_swatches(&self.preset_colors, left, y, right, draw);
            let rows = (self.preset_colors.len() + SWATCH_COLUMNS - 1) / SWATCH_COLUMNS;
            y += rows as f32 * (SWATCH_SIZE + SWATCH_SPACING) + SECTION_SPACING;
        }

        // Recent colours.
        if self.show_recent && !self.recent_colors.is_empty() {
            draw.commands.push(DrawCommand::Text {
                text: "Recent".to_string(),
                position: Vec2::new(left, y),
                font_size: self.font_size - 1.0,
                color: self.label_color,
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
            y += 14.0;

            self.paint_swatches(&self.recent_colors, left, y, right, draw);
        }
    }

    fn paint_swatches(
        &self,
        colors: &[Color],
        left: f32,
        y: f32,
        _right: f32,
        draw: &mut DrawList,
    ) {
        for (i, color) in colors.iter().enumerate() {
            let col = i % SWATCH_COLUMNS;
            let row = i / SWATCH_COLUMNS;
            let sx = left + col as f32 * (SWATCH_SIZE + SWATCH_SPACING);
            let sy = y + row as f32 * (SWATCH_SIZE + SWATCH_SPACING);
            let swatch_rect = Rect::new(
                Vec2::new(sx, sy),
                Vec2::new(sx + SWATCH_SIZE, sy + SWATCH_SIZE),
            );
            draw.commands.push(DrawCommand::Rect {
                rect: swatch_rect,
                color: *color,
                corner_radii: CornerRadii::all(2.0),
                border: BorderSpec::new(self.border_color, 1.0),
                shadow: None,
            });
        }
    }

    // =====================================================================
    // Event handling
    // =====================================================================

    /// Handle events.
    pub fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled || !self.visible {
            return EventReply::Unhandled;
        }

        self.color_changed = false;

        match event {
            UIEvent::Click { position, button, .. } => {
                if *button != MouseButton::Left {
                    return EventReply::Unhandled;
                }
                self.handle_mouse_down(*position, rect)
            }

            UIEvent::Hover { position } | UIEvent::DragMove { position, .. } => {
                self.handle_mouse_drag(*position, rect)
            }

            UIEvent::MouseUp { button, .. } => {
                if *button == MouseButton::Left && self.interaction != InteractionTarget::None {
                    if self.interaction != InteractionTarget::HexInput {
                        self.interaction = InteractionTarget::None;
                    }
                    return EventReply::ReleaseMouse;
                }
                EventReply::Unhandled
            }

            UIEvent::KeyInput { key, pressed, .. } => {
                if *pressed && self.hex_focused {
                    self.handle_hex_key(*key);
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }

            UIEvent::TextInput { character } => {
                if self.hex_focused {
                    if character.is_ascii_hexdigit() && self.hex_text.len() < 8 {
                        self.hex_text.push(character.to_ascii_uppercase());
                        self.hex_cursor += 1;
                    }
                    if self.hex_text.len() == 6 || self.hex_text.len() == 8 {
                        self.apply_hex();
                    }
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }

            _ => EventReply::Unhandled,
        }
    }

    fn handle_mouse_down(&mut self, pos: Vec2, rect: Rect) -> EventReply {
        let center = self.wheel_center(rect);
        let dist = (pos - center).length();
        let inner_r = WHEEL_OUTER_RADIUS - WHEEL_RING_WIDTH;

        // Hue ring.
        if dist >= inner_r && dist <= WHEEL_OUTER_RADIUS {
            self.interaction = InteractionTarget::HueRing;
            let angle = (pos.y - center.y).atan2(pos.x - center.x).to_degrees();
            self.hsv.0 = if angle < 0.0 { angle + 360.0 } else { angle };
            self.update_color_from_hsv();
            return EventReply::CaptureMouse;
        }

        // SV square.
        let sv = self.sv_square_rect(rect);
        if sv.contains(pos) {
            self.interaction = InteractionTarget::SVSquare;
            self.hsv.1 = ((pos.x - sv.min.x) / sv.width()).clamp(0.0, 1.0);
            self.hsv.2 = 1.0 - ((pos.y - sv.min.y) / sv.height()).clamp(0.0, 1.0);
            self.update_color_from_hsv();
            return EventReply::CaptureMouse;
        }

        // Sliders.
        let area = self.slider_area_rect(rect);
        let mut idx: usize = 0;
        if self.slider_mode == ColorSliderMode::RGB || self.slider_mode == ColorSliderMode::Both {
            for ch in 0..3 {
                let sr = self.slider_rect(area, idx);
                if sr.contains(pos) {
                    let t = ((pos.x - sr.min.x) / sr.width()).clamp(0.0, 1.0);
                    let val = (t * 255.0) as u8;
                    let (mut r, mut g, mut b) = self.rgb_u8();
                    match ch {
                        0 => r = val,
                        1 => g = val,
                        2 => b = val,
                        _ => {}
                    }
                    self.set_rgb_u8(r, g, b);
                    self.interaction = match ch {
                        0 => InteractionTarget::RedSlider,
                        1 => InteractionTarget::GreenSlider,
                        _ => InteractionTarget::BlueSlider,
                    };
                    return EventReply::CaptureMouse;
                }
                idx += 1;
            }
        }
        if self.slider_mode == ColorSliderMode::HSV || self.slider_mode == ColorSliderMode::Both {
            for ch in 0..3 {
                let sr = self.slider_rect(area, idx);
                if sr.contains(pos) {
                    let t = ((pos.x - sr.min.x) / sr.width()).clamp(0.0, 1.0);
                    match ch {
                        0 => self.hsv.0 = t * 360.0,
                        1 => self.hsv.1 = t,
                        2 => self.hsv.2 = t,
                        _ => {}
                    }
                    self.update_color_from_hsv();
                    self.interaction = match ch {
                        0 => InteractionTarget::HueSlider,
                        1 => InteractionTarget::SatSlider,
                        _ => InteractionTarget::ValSlider,
                    };
                    return EventReply::CaptureMouse;
                }
                idx += 1;
            }
        }
        if self.show_alpha {
            let sr = self.slider_rect(area, idx);
            if sr.contains(pos) {
                self.alpha = ((pos.x - sr.min.x) / sr.width()).clamp(0.0, 1.0);
                self.color = self.color.with_alpha(self.alpha);
                self.sync_hex_from_color();
                self.color_changed = true;
                self.interaction = InteractionTarget::AlphaSlider;
                return EventReply::CaptureMouse;
            }
        }

        // Preset/recent swatch click.
        if self.check_swatch_click(pos, rect) {
            return EventReply::Handled;
        }

        EventReply::Unhandled
    }

    fn handle_mouse_drag(&mut self, pos: Vec2, rect: Rect) -> EventReply {
        match self.interaction {
            InteractionTarget::HueRing => {
                let center = self.wheel_center(rect);
                let angle = (pos.y - center.y).atan2(pos.x - center.x).to_degrees();
                self.hsv.0 = if angle < 0.0 { angle + 360.0 } else { angle };
                self.update_color_from_hsv();
                EventReply::Handled
            }
            InteractionTarget::SVSquare => {
                let sv = self.sv_square_rect(rect);
                self.hsv.1 = ((pos.x - sv.min.x) / sv.width()).clamp(0.0, 1.0);
                self.hsv.2 = 1.0 - ((pos.y - sv.min.y) / sv.height()).clamp(0.0, 1.0);
                self.update_color_from_hsv();
                EventReply::Handled
            }
            InteractionTarget::RedSlider
            | InteractionTarget::GreenSlider
            | InteractionTarget::BlueSlider => {
                let area = self.slider_area_rect(rect);
                let ch = match self.interaction {
                    InteractionTarget::RedSlider => 0,
                    InteractionTarget::GreenSlider => 1,
                    _ => 2,
                };
                let sr = self.slider_rect(area, ch);
                let t = ((pos.x - sr.min.x) / sr.width()).clamp(0.0, 1.0);
                let val = (t * 255.0) as u8;
                let (mut r, mut g, mut b) = self.rgb_u8();
                match ch {
                    0 => r = val,
                    1 => g = val,
                    2 => b = val,
                    _ => {}
                }
                self.set_rgb_u8(r, g, b);
                EventReply::Handled
            }
            InteractionTarget::HueSlider
            | InteractionTarget::SatSlider
            | InteractionTarget::ValSlider => {
                let area = self.slider_area_rect(rect);
                let rgb_offset = if self.slider_mode == ColorSliderMode::Both { 3 } else { 0 };
                let ch = match self.interaction {
                    InteractionTarget::HueSlider => 0,
                    InteractionTarget::SatSlider => 1,
                    _ => 2,
                };
                let sr = self.slider_rect(area, rgb_offset + ch);
                let t = ((pos.x - sr.min.x) / sr.width()).clamp(0.0, 1.0);
                match ch {
                    0 => self.hsv.0 = t * 360.0,
                    1 => self.hsv.1 = t,
                    2 => self.hsv.2 = t,
                    _ => {}
                }
                self.update_color_from_hsv();
                EventReply::Handled
            }
            InteractionTarget::AlphaSlider => {
                let area = self.slider_area_rect(rect);
                let mut si = match self.slider_mode {
                    ColorSliderMode::RGB => 3,
                    ColorSliderMode::HSV => 3,
                    ColorSliderMode::Both => 6,
                };
                let sr = self.slider_rect(area, si);
                self.alpha = ((pos.x - sr.min.x) / sr.width()).clamp(0.0, 1.0);
                self.color = self.color.with_alpha(self.alpha);
                self.sync_hex_from_color();
                self.color_changed = true;
                EventReply::Handled
            }
            _ => EventReply::Unhandled,
        }
    }

    fn handle_hex_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Backspace => {
                if !self.hex_text.is_empty() {
                    self.hex_text.pop();
                    self.hex_cursor = self.hex_cursor.saturating_sub(1);
                }
            }
            KeyCode::Enter => {
                if self.hex_text.len() >= 6 {
                    self.apply_hex();
                }
                self.hex_focused = false;
                self.interaction = InteractionTarget::None;
            }
            KeyCode::Escape => {
                self.hex_focused = false;
                self.interaction = InteractionTarget::None;
                self.sync_hex_from_color();
            }
            _ => {}
        }
    }

    fn check_swatch_click(&mut self, pos: Vec2, rect: Rect) -> bool {
        let left = rect.min.x + WIDGET_PADDING;
        let mut y = match self.layout {
            ColorPickerLayout::Full => {
                rect.min.y + WIDGET_PADDING + WHEEL_OUTER_RADIUS * 2.0 + SECTION_SPACING
            }
            ColorPickerLayout::Compact => {
                let area = self.slider_area_rect(rect);
                let slider_count: usize = match self.slider_mode {
                    ColorSliderMode::RGB | ColorSliderMode::HSV => 3,
                    ColorSliderMode::Both => 6,
                } + if self.show_alpha { 1 } else { 0 };
                area.min.y + slider_count as f32 * (SLIDER_HEIGHT + SLIDER_SPACING) + SECTION_SPACING
            }
        };

        if self.show_comparison {
            y += COMPARISON_HEIGHT + SECTION_SPACING;
        }
        if self.show_hex || self.show_eye_dropper {
            y += HEX_INPUT_HEIGHT + SECTION_SPACING;
        }

        // Presets.
        if self.show_presets {
            y += 14.0; // "Presets" label
            if let Some(color) = self.swatch_at(pos, &self.preset_colors, left, y) {
                self.set_color(color);
                return true;
            }
            let rows = (self.preset_colors.len() + SWATCH_COLUMNS - 1) / SWATCH_COLUMNS;
            y += rows as f32 * (SWATCH_SIZE + SWATCH_SPACING) + SECTION_SPACING;
        }

        // Recent.
        if self.show_recent && !self.recent_colors.is_empty() {
            y += 14.0; // "Recent" label
            let recent_copy = self.recent_colors.clone();
            if let Some(color) = self.swatch_at(pos, &recent_copy, left, y) {
                self.set_color(color);
                return true;
            }
        }

        false
    }

    fn swatch_at(&self, pos: Vec2, colors: &[Color], left: f32, top: f32) -> Option<Color> {
        for (i, color) in colors.iter().enumerate() {
            let col = i % SWATCH_COLUMNS;
            let row = i / SWATCH_COLUMNS;
            let sx = left + col as f32 * (SWATCH_SIZE + SWATCH_SPACING);
            let sy = top + row as f32 * (SWATCH_SIZE + SWATCH_SPACING);
            let swatch_rect = Rect::new(
                Vec2::new(sx, sy),
                Vec2::new(sx + SWATCH_SIZE, sy + SWATCH_SIZE),
            );
            if swatch_rect.contains(pos) {
                return Some(*color);
            }
        }
        None
    }

    /// Take the color_changed flag.
    pub fn take_color_changed(&mut self) -> bool {
        let c = self.color_changed;
        self.color_changed = false;
        c
    }
}

impl Default for FullColorPicker {
    fn default() -> Self {
        Self::new(Color::WHITE)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_picker_create() {
        let picker = FullColorPicker::new(Color::RED);
        let (h, s, v) = picker.hsv;
        assert!((h - 0.0).abs() < 1.0 || (h - 360.0).abs() < 1.0);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_set_color() {
        let mut picker = FullColorPicker::new(Color::RED);
        picker.set_color(Color::BLUE);
        assert!((picker.color.b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rgb_u8() {
        let picker = FullColorPicker::new(Color::from_rgba8(128, 64, 255, 255));
        let (r, g, b) = picker.rgb_u8();
        assert_eq!(r, 128);
        assert_eq!(g, 64);
        assert_eq!(b, 255);
    }

    #[test]
    fn test_hex_sync() {
        let picker = FullColorPicker::new(Color::from_hex("#FF8800"));
        assert!(picker.hex_text.starts_with("FF88"));
    }

    #[test]
    fn test_recent_colors() {
        let mut picker = FullColorPicker::new(Color::RED);
        picker.push_recent();
        assert_eq!(picker.recent_colors.len(), 1);
        picker.set_color(Color::BLUE);
        picker.push_recent();
        assert_eq!(picker.recent_colors.len(), 2);
    }

    #[test]
    fn test_preset_colors() {
        let picker = FullColorPicker::new(Color::RED);
        assert_eq!(picker.preset_colors.len(), 16);
    }

    #[test]
    fn test_srgb_conversion() {
        let linear = Color::new(0.5, 0.5, 0.5, 1.0);
        let srgb = color_linear_to_srgb(linear);
        assert!(srgb.r > linear.r); // sRGB should be brighter for mid-tones.
        let back = color_srgb_to_linear(srgb);
        assert!((back.r - linear.r).abs() < 0.001);
    }
}
