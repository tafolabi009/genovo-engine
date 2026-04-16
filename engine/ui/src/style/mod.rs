//! Styling and theming system.
//!
//! The style system provides per-widget visual properties (colours, borders,
//! fonts), pseudo-state variants (hovered, pressed, disabled, etc.), cascading
//! style sheets, theme presets, and animated transitions between style states.

use std::collections::HashMap;

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::render_commands::{Color, CornerRadii, Shadow};

// ---------------------------------------------------------------------------
// PseudoState
// ---------------------------------------------------------------------------

/// Interaction / visual state of a widget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PseudoState {
    Normal,
    Hovered,
    Pressed,
    Focused,
    Disabled,
    Selected,
    /// Active is used for toggled-on controls (e.g., a pressed tab).
    Active,
    /// For text inputs when text is being edited.
    Editing,
}

impl Default for PseudoState {
    fn default() -> Self {
        Self::Normal
    }
}

// ---------------------------------------------------------------------------
// CursorIcon
// ---------------------------------------------------------------------------

/// Mouse cursor icons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CursorIcon {
    Default,
    Pointer,
    Text,
    Crosshair,
    Move,
    NotAllowed,
    ResizeNS,
    ResizeEW,
    ResizeNWSE,
    ResizeNESW,
    Grab,
    Grabbing,
    Wait,
    Progress,
    Help,
}

impl Default for CursorIcon {
    fn default() -> Self {
        Self::Default
    }
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

/// Complete visual style for a UI node. Every field is `Option` so that
/// styles can be partially specified and merged (child overrides parent).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Style {
    // Background
    pub background_color: Option<Color>,
    pub background_gradient: Option<crate::render_commands::Gradient>,

    // Border
    pub border_color: Option<Color>,
    pub border_width: Option<f32>,
    pub border_radius: Option<CornerRadii>,

    // Padding/Margin are also in LayoutParams; style overrides are applied
    // last so that themes can set default spacing.
    pub padding_left: Option<f32>,
    pub padding_top: Option<f32>,
    pub padding_right: Option<f32>,
    pub padding_bottom: Option<f32>,
    pub margin_left: Option<f32>,
    pub margin_top: Option<f32>,
    pub margin_right: Option<f32>,
    pub margin_bottom: Option<f32>,

    // Text
    pub font_size: Option<f32>,
    pub font_color: Option<Color>,
    pub font_id: Option<u32>,
    pub font_weight: Option<FontWeight>,
    pub font_style: Option<FontStyle>,
    pub text_align: Option<crate::render_commands::TextAlign>,
    pub line_height: Option<f32>,
    pub letter_spacing: Option<f32>,

    // Opacity / shadow
    pub opacity: Option<f32>,
    pub shadow: Option<Shadow>,

    // Cursor
    pub cursor: Option<CursorIcon>,

    // Size overrides
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub min_width: Option<f32>,
    pub max_width: Option<f32>,
    pub min_height: Option<f32>,
    pub max_height: Option<f32>,

    // Transforms
    pub scale: Option<f32>,
    pub rotation: Option<f32>,
    pub translate: Option<Vec2>,

    // Transitions (animated on state change).
    pub transitions: Vec<Transition>,
}

impl Style {
    /// Create a new empty (fully-inherited) style.
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge `other` on top of `self`. Fields set in `other` override `self`.
    pub fn merge(&self, other: &Style) -> Style {
        Style {
            background_color: other.background_color.or(self.background_color),
            background_gradient: other
                .background_gradient
                .clone()
                .or_else(|| self.background_gradient.clone()),
            border_color: other.border_color.or(self.border_color),
            border_width: other.border_width.or(self.border_width),
            border_radius: other.border_radius.or(self.border_radius),
            padding_left: other.padding_left.or(self.padding_left),
            padding_top: other.padding_top.or(self.padding_top),
            padding_right: other.padding_right.or(self.padding_right),
            padding_bottom: other.padding_bottom.or(self.padding_bottom),
            margin_left: other.margin_left.or(self.margin_left),
            margin_top: other.margin_top.or(self.margin_top),
            margin_right: other.margin_right.or(self.margin_right),
            margin_bottom: other.margin_bottom.or(self.margin_bottom),
            font_size: other.font_size.or(self.font_size),
            font_color: other.font_color.or(self.font_color),
            font_id: other.font_id.or(self.font_id),
            font_weight: other.font_weight.or(self.font_weight),
            font_style: other.font_style.or(self.font_style),
            text_align: other.text_align.or(self.text_align),
            line_height: other.line_height.or(self.line_height),
            letter_spacing: other.letter_spacing.or(self.letter_spacing),
            opacity: other.opacity.or(self.opacity),
            shadow: other.shadow.or(self.shadow),
            cursor: other.cursor.or(self.cursor),
            width: other.width.or(self.width),
            height: other.height.or(self.height),
            min_width: other.min_width.or(self.min_width),
            max_width: other.max_width.or(self.max_width),
            min_height: other.min_height.or(self.min_height),
            max_height: other.max_height.or(self.max_height),
            scale: other.scale.or(self.scale),
            rotation: other.rotation.or(self.rotation),
            translate: other.translate.or(self.translate),
            transitions: if other.transitions.is_empty() {
                self.transitions.clone()
            } else {
                other.transitions.clone()
            },
        }
    }

    /// Resolve all `None` fields to concrete defaults.
    pub fn resolved(&self) -> ResolvedStyle {
        ResolvedStyle {
            background_color: self.background_color.unwrap_or(Color::TRANSPARENT),
            border_color: self.border_color.unwrap_or(Color::TRANSPARENT),
            border_width: self.border_width.unwrap_or(0.0),
            border_radius: self.border_radius.unwrap_or(CornerRadii::ZERO),
            font_size: self.font_size.unwrap_or(14.0),
            font_color: self.font_color.unwrap_or(Color::BLACK),
            font_id: self.font_id.unwrap_or(0),
            font_weight: self.font_weight.unwrap_or(FontWeight::Normal),
            font_style: self.font_style.unwrap_or(FontStyle::Normal),
            opacity: self.opacity.unwrap_or(1.0),
            shadow: self.shadow,
            cursor: self.cursor.unwrap_or(CursorIcon::Default),
            text_align: self
                .text_align
                .unwrap_or(crate::render_commands::TextAlign::Left),
            line_height: self.line_height.unwrap_or(1.2),
            letter_spacing: self.letter_spacing.unwrap_or(0.0),
        }
    }

    // ---- Builder helpers ---------------------------------------------------

    pub fn with_bg(mut self, color: Color) -> Self {
        self.background_color = Some(color);
        self
    }

    pub fn with_font_color(mut self, color: Color) -> Self {
        self.font_color = Some(color);
        self
    }

    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = Some(size);
        self
    }

    pub fn with_border(mut self, color: Color, width: f32) -> Self {
        self.border_color = Some(color);
        self.border_width = Some(width);
        self
    }

    pub fn with_border_radius(mut self, radius: f32) -> Self {
        self.border_radius = Some(CornerRadii::all(radius));
        self
    }

    pub fn with_padding(mut self, all: f32) -> Self {
        self.padding_left = Some(all);
        self.padding_top = Some(all);
        self.padding_right = Some(all);
        self.padding_bottom = Some(all);
        self
    }

    pub fn with_margin(mut self, all: f32) -> Self {
        self.margin_left = Some(all);
        self.margin_top = Some(all);
        self.margin_right = Some(all);
        self.margin_bottom = Some(all);
        self
    }

    pub fn with_shadow(mut self, shadow: Shadow) -> Self {
        self.shadow = Some(shadow);
        self
    }

    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = Some(opacity);
        self
    }

    pub fn with_cursor(mut self, cursor: CursorIcon) -> Self {
        self.cursor = Some(cursor);
        self
    }

    pub fn with_transition(mut self, transition: Transition) -> Self {
        self.transitions.push(transition);
        self
    }
}

/// Style with all fields resolved to concrete values.
#[derive(Debug, Clone)]
pub struct ResolvedStyle {
    pub background_color: Color,
    pub border_color: Color,
    pub border_width: f32,
    pub border_radius: CornerRadii,
    pub font_size: f32,
    pub font_color: Color,
    pub font_id: u32,
    pub font_weight: FontWeight,
    pub font_style: FontStyle,
    pub opacity: f32,
    pub shadow: Option<Shadow>,
    pub cursor: CursorIcon,
    pub text_align: crate::render_commands::TextAlign,
    pub line_height: f32,
    pub letter_spacing: f32,
}

// ---------------------------------------------------------------------------
// Font style enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FontWeight {
    Thin,
    Light,
    Normal,
    Medium,
    SemiBold,
    Bold,
    ExtraBold,
    Black,
}

impl Default for FontWeight {
    fn default() -> Self {
        Self::Normal
    }
}

impl FontWeight {
    /// Numeric weight (100 - 900).
    pub fn numeric(&self) -> u16 {
        match self {
            Self::Thin => 100,
            Self::Light => 300,
            Self::Normal => 400,
            Self::Medium => 500,
            Self::SemiBold => 600,
            Self::Bold => 700,
            Self::ExtraBold => 800,
            Self::Black => 900,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

impl Default for FontStyle {
    fn default() -> Self {
        Self::Normal
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// Defines an animated transition for a style property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Name of the property (e.g. "background_color", "opacity").
    pub property: String,
    /// Duration in seconds.
    pub duration: f32,
    /// Delay before the transition starts, in seconds.
    pub delay: f32,
    /// Easing function.
    pub easing: crate::animation::EasingFunction,
}

impl Transition {
    pub fn new(property: &str, duration: f32) -> Self {
        Self {
            property: property.to_string(),
            duration,
            delay: 0.0,
            easing: crate::animation::EasingFunction::EaseInOut,
        }
    }

    pub fn with_delay(mut self, delay: f32) -> Self {
        self.delay = delay;
        self
    }

    pub fn with_easing(mut self, easing: crate::animation::EasingFunction) -> Self {
        self.easing = easing;
        self
    }
}

// ---------------------------------------------------------------------------
// StyleSheet
// ---------------------------------------------------------------------------

/// A named collection of styles, mapping (class_name, pseudo_state) to a
/// [`Style`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StyleSheet {
    /// Style rules keyed by (class_name, pseudo_state).
    rules: HashMap<String, HashMap<PseudoState, Style>>,
    /// Global (classless) styles per pseudo-state.
    global: HashMap<PseudoState, Style>,
}

impl StyleSheet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a style rule for a class + pseudo state.
    pub fn add_rule(&mut self, class: &str, state: PseudoState, style: Style) {
        self.rules
            .entry(class.to_string())
            .or_default()
            .insert(state, style);
    }

    /// Add a global style for a pseudo state (applied to all nodes).
    pub fn add_global(&mut self, state: PseudoState, style: Style) {
        self.global.insert(state, style);
    }

    /// Resolve the style for a set of classes and a pseudo state.
    ///
    /// Resolution order: global normal -> global state -> class normal ->
    /// class state. Later entries override earlier ones.
    pub fn resolve(&self, classes: &[String], state: PseudoState) -> Style {
        let mut result = Style::new();

        // Global normal.
        if let Some(s) = self.global.get(&PseudoState::Normal) {
            result = result.merge(s);
        }
        // Global state.
        if state != PseudoState::Normal {
            if let Some(s) = self.global.get(&state) {
                result = result.merge(s);
            }
        }

        // Per-class normal, then per-class state.
        for class in classes {
            if let Some(class_rules) = self.rules.get(class) {
                if let Some(s) = class_rules.get(&PseudoState::Normal) {
                    result = result.merge(s);
                }
                if state != PseudoState::Normal {
                    if let Some(s) = class_rules.get(&state) {
                        result = result.merge(s);
                    }
                }
            }
        }

        result
    }

    /// Merge another stylesheet into this one. Rules from `other` override.
    pub fn merge(&mut self, other: &StyleSheet) {
        for (state, style) in &other.global {
            let existing = self.global.entry(*state).or_insert_with(Style::new);
            *existing = existing.merge(style);
        }
        for (class, states) in &other.rules {
            let class_entry = self.rules.entry(class.clone()).or_default();
            for (state, style) in states {
                let existing = class_entry.entry(*state).or_insert_with(Style::new);
                *existing = existing.merge(style);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

/// A complete color palette for a UI theme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theme {
    pub name: String,

    // Surface colours
    pub primary: Color,
    pub primary_variant: Color,
    pub secondary: Color,
    pub secondary_variant: Color,
    pub accent: Color,
    pub background: Color,
    pub surface: Color,
    pub surface_variant: Color,
    pub error: Color,
    pub warning: Color,
    pub success: Color,
    pub info: Color,

    // On-colours (text/icon colour for contrast on the named surface)
    pub on_primary: Color,
    pub on_secondary: Color,
    pub on_background: Color,
    pub on_surface: Color,
    pub on_error: Color,

    // Additional
    pub divider: Color,
    pub disabled: Color,
    pub disabled_text: Color,
    pub overlay: Color,
    pub shadow_color: Color,
    pub focus_ring: Color,

    // Typography defaults
    pub default_font_size: f32,
    pub heading_font_size: f32,
    pub small_font_size: f32,
    pub font_id: u32,

    // Spacing defaults
    pub spacing_xs: f32,
    pub spacing_sm: f32,
    pub spacing_md: f32,
    pub spacing_lg: f32,
    pub spacing_xl: f32,

    // Shape defaults
    pub border_radius_sm: f32,
    pub border_radius_md: f32,
    pub border_radius_lg: f32,
    pub border_width: f32,

    // Elevation (shadow presets)
    pub elevation_low: Shadow,
    pub elevation_medium: Shadow,
    pub elevation_high: Shadow,
}

impl Theme {
    /// A dark theme preset.
    pub fn dark() -> Self {
        Self {
            name: "Dark".to_string(),
            primary: Color::from_hex("#BB86FC"),
            primary_variant: Color::from_hex("#3700B3"),
            secondary: Color::from_hex("#03DAC6"),
            secondary_variant: Color::from_hex("#018786"),
            accent: Color::from_hex("#FF4081"),
            background: Color::from_hex("#121212"),
            surface: Color::from_hex("#1E1E1E"),
            surface_variant: Color::from_hex("#2D2D2D"),
            error: Color::from_hex("#CF6679"),
            warning: Color::from_hex("#FF9800"),
            success: Color::from_hex("#4CAF50"),
            info: Color::from_hex("#2196F3"),
            on_primary: Color::BLACK,
            on_secondary: Color::BLACK,
            on_background: Color::WHITE,
            on_surface: Color::WHITE,
            on_error: Color::BLACK,
            divider: Color::from_rgba8(255, 255, 255, 30),
            disabled: Color::from_rgba8(255, 255, 255, 38),
            disabled_text: Color::from_rgba8(255, 255, 255, 100),
            overlay: Color::from_rgba8(0, 0, 0, 128),
            shadow_color: Color::from_rgba8(0, 0, 0, 80),
            focus_ring: Color::from_hex("#BB86FC").with_alpha(0.5),
            default_font_size: 14.0,
            heading_font_size: 24.0,
            small_font_size: 12.0,
            font_id: 0,
            spacing_xs: 4.0,
            spacing_sm: 8.0,
            spacing_md: 16.0,
            spacing_lg: 24.0,
            spacing_xl: 32.0,
            border_radius_sm: 4.0,
            border_radius_md: 8.0,
            border_radius_lg: 16.0,
            border_width: 1.0,
            elevation_low: Shadow::new(
                Color::from_rgba8(0, 0, 0, 40),
                Vec2::new(0.0, 1.0),
                3.0,
                0.0,
            ),
            elevation_medium: Shadow::new(
                Color::from_rgba8(0, 0, 0, 60),
                Vec2::new(0.0, 3.0),
                8.0,
                0.0,
            ),
            elevation_high: Shadow::new(
                Color::from_rgba8(0, 0, 0, 80),
                Vec2::new(0.0, 6.0),
                16.0,
                2.0,
            ),
        }
    }

    /// A light theme preset.
    pub fn light() -> Self {
        Self {
            name: "Light".to_string(),
            primary: Color::from_hex("#6200EE"),
            primary_variant: Color::from_hex("#3700B3"),
            secondary: Color::from_hex("#03DAC6"),
            secondary_variant: Color::from_hex("#018786"),
            accent: Color::from_hex("#FF4081"),
            background: Color::from_hex("#FAFAFA"),
            surface: Color::WHITE,
            surface_variant: Color::from_hex("#F5F5F5"),
            error: Color::from_hex("#B00020"),
            warning: Color::from_hex("#FF9800"),
            success: Color::from_hex("#4CAF50"),
            info: Color::from_hex("#2196F3"),
            on_primary: Color::WHITE,
            on_secondary: Color::BLACK,
            on_background: Color::from_hex("#212121"),
            on_surface: Color::from_hex("#212121"),
            on_error: Color::WHITE,
            divider: Color::from_rgba8(0, 0, 0, 30),
            disabled: Color::from_rgba8(0, 0, 0, 38),
            disabled_text: Color::from_rgba8(0, 0, 0, 100),
            overlay: Color::from_rgba8(0, 0, 0, 80),
            shadow_color: Color::from_rgba8(0, 0, 0, 40),
            focus_ring: Color::from_hex("#6200EE").with_alpha(0.3),
            default_font_size: 14.0,
            heading_font_size: 24.0,
            small_font_size: 12.0,
            font_id: 0,
            spacing_xs: 4.0,
            spacing_sm: 8.0,
            spacing_md: 16.0,
            spacing_lg: 24.0,
            spacing_xl: 32.0,
            border_radius_sm: 4.0,
            border_radius_md: 8.0,
            border_radius_lg: 16.0,
            border_width: 1.0,
            elevation_low: Shadow::new(
                Color::from_rgba8(0, 0, 0, 20),
                Vec2::new(0.0, 1.0),
                3.0,
                0.0,
            ),
            elevation_medium: Shadow::new(
                Color::from_rgba8(0, 0, 0, 30),
                Vec2::new(0.0, 3.0),
                8.0,
                0.0,
            ),
            elevation_high: Shadow::new(
                Color::from_rgba8(0, 0, 0, 50),
                Vec2::new(0.0, 6.0),
                16.0,
                2.0,
            ),
        }
    }

    /// Generate a default stylesheet from the theme.
    pub fn to_stylesheet(&self) -> StyleSheet {
        let mut sheet = StyleSheet::new();

        // Button styles.
        sheet.add_rule(
            "button",
            PseudoState::Normal,
            Style::new()
                .with_bg(self.primary)
                .with_font_color(self.on_primary)
                .with_font_size(self.default_font_size)
                .with_border_radius(self.border_radius_md)
                .with_padding(self.spacing_sm)
                .with_cursor(CursorIcon::Pointer),
        );
        sheet.add_rule(
            "button",
            PseudoState::Hovered,
            Style::new().with_bg(self.primary.lighten(0.1)),
        );
        sheet.add_rule(
            "button",
            PseudoState::Pressed,
            Style::new().with_bg(self.primary.darken(0.1)),
        );
        sheet.add_rule(
            "button",
            PseudoState::Disabled,
            Style::new()
                .with_bg(self.disabled)
                .with_font_color(self.disabled_text)
                .with_cursor(CursorIcon::NotAllowed),
        );

        // Text input styles.
        sheet.add_rule(
            "text_input",
            PseudoState::Normal,
            Style::new()
                .with_bg(self.surface)
                .with_font_color(self.on_surface)
                .with_border(self.divider, self.border_width)
                .with_border_radius(self.border_radius_sm)
                .with_padding(self.spacing_sm)
                .with_cursor(CursorIcon::Text),
        );
        sheet.add_rule(
            "text_input",
            PseudoState::Focused,
            Style::new().with_border(self.primary, 2.0),
        );
        sheet.add_rule(
            "text_input",
            PseudoState::Disabled,
            Style::new()
                .with_bg(self.disabled)
                .with_font_color(self.disabled_text),
        );

        // Panel / window styles.
        sheet.add_rule(
            "panel",
            PseudoState::Normal,
            Style::new()
                .with_bg(self.surface)
                .with_border_radius(self.border_radius_md)
                .with_shadow(self.elevation_medium),
        );

        // Label.
        sheet.add_rule(
            "label",
            PseudoState::Normal,
            Style::new()
                .with_font_color(self.on_background)
                .with_font_size(self.default_font_size),
        );

        // Checkbox.
        sheet.add_rule(
            "checkbox",
            PseudoState::Normal,
            Style::new()
                .with_border(self.on_surface.with_alpha(0.6), self.border_width)
                .with_border_radius(self.border_radius_sm)
                .with_cursor(CursorIcon::Pointer),
        );
        sheet.add_rule(
            "checkbox",
            PseudoState::Selected,
            Style::new()
                .with_bg(self.primary)
                .with_border(self.primary, self.border_width),
        );

        // Slider.
        sheet.add_rule(
            "slider_track",
            PseudoState::Normal,
            Style::new()
                .with_bg(self.divider)
                .with_border_radius(2.0),
        );
        sheet.add_rule(
            "slider_thumb",
            PseudoState::Normal,
            Style::new()
                .with_bg(self.primary)
                .with_border_radius(10.0)
                .with_shadow(self.elevation_low),
        );

        sheet
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

// ---------------------------------------------------------------------------
// ThemeManager
// ---------------------------------------------------------------------------

/// Manages the current theme and provides theme switching.
pub struct ThemeManager {
    themes: HashMap<String, Theme>,
    active: String,
    stylesheet: StyleSheet,
    /// Custom user stylesheet that overlays the theme's defaults.
    custom_stylesheet: StyleSheet,
}

impl ThemeManager {
    pub fn new() -> Self {
        let dark = Theme::dark();
        let light = Theme::light();
        let stylesheet = dark.to_stylesheet();
        let mut themes = HashMap::new();
        themes.insert("Dark".to_string(), dark);
        themes.insert("Light".to_string(), light);
        Self {
            themes,
            active: "Dark".to_string(),
            stylesheet,
            custom_stylesheet: StyleSheet::new(),
        }
    }

    /// Returns the active theme.
    pub fn active_theme(&self) -> &Theme {
        self.themes
            .get(&self.active)
            .expect("active theme must exist")
    }

    /// Returns the active theme's name.
    pub fn active_name(&self) -> &str {
        &self.active
    }

    /// Switch to a theme by name. Returns `false` if the name is unknown.
    pub fn set_theme(&mut self, name: &str) -> bool {
        if let Some(theme) = self.themes.get(name) {
            self.stylesheet = theme.to_stylesheet();
            self.stylesheet.merge(&self.custom_stylesheet);
            self.active = name.to_string();
            true
        } else {
            false
        }
    }

    /// Register a custom theme.
    pub fn register_theme(&mut self, theme: Theme) {
        self.themes.insert(theme.name.clone(), theme);
    }

    /// Returns the combined stylesheet (theme + custom overrides).
    pub fn stylesheet(&self) -> &StyleSheet {
        &self.stylesheet
    }

    /// Add custom style rules that overlay any theme.
    pub fn add_custom_rules(&mut self, sheet: &StyleSheet) {
        self.custom_stylesheet.merge(sheet);
        // Rebuild the active stylesheet.
        if let Some(theme) = self.themes.get(&self.active) {
            self.stylesheet = theme.to_stylesheet();
            self.stylesheet.merge(&self.custom_stylesheet);
        }
    }

    /// Resolve the style for a widget given its classes and pseudo state.
    pub fn resolve_style(&self, classes: &[String], state: PseudoState) -> Style {
        self.stylesheet.resolve(classes, state)
    }

    /// List all registered theme names.
    pub fn theme_names(&self) -> Vec<&str> {
        self.themes.keys().map(|s| s.as_str()).collect()
    }

    /// Toggle between dark and light themes.
    pub fn toggle_dark_light(&mut self) {
        let next = if self.active == "Dark" {
            "Light"
        } else {
            "Dark"
        };
        self.set_theme(next);
    }
}

impl Default for ThemeManager {
    fn default() -> Self {
        Self::new()
    }
}
