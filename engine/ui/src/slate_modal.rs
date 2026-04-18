//! Modal dialog system for the Genovo Slate UI.
//!
//! Provides:
//! - `ModalDialog`: centered overlay with backdrop dimming
//! - Focus trap: Tab key cycles within the dialog
//! - Built-in dialog types: message (OK), confirm (OK/Cancel), input dialog
//! - Custom content support
//! - Fade in/out animation
//! - Backdrop: semi-transparent overlay, optional click-to-dismiss

use glam::Vec2;
use genovo_core::Rect;

use crate::core::{KeyCode, KeyModifiers, MouseButton, Padding, UIEvent, UIId};
use crate::render_commands::{
    Border as BorderSpec, Color, CornerRadii, DrawCommand, DrawList, TextAlign,
    TextVerticalAlign, TextureId,
};
use crate::slate_widgets::EventReply;

// =========================================================================
// Constants
// =========================================================================

const DIALOG_MIN_WIDTH: f32 = 300.0;
const DIALOG_MAX_WIDTH: f32 = 600.0;
const DIALOG_PADDING: f32 = 24.0;
const BUTTON_HEIGHT: f32 = 32.0;
const BUTTON_MIN_WIDTH: f32 = 80.0;
const BUTTON_PADDING: f32 = 16.0;
const BUTTON_SPACING: f32 = 8.0;
const TITLE_HEIGHT: f32 = 28.0;
const INPUT_HEIGHT: f32 = 30.0;
const FADE_SPEED: f32 = 6.0;

// =========================================================================
// DialogResult
// =========================================================================

/// Result of a modal dialog interaction.
#[derive(Debug, Clone, PartialEq)]
pub enum ModalDialogResult {
    /// Dialog is still open, no result yet.
    Pending,
    /// OK / Confirm button was pressed.
    Ok,
    /// Cancel button was pressed (or Escape, or backdrop click).
    Cancel,
    /// OK was pressed with an input value.
    OkWithInput(String),
    /// A custom button was pressed, identified by index.
    Custom(usize),
}

impl Default for ModalDialogResult {
    fn default() -> Self {
        ModalDialogResult::Pending
    }
}

// =========================================================================
// DialogButton
// =========================================================================

/// A button in a modal dialog.
#[derive(Debug, Clone)]
pub struct ModalButton {
    /// Button label.
    pub label: String,
    /// Whether this is the primary (accented) button.
    pub primary: bool,
    /// Whether this button is enabled.
    pub enabled: bool,
    /// Result returned when clicked.
    pub result: ModalDialogResult,
    /// Whether the button is hovered.
    pub hovered: bool,
    /// Whether the button is pressed.
    pub pressed: bool,
}

impl ModalButton {
    pub fn ok() -> Self {
        Self {
            label: "OK".to_string(),
            primary: true,
            enabled: true,
            result: ModalDialogResult::Ok,
            hovered: false,
            pressed: false,
        }
    }

    pub fn cancel() -> Self {
        Self {
            label: "Cancel".to_string(),
            primary: false,
            enabled: true,
            result: ModalDialogResult::Cancel,
            hovered: false,
            pressed: false,
        }
    }

    pub fn custom(label: &str, index: usize, primary: bool) -> Self {
        Self {
            label: label.to_string(),
            primary,
            enabled: true,
            result: ModalDialogResult::Custom(index),
            hovered: false,
            pressed: false,
        }
    }
}

// =========================================================================
// DialogIcon
// =========================================================================

/// Icon type for built-in dialogs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalDialogIcon {
    None,
    Info,
    Warning,
    Error,
    Question,
}

impl Default for ModalDialogIcon {
    fn default() -> Self {
        ModalDialogIcon::None
    }
}

impl ModalDialogIcon {
    /// Get the colour associated with this icon type.
    pub fn color(&self) -> Color {
        match self {
            ModalDialogIcon::None => Color::TRANSPARENT,
            ModalDialogIcon::Info => Color::from_hex("#3794FF"),
            ModalDialogIcon::Warning => Color::from_hex("#CCA700"),
            ModalDialogIcon::Error => Color::from_hex("#F44747"),
            ModalDialogIcon::Question => Color::from_hex("#75BEFF"),
        }
    }

    /// Get the symbol character for this icon.
    pub fn symbol(&self) -> &str {
        match self {
            ModalDialogIcon::None => "",
            ModalDialogIcon::Info => "i",
            ModalDialogIcon::Warning => "!",
            ModalDialogIcon::Error => "X",
            ModalDialogIcon::Question => "?",
        }
    }
}

// =========================================================================
// ModalDialogStyle
// =========================================================================

/// Visual style for modal dialogs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModalDialogStyle {
    /// Backdrop colour (semi-transparent).
    pub backdrop_color: Color,
    /// Dialog background colour.
    pub background: Color,
    /// Dialog border colour.
    pub border_color: Color,
    /// Dialog border width.
    pub border_width: f32,
    /// Corner radius.
    pub corner_radius: f32,
    /// Title text colour.
    pub title_color: Color,
    /// Body text colour.
    pub body_color: Color,
    /// Title font size.
    pub title_font_size: f32,
    /// Body font size.
    pub body_font_size: f32,
    /// Font ID.
    pub font_id: u32,
    /// Primary button colour.
    pub primary_button_color: Color,
    /// Primary button hover colour.
    pub primary_button_hover: Color,
    /// Primary button text colour.
    pub primary_button_text: Color,
    /// Secondary button colour.
    pub secondary_button_color: Color,
    /// Secondary button hover colour.
    pub secondary_button_hover: Color,
    /// Secondary button text colour.
    pub secondary_button_text: Color,
    /// Shadow colour.
    pub shadow_color: Color,
    /// Shadow blur.
    pub shadow_blur: f32,
    /// Input field background.
    pub input_background: Color,
    /// Input field border.
    pub input_border: Color,
    /// Input text colour.
    pub input_text_color: Color,
    /// Input cursor colour.
    pub input_cursor_color: Color,
}

impl Default for ModalDialogStyle {
    fn default() -> Self {
        Self {
            backdrop_color: Color::new(0.0, 0.0, 0.0, 0.5),
            background: Color::from_hex("#252526"),
            border_color: Color::from_hex("#3F3F46"),
            border_width: 1.0,
            corner_radius: 8.0,
            title_color: Color::from_hex("#FFFFFF"),
            body_color: Color::from_hex("#CCCCCC"),
            title_font_size: 16.0,
            body_font_size: 13.0,
            font_id: 0,
            primary_button_color: Color::from_hex("#0E639C"),
            primary_button_hover: Color::from_hex("#1177BB"),
            primary_button_text: Color::WHITE,
            secondary_button_color: Color::from_hex("#3A3D41"),
            secondary_button_hover: Color::from_hex("#45494E"),
            secondary_button_text: Color::from_hex("#CCCCCC"),
            shadow_color: Color::new(0.0, 0.0, 0.0, 0.6),
            shadow_blur: 16.0,
            input_background: Color::from_hex("#1E1E1E"),
            input_border: Color::from_hex("#3F3F46"),
            input_text_color: Color::from_hex("#CCCCCC"),
            input_cursor_color: Color::from_hex("#AEAFAD"),
        }
    }
}

// =========================================================================
// ModalDialog
// =========================================================================

/// A modal dialog that overlays the entire screen with a backdrop and a
/// centered dialog box.
#[derive(Debug, Clone)]
pub struct ModalDialog {
    /// Unique ID.
    pub id: UIId,
    /// Dialog title (optional).
    pub title: Option<String>,
    /// Dialog message/body text.
    pub message: String,
    /// Icon type.
    pub icon: ModalDialogIcon,
    /// Buttons.
    pub buttons: Vec<ModalButton>,
    /// Visual style.
    pub style: ModalDialogStyle,
    /// Whether the dialog is open.
    pub is_open: bool,
    /// Computed dialog result.
    pub result: ModalDialogResult,
    /// Whether clicking the backdrop dismisses the dialog.
    pub dismiss_on_backdrop: bool,
    /// Whether pressing Escape dismisses the dialog.
    pub dismiss_on_escape: bool,

    // --- Animation ---
    /// Fade animation progress (0.0 = invisible, 1.0 = fully visible).
    pub fade: f32,
    /// Target fade value.
    pub fade_target: f32,

    // --- Input dialog ---
    /// Input text (for input dialogs).
    pub input_text: String,
    /// Input cursor position.
    pub input_cursor: usize,
    /// Whether this is an input dialog.
    pub has_input: bool,
    /// Input placeholder text.
    pub input_placeholder: String,
    /// Whether the input field is focused.
    pub input_focused: bool,

    // --- Focus management ---
    /// Currently focused button index (for Tab cycling).
    pub focused_button: Option<usize>,

    // --- Layout ---
    /// Dialog width (computed or explicit).
    pub width: f32,
    /// Screen size for centering.
    pub screen_size: Vec2,

    /// Custom content flag -- when true, the parent is responsible for
    /// painting inside the dialog content area.
    pub custom_content: bool,
    /// Custom content height (set by the parent before `paint`).
    pub custom_content_height: f32,
}

impl ModalDialog {
    /// Create a simple message dialog (OK button only).
    pub fn message(title: &str, message: &str) -> Self {
        Self {
            id: UIId::INVALID,
            title: Some(title.to_string()),
            message: message.to_string(),
            icon: ModalDialogIcon::Info,
            buttons: vec![ModalButton::ok()],
            style: ModalDialogStyle::default(),
            is_open: false,
            result: ModalDialogResult::Pending,
            dismiss_on_backdrop: false,
            dismiss_on_escape: true,
            fade: 0.0,
            fade_target: 0.0,
            input_text: String::new(),
            input_cursor: 0,
            has_input: false,
            input_placeholder: String::new(),
            input_focused: false,
            focused_button: Some(0),
            width: 400.0,
            screen_size: Vec2::new(1920.0, 1080.0),
            custom_content: false,
            custom_content_height: 0.0,
        }
    }

    /// Create a confirmation dialog (OK + Cancel).
    pub fn confirm(title: &str, message: &str) -> Self {
        let mut d = Self::message(title, message);
        d.icon = ModalDialogIcon::Question;
        d.buttons = vec![ModalButton::ok(), ModalButton::cancel()];
        d.focused_button = Some(0);
        d
    }

    /// Create a warning dialog.
    pub fn warning(title: &str, message: &str) -> Self {
        let mut d = Self::message(title, message);
        d.icon = ModalDialogIcon::Warning;
        d
    }

    /// Create an error dialog.
    pub fn error(title: &str, message: &str) -> Self {
        let mut d = Self::message(title, message);
        d.icon = ModalDialogIcon::Error;
        d
    }

    /// Create an input dialog (text field + OK/Cancel).
    pub fn input(title: &str, message: &str, placeholder: &str) -> Self {
        let mut d = Self::confirm(title, message);
        d.has_input = true;
        d.input_placeholder = placeholder.to_string();
        d.input_focused = true;
        d
    }

    /// Create a dialog with custom content.
    pub fn custom(title: &str, content_height: f32) -> Self {
        let mut d = Self::message(title, "");
        d.custom_content = true;
        d.custom_content_height = content_height;
        d.buttons = vec![ModalButton::ok(), ModalButton::cancel()];
        d
    }

    /// Open the dialog.
    pub fn open(&mut self, screen_size: Vec2) {
        self.screen_size = screen_size;
        self.is_open = true;
        self.result = ModalDialogResult::Pending;
        self.fade = 0.0;
        self.fade_target = 1.0;
        if self.has_input {
            self.input_focused = true;
        }
    }

    /// Close the dialog with a result.
    pub fn close(&mut self, result: ModalDialogResult) {
        self.result = result;
        self.fade_target = 0.0;
    }

    /// Whether the dialog is currently visible.
    pub fn is_visible(&self) -> bool {
        self.is_open && self.fade > 0.001
    }

    /// Whether the dialog has been resolved.
    pub fn is_resolved(&self) -> bool {
        self.result != ModalDialogResult::Pending
    }

    /// Compute the dialog rectangle.
    fn dialog_rect(&self) -> Rect {
        let h = self.compute_height();
        let x = (self.screen_size.x - self.width) * 0.5;
        let y = (self.screen_size.y - h) * 0.5 - 40.0; // Slightly above center.
        Rect::new(
            Vec2::new(x.max(0.0), y.max(0.0)),
            Vec2::new((x + self.width).min(self.screen_size.x), (y + h).min(self.screen_size.y)),
        )
    }

    fn compute_height(&self) -> f32 {
        let mut h = DIALOG_PADDING; // Top padding.

        // Title.
        if self.title.is_some() {
            h += TITLE_HEIGHT + 8.0;
        }

        // Message body.
        if !self.message.is_empty() {
            // Approximate text height (lines * line_height).
            let char_per_line = ((self.width - DIALOG_PADDING * 2.0) / 7.5).max(1.0);
            let lines = (self.message.len() as f32 / char_per_line).ceil().max(1.0);
            h += lines * (self.style.body_font_size + 4.0) + 12.0;
        }

        // Custom content.
        if self.custom_content {
            h += self.custom_content_height + 12.0;
        }

        // Input field.
        if self.has_input {
            h += INPUT_HEIGHT + 12.0;
        }

        // Buttons.
        h += BUTTON_HEIGHT + DIALOG_PADDING;

        h
    }

    /// Compute button rects.
    fn button_rects(&self, dialog_rect: Rect) -> Vec<Rect> {
        let count = self.buttons.len();
        if count == 0 {
            return Vec::new();
        }

        let total_width: f32 = self.buttons.iter().map(|b| {
            let text_w = b.label.len() as f32 * 8.0;
            text_w.max(BUTTON_MIN_WIDTH) + BUTTON_PADDING * 2.0
        }).sum::<f32>() + BUTTON_SPACING * (count as f32 - 1.0);

        let start_x = dialog_rect.max.x - DIALOG_PADDING - total_width;
        let y = dialog_rect.max.y - DIALOG_PADDING - BUTTON_HEIGHT;

        let mut rects = Vec::new();
        let mut x = start_x;
        for button in &self.buttons {
            let text_w = button.label.len() as f32 * 8.0;
            let w = text_w.max(BUTTON_MIN_WIDTH) + BUTTON_PADDING * 2.0;
            rects.push(Rect::new(
                Vec2::new(x, y),
                Vec2::new(x + w, y + BUTTON_HEIGHT),
            ));
            x += w + BUTTON_SPACING;
        }

        rects
    }

    /// Compute input field rect.
    fn input_rect(&self, dialog_rect: Rect) -> Rect {
        let mut y = dialog_rect.min.y + DIALOG_PADDING;
        if self.title.is_some() {
            y += TITLE_HEIGHT + 8.0;
        }
        if !self.message.is_empty() {
            let char_per_line = ((self.width - DIALOG_PADDING * 2.0) / 7.5).max(1.0);
            let lines = (self.message.len() as f32 / char_per_line).ceil().max(1.0);
            y += lines * (self.style.body_font_size + 4.0) + 12.0;
        }

        Rect::new(
            Vec2::new(dialog_rect.min.x + DIALOG_PADDING, y),
            Vec2::new(dialog_rect.max.x - DIALOG_PADDING, y + INPUT_HEIGHT),
        )
    }

    /// Content rect for custom content dialogs.
    pub fn content_rect(&self) -> Rect {
        let dialog_rect = self.dialog_rect();
        let mut y = dialog_rect.min.y + DIALOG_PADDING;
        if self.title.is_some() {
            y += TITLE_HEIGHT + 8.0;
        }
        Rect::new(
            Vec2::new(dialog_rect.min.x + DIALOG_PADDING, y),
            Vec2::new(
                dialog_rect.max.x - DIALOG_PADDING,
                y + self.custom_content_height,
            ),
        )
    }

    /// Update animations.
    pub fn update(&mut self, dt: f32) {
        // Fade animation.
        if (self.fade - self.fade_target).abs() > 0.001 {
            if self.fade_target > self.fade {
                self.fade = (self.fade + FADE_SPEED * dt).min(self.fade_target);
            } else {
                self.fade = (self.fade - FADE_SPEED * dt).max(self.fade_target);
            }
        }

        // Close when fade-out completes.
        if self.fade_target == 0.0 && self.fade <= 0.001 {
            self.is_open = false;
        }
    }

    /// Paint the dialog.
    pub fn paint(&self, draw: &mut DrawList) {
        if !self.is_visible() {
            return;
        }

        let alpha = self.fade;
        let dialog_rect = self.dialog_rect();

        // Backdrop.
        draw.commands.push(DrawCommand::Rect {
            rect: Rect::new(Vec2::ZERO, self.screen_size),
            color: self.style.backdrop_color.with_alpha(self.style.backdrop_color.a * alpha),
            corner_radii: CornerRadii::ZERO,
            border: BorderSpec::default(),
            shadow: None,
        });

        // Dialog shadow.
        draw.commands.push(DrawCommand::Rect {
            rect: Rect::new(
                dialog_rect.min + Vec2::new(4.0, 4.0),
                dialog_rect.max + Vec2::new(self.style.shadow_blur, self.style.shadow_blur),
            ),
            color: self.style.shadow_color.with_alpha(self.style.shadow_color.a * alpha * 0.6),
            corner_radii: CornerRadii::all(self.style.corner_radius + 2.0),
            border: BorderSpec::default(),
            shadow: None,
        });

        // Dialog background.
        draw.commands.push(DrawCommand::Rect {
            rect: dialog_rect,
            color: self.style.background.with_alpha(alpha),
            corner_radii: CornerRadii::all(self.style.corner_radius),
            border: BorderSpec::new(
                self.style.border_color.with_alpha(alpha),
                self.style.border_width,
            ),
            shadow: None,
        });

        let mut y = dialog_rect.min.y + DIALOG_PADDING;

        // Icon + Title.
        if let Some(ref title) = self.title {
            let icon_offset = if self.icon != ModalDialogIcon::None {
                // Draw icon circle.
                let icon_r = 12.0;
                let icon_cx = dialog_rect.min.x + DIALOG_PADDING + icon_r;
                let icon_cy = y + TITLE_HEIGHT * 0.5;
                draw.commands.push(DrawCommand::Circle {
                    center: Vec2::new(icon_cx, icon_cy),
                    radius: icon_r,
                    color: self.icon.color().with_alpha(alpha),
                    border: BorderSpec::default(),
                });
                draw.commands.push(DrawCommand::Text {
                    text: self.icon.symbol().to_string(),
                    position: Vec2::new(icon_cx - 4.0, icon_cy - 8.0),
                    font_size: 14.0,
                    color: Color::WHITE.with_alpha(alpha),
                    font_id: self.style.font_id,
                    max_width: None,
                    align: TextAlign::Center,
                    vertical_align: TextVerticalAlign::Top,
                });
                icon_r * 2.0 + 12.0
            } else {
                0.0
            };

            draw.commands.push(DrawCommand::Text {
                text: title.clone(),
                position: Vec2::new(
                    dialog_rect.min.x + DIALOG_PADDING + icon_offset,
                    y + 4.0,
                ),
                font_size: self.style.title_font_size,
                color: self.style.title_color.with_alpha(alpha),
                font_id: self.style.font_id,
                max_width: Some(self.width - DIALOG_PADDING * 2.0 - icon_offset),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
            y += TITLE_HEIGHT + 8.0;
        }

        // Body message.
        if !self.message.is_empty() {
            draw.commands.push(DrawCommand::Text {
                text: self.message.clone(),
                position: Vec2::new(dialog_rect.min.x + DIALOG_PADDING, y),
                font_size: self.style.body_font_size,
                color: self.style.body_color.with_alpha(alpha),
                font_id: self.style.font_id,
                max_width: Some(self.width - DIALOG_PADDING * 2.0),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }

        // Input field.
        if self.has_input {
            let input_r = self.input_rect(dialog_rect);
            draw.commands.push(DrawCommand::Rect {
                rect: input_r,
                color: self.style.input_background.with_alpha(alpha),
                corner_radii: CornerRadii::all(4.0),
                border: BorderSpec::new(
                    if self.input_focused {
                        self.style.primary_button_color.with_alpha(alpha)
                    } else {
                        self.style.input_border.with_alpha(alpha)
                    },
                    1.0,
                ),
                shadow: None,
            });

            let display_text = if self.input_text.is_empty() {
                &self.input_placeholder
            } else {
                &self.input_text
            };
            let text_color = if self.input_text.is_empty() {
                self.style.body_color.with_alpha(alpha * 0.5)
            } else {
                self.style.input_text_color.with_alpha(alpha)
            };

            draw.commands.push(DrawCommand::Text {
                text: display_text.clone(),
                position: Vec2::new(input_r.min.x + 8.0, input_r.min.y + 7.0),
                font_size: self.style.body_font_size,
                color: text_color,
                font_id: self.style.font_id,
                max_width: Some(input_r.width() - 16.0),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            // Cursor.
            if self.input_focused && !self.input_text.is_empty() {
                let cursor_x = input_r.min.x + 8.0
                    + self.input_text[..self.input_cursor.min(self.input_text.len())]
                        .len() as f32
                        * 7.5;
                draw.commands.push(DrawCommand::Line {
                    start: Vec2::new(cursor_x, input_r.min.y + 5.0),
                    end: Vec2::new(cursor_x, input_r.max.y - 5.0),
                    color: self.style.input_cursor_color.with_alpha(alpha),
                    thickness: 1.5,
                });
            }
        }

        // Buttons.
        let button_rects = self.button_rects(dialog_rect);
        for (i, (button, rect)) in self.buttons.iter().zip(button_rects.iter()).enumerate() {
            let is_focused = self.focused_button == Some(i);
            let bg_color = if button.pressed {
                if button.primary {
                    self.style.primary_button_color.darken(0.15)
                } else {
                    self.style.secondary_button_color.darken(0.15)
                }
            } else if button.hovered {
                if button.primary {
                    self.style.primary_button_hover
                } else {
                    self.style.secondary_button_hover
                }
            } else if button.primary {
                self.style.primary_button_color
            } else {
                self.style.secondary_button_color
            };

            let text_color = if button.primary {
                self.style.primary_button_text
            } else {
                self.style.secondary_button_text
            };

            // Focus ring.
            if is_focused {
                draw.commands.push(DrawCommand::Rect {
                    rect: Rect::new(
                        rect.min - Vec2::new(2.0, 2.0),
                        rect.max + Vec2::new(2.0, 2.0),
                    ),
                    color: Color::TRANSPARENT,
                    corner_radii: CornerRadii::all(6.0),
                    border: BorderSpec::new(
                        self.style.primary_button_color.with_alpha(alpha * 0.7),
                        2.0,
                    ),
                    shadow: None,
                });
            }

            draw.commands.push(DrawCommand::Rect {
                rect: *rect,
                color: bg_color.with_alpha(alpha),
                corner_radii: CornerRadii::all(4.0),
                border: BorderSpec::default(),
                shadow: None,
            });

            draw.commands.push(DrawCommand::Text {
                text: button.label.clone(),
                position: Vec2::new(
                    (rect.min.x + rect.max.x) * 0.5,
                    rect.min.y + (BUTTON_HEIGHT - self.style.body_font_size) * 0.5,
                ),
                font_size: self.style.body_font_size,
                color: text_color.with_alpha(alpha),
                font_id: self.style.font_id,
                max_width: None,
                align: TextAlign::Center,
                vertical_align: TextVerticalAlign::Top,
            });
        }
    }

    /// Handle events. Returns EventReply.
    pub fn handle_event(&mut self, event: &UIEvent) -> EventReply {
        if !self.is_visible() {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                let pos = *position;
                let dialog_rect = self.dialog_rect();
                let button_rects = self.button_rects(dialog_rect);

                for (i, rect) in button_rects.iter().enumerate() {
                    if i < self.buttons.len() {
                        self.buttons[i].hovered = rect.contains(pos);
                    }
                }

                EventReply::Handled // Trap all mouse events.
            }

            UIEvent::Click { position, button, .. } => {
                if *button != MouseButton::Left {
                    return EventReply::Handled;
                }

                let pos = *position;
                let dialog_rect = self.dialog_rect();
                let button_rects = self.button_rects(dialog_rect);

                // Check button clicks.
                for (i, rect) in button_rects.iter().enumerate() {
                    if rect.contains(pos) && i < self.buttons.len() && self.buttons[i].enabled {
                        self.buttons[i].pressed = true;
                        return EventReply::Handled;
                    }
                }

                // Check input field click.
                if self.has_input {
                    let input_r = self.input_rect(dialog_rect);
                    if input_r.contains(pos) {
                        self.input_focused = true;
                        // Approximate cursor position.
                        let rel_x = pos.x - input_r.min.x - 8.0;
                        self.input_cursor = (rel_x / 7.5).round() as usize;
                        self.input_cursor = self.input_cursor.min(self.input_text.len());
                        return EventReply::Handled;
                    }
                }

                // Backdrop click.
                if !dialog_rect.contains(pos) && self.dismiss_on_backdrop {
                    self.close(ModalDialogResult::Cancel);
                }

                EventReply::Handled
            }

            UIEvent::MouseUp { position, button } => {
                if *button != MouseButton::Left {
                    return EventReply::Handled;
                }

                let pos = *position;
                let dialog_rect = self.dialog_rect();
                let button_rects = self.button_rects(dialog_rect);

                for (i, rect) in button_rects.iter().enumerate() {
                    if i < self.buttons.len() && self.buttons[i].pressed {
                        self.buttons[i].pressed = false;
                        if rect.contains(pos) && self.buttons[i].enabled {
                            let result = if self.has_input
                                && self.buttons[i].result == ModalDialogResult::Ok
                            {
                                ModalDialogResult::OkWithInput(self.input_text.clone())
                            } else {
                                self.buttons[i].result.clone()
                            };
                            self.close(result);
                            return EventReply::Handled;
                        }
                    }
                }

                EventReply::Handled
            }

            UIEvent::KeyInput { key, pressed, modifiers } => {
                if *pressed {
                    self.handle_key(*key, *modifiers)
                } else {
                    EventReply::Handled
                }
            }

            UIEvent::TextInput { character } => {
                if self.has_input && self.input_focused {
                    if !character.is_control() {
                        self.input_text.insert(self.input_cursor, *character);
                        self.input_cursor += character.len_utf8();
                    }
                    return EventReply::Handled;
                }
                EventReply::Handled
            }

            _ => EventReply::Handled, // Trap all events.
        }
    }

    fn handle_key(&mut self, key: KeyCode, modifiers: KeyModifiers) -> EventReply {
        match key {
            KeyCode::Escape => {
                if self.dismiss_on_escape {
                    self.close(ModalDialogResult::Cancel);
                }
            }

            KeyCode::Enter => {
                // Press the focused button.
                if let Some(idx) = self.focused_button {
                    if idx < self.buttons.len() && self.buttons[idx].enabled {
                        let result = if self.has_input
                            && self.buttons[idx].result == ModalDialogResult::Ok
                        {
                            ModalDialogResult::OkWithInput(self.input_text.clone())
                        } else {
                            self.buttons[idx].result.clone()
                        };
                        self.close(result);
                    }
                }
            }

            KeyCode::Tab => {
                // Cycle focus.
                let count = self.buttons.len();
                if count > 0 {
                    if modifiers.shift {
                        self.focused_button = Some(match self.focused_button {
                            Some(0) | None => count - 1,
                            Some(i) => i - 1,
                        });
                    } else {
                        self.focused_button = Some(match self.focused_button {
                            Some(i) if i >= count - 1 => 0,
                            Some(i) => i + 1,
                            None => 0,
                        });
                    }
                }
            }

            KeyCode::Backspace => {
                if self.has_input && self.input_focused && self.input_cursor > 0 {
                    // Remove the character before the cursor.
                    let mut chars: Vec<char> = self.input_text.chars().collect();
                    let char_idx = self.input_text[..self.input_cursor]
                        .chars()
                        .count()
                        .saturating_sub(1);
                    if char_idx < chars.len() {
                        let removed = chars.remove(char_idx);
                        self.input_cursor -= removed.len_utf8();
                        self.input_text = chars.into_iter().collect();
                    }
                }
            }

            KeyCode::Delete => {
                if self.has_input && self.input_focused && self.input_cursor < self.input_text.len()
                {
                    let char_count = self.input_text[..self.input_cursor].chars().count();
                    let mut chars: Vec<char> = self.input_text.chars().collect();
                    if char_count < chars.len() {
                        chars.remove(char_count);
                        self.input_text = chars.into_iter().collect();
                    }
                }
            }

            KeyCode::ArrowLeft => {
                if self.has_input && self.input_focused && self.input_cursor > 0 {
                    // Move cursor left by one character.
                    let mut new_cursor = self.input_cursor - 1;
                    while new_cursor > 0 && !self.input_text.is_char_boundary(new_cursor) {
                        new_cursor -= 1;
                    }
                    self.input_cursor = new_cursor;
                }
            }

            KeyCode::ArrowRight => {
                if self.has_input
                    && self.input_focused
                    && self.input_cursor < self.input_text.len()
                {
                    let mut new_cursor = self.input_cursor + 1;
                    while new_cursor < self.input_text.len()
                        && !self.input_text.is_char_boundary(new_cursor)
                    {
                        new_cursor += 1;
                    }
                    self.input_cursor = new_cursor;
                }
            }

            KeyCode::Home => {
                if self.has_input && self.input_focused {
                    self.input_cursor = 0;
                }
            }

            KeyCode::End => {
                if self.has_input && self.input_focused {
                    self.input_cursor = self.input_text.len();
                }
            }

            _ => {}
        }

        EventReply::Handled
    }
}

impl Default for ModalDialog {
    fn default() -> Self {
        Self::message("", "")
    }
}

// =========================================================================
// ModalDialogStack
// =========================================================================

/// Manages a stack of modal dialogs. Only the topmost dialog receives input.
#[derive(Debug, Clone)]
pub struct ModalDialogStack {
    /// Dialog stack (topmost = last).
    pub dialogs: Vec<ModalDialog>,
}

impl ModalDialogStack {
    pub fn new() -> Self {
        Self {
            dialogs: Vec::new(),
        }
    }

    /// Push a new dialog onto the stack.
    pub fn push(&mut self, mut dialog: ModalDialog, screen_size: Vec2) {
        dialog.open(screen_size);
        self.dialogs.push(dialog);
    }

    /// Pop the topmost dialog.
    pub fn pop(&mut self) -> Option<ModalDialog> {
        self.dialogs.pop()
    }

    /// Whether any dialog is open.
    pub fn has_open(&self) -> bool {
        self.dialogs.iter().any(|d| d.is_open)
    }

    /// Get the result of the topmost dialog without removing it.
    pub fn top_result(&self) -> Option<&ModalDialogResult> {
        self.dialogs.last().map(|d| &d.result)
    }

    /// Take the result of the topmost dialog, closing it if resolved.
    pub fn take_result(&mut self) -> Option<ModalDialogResult> {
        if let Some(top) = self.dialogs.last() {
            if top.is_resolved() {
                return self.dialogs.pop().map(|d| d.result);
            }
        }
        None
    }

    /// Update all dialogs.
    pub fn update(&mut self, dt: f32) {
        // Remove closed dialogs.
        self.dialogs.retain(|d| d.is_open || d.fade > 0.001);

        for dialog in &mut self.dialogs {
            dialog.update(dt);
        }
    }

    /// Paint all dialogs (bottom to top).
    pub fn paint(&self, draw: &mut DrawList) {
        for dialog in &self.dialogs {
            dialog.paint(draw);
        }
    }

    /// Handle event (only the topmost dialog receives events).
    pub fn handle_event(&mut self, event: &UIEvent) -> EventReply {
        if let Some(top) = self.dialogs.last_mut() {
            if top.is_visible() {
                return top.handle_event(event);
            }
        }
        EventReply::Unhandled
    }
}

impl Default for ModalDialogStack {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_dialog() {
        let d = ModalDialog::message("Test", "Hello World");
        assert_eq!(d.buttons.len(), 1);
        assert!(d.title.is_some());
    }

    #[test]
    fn test_confirm_dialog() {
        let d = ModalDialog::confirm("Confirm", "Are you sure?");
        assert_eq!(d.buttons.len(), 2);
    }

    #[test]
    fn test_input_dialog() {
        let d = ModalDialog::input("Input", "Enter name:", "Name...");
        assert!(d.has_input);
        assert_eq!(d.input_placeholder, "Name...");
    }

    #[test]
    fn test_dialog_open_close() {
        let mut d = ModalDialog::message("Test", "Body");
        d.open(Vec2::new(1920.0, 1080.0));
        assert!(d.is_open);
        d.close(ModalDialogResult::Ok);
        assert_eq!(d.result, ModalDialogResult::Ok);
    }

    #[test]
    fn test_dialog_stack() {
        let mut stack = ModalDialogStack::new();
        assert!(!stack.has_open());

        let d = ModalDialog::message("Test", "Body");
        stack.push(d, Vec2::new(1920.0, 1080.0));
        assert!(stack.has_open());
    }

    #[test]
    fn test_dialog_rect_on_screen() {
        let mut d = ModalDialog::message("Test", "Body");
        d.open(Vec2::new(1920.0, 1080.0));
        let r = d.dialog_rect();
        assert!(r.min.x >= 0.0);
        assert!(r.min.y >= 0.0);
        assert!(r.max.x <= 1920.0);
        assert!(r.max.y <= 1080.0);
    }
}
