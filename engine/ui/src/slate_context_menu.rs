//! Right-click context menus for the Genovo Slate UI.
//!
//! Provides:
//! - `ContextMenu`: positioned popup with a list of items
//! - `ContextMenuItem`: label, shortcut, icon, enabled/disabled, separator, submenu
//! - Popup positioning with screen-edge clamping
//! - Keyboard navigation (arrow keys, Enter, Escape)
//! - Submenu expansion with hover delay
//! - Click-outside to close

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

const MENU_ITEM_HEIGHT: f32 = 26.0;
const SEPARATOR_HEIGHT: f32 = 9.0;
const MENU_PADDING: f32 = 4.0;
const SUBMENU_DELAY: f32 = 0.35;
const ICON_SIZE: f32 = 16.0;
const ICON_PADDING: f32 = 6.0;
const SHORTCUT_PADDING: f32 = 32.0;
const SUBMENU_ARROW_WIDTH: f32 = 16.0;
const MIN_MENU_WIDTH: f32 = 160.0;
const MAX_MENU_WIDTH: f32 = 400.0;
const MENU_SHADOW_SIZE: f32 = 8.0;

// =========================================================================
// ContextMenuAction
// =========================================================================

/// Action identifier returned when a menu item is selected.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MenuActionId(pub u64);

impl MenuActionId {
    pub const NONE: Self = Self(0);
}

impl Default for MenuActionId {
    fn default() -> Self {
        Self::NONE
    }
}

// =========================================================================
// ContextMenuItem
// =========================================================================

/// A single item in a context menu.
#[derive(Debug, Clone)]
pub enum ContextMenuItem {
    /// A clickable action item.
    Action {
        /// Display label.
        label: String,
        /// Action ID (returned on click).
        action: MenuActionId,
        /// Keyboard shortcut text (display only, e.g. "Ctrl+C").
        shortcut: Option<String>,
        /// Icon texture.
        icon: Option<TextureId>,
        /// Whether this item is enabled.
        enabled: bool,
        /// Whether this item is checked/toggled.
        checked: bool,
    },
    /// A separator line.
    Separator,
    /// A submenu.
    SubMenu {
        /// Display label.
        label: String,
        /// Icon texture.
        icon: Option<TextureId>,
        /// Submenu items.
        items: Vec<ContextMenuItem>,
        /// Whether this submenu is enabled.
        enabled: bool,
    },
    /// A header/section label (non-interactive).
    Header {
        label: String,
    },
}

impl ContextMenuItem {
    /// Create a simple action item.
    pub fn action(label: &str, action_id: u64) -> Self {
        ContextMenuItem::Action {
            label: label.to_string(),
            action: MenuActionId(action_id),
            shortcut: None,
            icon: None,
            enabled: true,
            checked: false,
        }
    }

    /// Create an action item with a shortcut label.
    pub fn action_with_shortcut(label: &str, action_id: u64, shortcut: &str) -> Self {
        ContextMenuItem::Action {
            label: label.to_string(),
            action: MenuActionId(action_id),
            shortcut: Some(shortcut.to_string()),
            icon: None,
            enabled: true,
            checked: false,
        }
    }

    /// Create a disabled action item.
    pub fn action_disabled(label: &str, action_id: u64) -> Self {
        ContextMenuItem::Action {
            label: label.to_string(),
            action: MenuActionId(action_id),
            shortcut: None,
            icon: None,
            enabled: false,
            checked: false,
        }
    }

    /// Create a checked/toggle action item.
    pub fn action_checked(label: &str, action_id: u64, checked: bool) -> Self {
        ContextMenuItem::Action {
            label: label.to_string(),
            action: MenuActionId(action_id),
            shortcut: None,
            icon: None,
            enabled: true,
            checked,
        }
    }

    /// Create a submenu item.
    pub fn submenu(label: &str, items: Vec<ContextMenuItem>) -> Self {
        ContextMenuItem::SubMenu {
            label: label.to_string(),
            icon: None,
            items,
            enabled: true,
        }
    }

    /// Create a separator.
    pub fn separator() -> Self {
        ContextMenuItem::Separator
    }

    /// Create a header.
    pub fn header(label: &str) -> Self {
        ContextMenuItem::Header {
            label: label.to_string(),
        }
    }

    /// Height of this item.
    fn height(&self) -> f32 {
        match self {
            ContextMenuItem::Action { .. } | ContextMenuItem::SubMenu { .. } => MENU_ITEM_HEIGHT,
            ContextMenuItem::Separator => SEPARATOR_HEIGHT,
            ContextMenuItem::Header { .. } => MENU_ITEM_HEIGHT,
        }
    }

    /// Whether this item is interactive (can be hovered/clicked).
    fn is_interactive(&self) -> bool {
        match self {
            ContextMenuItem::Action { enabled, .. } => *enabled,
            ContextMenuItem::SubMenu { enabled, .. } => *enabled,
            ContextMenuItem::Separator | ContextMenuItem::Header { .. } => false,
        }
    }
}

// =========================================================================
// ContextMenuStyle
// =========================================================================

/// Visual style for context menus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextMenuStyle {
    pub background: Color,
    pub border_color: Color,
    pub border_width: f32,
    pub corner_radius: f32,
    pub item_hover_color: Color,
    pub text_color: Color,
    pub text_disabled_color: Color,
    pub shortcut_color: Color,
    pub separator_color: Color,
    pub header_color: Color,
    pub check_color: Color,
    pub shadow_color: Color,
    pub shadow_blur: f32,
    pub font_size: f32,
    pub font_id: u32,
}

impl Default for ContextMenuStyle {
    fn default() -> Self {
        Self {
            background: Color::from_hex("#2D2D30"),
            border_color: Color::from_hex("#3F3F46"),
            border_width: 1.0,
            corner_radius: 4.0,
            item_hover_color: Color::from_hex("#094771"),
            text_color: Color::from_hex("#CCCCCC"),
            text_disabled_color: Color::from_hex("#656565"),
            shortcut_color: Color::from_hex("#888888"),
            separator_color: Color::from_hex("#3F3F46"),
            header_color: Color::from_hex("#888888"),
            check_color: Color::from_hex("#3794FF"),
            shadow_color: Color::new(0.0, 0.0, 0.0, 0.5),
            shadow_blur: MENU_SHADOW_SIZE,
            font_size: 13.0,
            font_id: 0,
        }
    }
}

// =========================================================================
// MenuState
// =========================================================================

/// Tracks the open/close state and interaction for a single menu level.
#[derive(Debug, Clone)]
struct MenuLevel {
    /// Items at this level.
    items: Vec<ContextMenuItem>,
    /// Currently hovered item index (None if no item hovered).
    hovered: Option<usize>,
    /// Position of this menu (top-left corner).
    position: Vec2,
    /// Computed menu size.
    size: Vec2,
    /// Time the current item has been hovered (for submenu delay).
    hover_time: f32,
    /// Whether this level's submenu is expanded.
    submenu_open: bool,
}

impl MenuLevel {
    fn new(items: Vec<ContextMenuItem>, position: Vec2) -> Self {
        let width = Self::compute_width(&items);
        let height = items.iter().map(|i| i.height()).sum::<f32>() + MENU_PADDING * 2.0;
        Self {
            items,
            hovered: None,
            position,
            size: Vec2::new(width, height),
            hover_time: 0.0,
            submenu_open: false,
        }
    }

    fn compute_width(items: &[ContextMenuItem]) -> f32 {
        let mut max_label_w: f32 = 0.0;
        let mut max_shortcut_w: f32 = 0.0;
        let mut has_icon = false;
        let mut has_submenu = false;

        for item in items {
            match item {
                ContextMenuItem::Action {
                    label, shortcut, icon, ..
                } => {
                    let label_w = label.len() as f32 * 7.5; // Approximate.
                    max_label_w = max_label_w.max(label_w);
                    if let Some(sc) = shortcut {
                        let sc_w = sc.len() as f32 * 7.0;
                        max_shortcut_w = max_shortcut_w.max(sc_w);
                    }
                    if icon.is_some() {
                        has_icon = true;
                    }
                }
                ContextMenuItem::SubMenu { label, icon, .. } => {
                    let label_w = label.len() as f32 * 7.5;
                    max_label_w = max_label_w.max(label_w);
                    has_submenu = true;
                    if icon.is_some() {
                        has_icon = true;
                    }
                }
                ContextMenuItem::Header { label } => {
                    let label_w = label.len() as f32 * 7.0;
                    max_label_w = max_label_w.max(label_w);
                }
                _ => {}
            }
        }

        let mut width = MENU_PADDING * 2.0 + max_label_w;
        if has_icon {
            width += ICON_SIZE + ICON_PADDING * 2.0;
        }
        if max_shortcut_w > 0.0 {
            width += SHORTCUT_PADDING + max_shortcut_w;
        }
        if has_submenu {
            width += SUBMENU_ARROW_WIDTH;
        }

        width.clamp(MIN_MENU_WIDTH, MAX_MENU_WIDTH)
    }

    fn rect(&self) -> Rect {
        Rect::new(self.position, self.position + self.size)
    }

    /// Compute the rect for item at index.
    fn item_rect(&self, index: usize) -> Rect {
        let mut y = self.position.y + MENU_PADDING;
        for (i, item) in self.items.iter().enumerate() {
            let h = item.height();
            if i == index {
                return Rect::new(
                    Vec2::new(self.position.x + MENU_PADDING, y),
                    Vec2::new(self.position.x + self.size.x - MENU_PADDING, y + h),
                );
            }
            y += h;
        }
        Rect::new(Vec2::ZERO, Vec2::ZERO)
    }

    /// Find item index at a given position.
    fn item_at(&self, pos: Vec2) -> Option<usize> {
        let r = self.rect();
        if !r.contains(pos) {
            return None;
        }

        let mut y = self.position.y + MENU_PADDING;
        for (i, item) in self.items.iter().enumerate() {
            let h = item.height();
            if pos.y >= y && pos.y < y + h {
                if item.is_interactive() {
                    return Some(i);
                }
                return None;
            }
            y += h;
        }
        None
    }
}

// =========================================================================
// ContextMenu
// =========================================================================

/// A right-click context menu popup.
///
/// Usage:
/// 1. Create with `ContextMenu::new(items)`
/// 2. Call `open(position, screen_size)` to show
/// 3. Call `handle_event` each frame
/// 4. Check `take_action()` for selected items
/// 5. Call `paint` to render
#[derive(Debug, Clone)]
pub struct ContextMenu {
    /// Stack of open menu levels (root + submenus).
    levels: Vec<MenuLevel>,
    /// Whether the menu is currently open.
    pub is_open: bool,
    /// Visual style.
    pub style: ContextMenuStyle,
    /// The selected action, if any.
    pending_action: Option<MenuActionId>,
    /// Screen/window size for clamping.
    screen_size: Vec2,
    /// Fade-in animation progress (0-1).
    pub fade_progress: f32,
    /// Fade speed.
    pub fade_speed: f32,
    /// Whether keyboard navigation is active.
    keyboard_active: bool,
}

impl ContextMenu {
    /// Create a new context menu (initially closed).
    pub fn new() -> Self {
        Self {
            levels: Vec::new(),
            is_open: false,
            style: ContextMenuStyle::default(),
            pending_action: None,
            screen_size: Vec2::new(1920.0, 1080.0),
            fade_progress: 0.0,
            fade_speed: 8.0,
            keyboard_active: false,
        }
    }

    /// Open the menu at the given screen position with the given items.
    pub fn open(&mut self, items: Vec<ContextMenuItem>, position: Vec2, screen_size: Vec2) {
        self.screen_size = screen_size;
        self.levels.clear();

        let clamped = self.clamp_position(position, &items);
        self.levels.push(MenuLevel::new(items, clamped));
        self.is_open = true;
        self.pending_action = None;
        self.fade_progress = 0.0;
        self.keyboard_active = false;
    }

    /// Close the menu.
    pub fn close(&mut self) {
        self.levels.clear();
        self.is_open = false;
        self.keyboard_active = false;
    }

    /// Take the pending action (if a menu item was selected).
    pub fn take_action(&mut self) -> Option<MenuActionId> {
        self.pending_action.take()
    }

    /// Whether an action is pending.
    pub fn has_pending_action(&self) -> bool {
        self.pending_action.is_some()
    }

    fn clamp_position(&self, pos: Vec2, items: &[ContextMenuItem]) -> Vec2 {
        let width = MenuLevel::compute_width(items);
        let height = items.iter().map(|i| i.height()).sum::<f32>() + MENU_PADDING * 2.0;

        let mut x = pos.x;
        let mut y = pos.y;

        if x + width > self.screen_size.x {
            x = (self.screen_size.x - width).max(0.0);
        }
        if y + height > self.screen_size.y {
            y = (self.screen_size.y - height).max(0.0);
        }

        Vec2::new(x, y)
    }

    /// Update animation (call each frame with delta time).
    pub fn update(&mut self, dt: f32) {
        if self.is_open && self.fade_progress < 1.0 {
            self.fade_progress = (self.fade_progress + self.fade_speed * dt).min(1.0);
        }

        // Update submenu hover timers.
        if let Some(level) = self.levels.last_mut() {
            if level.hovered.is_some() {
                level.hover_time += dt;
            }
        }

        // Open submenu if hovered long enough.
        self.try_open_submenu();
    }

    fn try_open_submenu(&mut self) {
        let depth = self.levels.len();
        if depth == 0 {
            return;
        }

        let level = &self.levels[depth - 1];
        if level.hover_time >= SUBMENU_DELAY && !level.submenu_open {
            if let Some(idx) = level.hovered {
                if let Some(ContextMenuItem::SubMenu { items, enabled, .. }) =
                    level.items.get(idx)
                {
                    if *enabled && !items.is_empty() {
                        let item_rect = level.item_rect(idx);
                        let sub_pos = Vec2::new(
                            level.position.x + level.size.x - 2.0,
                            item_rect.min.y,
                        );
                        let clamped = self.clamp_position(sub_pos, items);
                        let sub_items = items.clone();

                        // Mark the current level's submenu as open.
                        self.levels[depth - 1].submenu_open = true;

                        // Close any deeper submenus.
                        while self.levels.len() > depth {
                            self.levels.pop();
                        }

                        self.levels.push(MenuLevel::new(sub_items, clamped));
                    }
                }
            }
        }
    }

    /// Paint the context menu.
    pub fn paint(&self, draw: &mut DrawList) {
        if !self.is_open {
            return;
        }

        let alpha = self.fade_progress;

        for level in &self.levels {
            self.paint_level(level, alpha, draw);
        }
    }

    fn paint_level(&self, level: &MenuLevel, alpha: f32, draw: &mut DrawList) {
        let menu_rect = level.rect();

        // Shadow.
        draw.commands.push(DrawCommand::Rect {
            rect: Rect::new(
                menu_rect.min + Vec2::new(2.0, 2.0),
                menu_rect.max + Vec2::new(MENU_SHADOW_SIZE, MENU_SHADOW_SIZE),
            ),
            color: self.style.shadow_color.with_alpha(self.style.shadow_color.a * alpha * 0.5),
            corner_radii: CornerRadii::all(self.style.corner_radius),
            border: BorderSpec::default(),
            shadow: None,
        });

        // Background.
        draw.commands.push(DrawCommand::Rect {
            rect: menu_rect,
            color: self.style.background.with_alpha(alpha),
            corner_radii: CornerRadii::all(self.style.corner_radius),
            border: BorderSpec::new(
                self.style.border_color.with_alpha(alpha),
                self.style.border_width,
            ),
            shadow: None,
        });

        // Items.
        let mut y = level.position.y + MENU_PADDING;
        for (i, item) in level.items.iter().enumerate() {
            let h = item.height();
            let item_rect = Rect::new(
                Vec2::new(level.position.x + MENU_PADDING, y),
                Vec2::new(level.position.x + level.size.x - MENU_PADDING, y + h),
            );

            match item {
                ContextMenuItem::Action {
                    label,
                    shortcut,
                    icon,
                    enabled,
                    checked,
                    ..
                } => {
                    let is_hovered = level.hovered == Some(i);

                    // Hover highlight.
                    if is_hovered && *enabled {
                        draw.commands.push(DrawCommand::Rect {
                            rect: item_rect,
                            color: self.style.item_hover_color.with_alpha(alpha),
                            corner_radii: CornerRadii::all(2.0),
                            border: BorderSpec::default(),
                            shadow: None,
                        });
                    }

                    let text_color = if *enabled {
                        self.style.text_color
                    } else {
                        self.style.text_disabled_color
                    };

                    // Check mark.
                    let mut text_x = item_rect.min.x + 6.0;
                    if *checked {
                        draw.commands.push(DrawCommand::Text {
                            text: "\u{2713}".to_string(), // Checkmark
                            position: Vec2::new(text_x, item_rect.min.y + 4.0),
                            font_size: self.style.font_size,
                            color: self.style.check_color.with_alpha(alpha),
                            font_id: self.style.font_id,
                            max_width: None,
                            align: TextAlign::Left,
                            vertical_align: TextVerticalAlign::Top,
                        });
                    }
                    text_x += ICON_SIZE + ICON_PADDING;

                    // Icon.
                    if let Some(tex) = icon {
                        draw.commands.push(DrawCommand::Image {
                            rect: Rect::new(
                                Vec2::new(text_x, item_rect.min.y + (h - ICON_SIZE) * 0.5),
                                Vec2::new(text_x + ICON_SIZE, item_rect.min.y + (h + ICON_SIZE) * 0.5),
                            ),
                            texture: *tex,
                            tint: Color::WHITE.with_alpha(alpha),
                            corner_radii: CornerRadii::ZERO,
                            scale_mode: crate::render_commands::ImageScaleMode::Fit,
                            uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
                        });
                        text_x += ICON_SIZE + ICON_PADDING;
                    }

                    // Label.
                    draw.commands.push(DrawCommand::Text {
                        text: label.clone(),
                        position: Vec2::new(text_x, item_rect.min.y + 5.0),
                        font_size: self.style.font_size,
                        color: text_color.with_alpha(alpha),
                        font_id: self.style.font_id,
                        max_width: None,
                        align: TextAlign::Left,
                        vertical_align: TextVerticalAlign::Top,
                    });

                    // Shortcut.
                    if let Some(sc) = shortcut {
                        draw.commands.push(DrawCommand::Text {
                            text: sc.clone(),
                            position: Vec2::new(
                                item_rect.max.x - 8.0,
                                item_rect.min.y + 5.0,
                            ),
                            font_size: self.style.font_size - 1.0,
                            color: self.style.shortcut_color.with_alpha(alpha),
                            font_id: self.style.font_id,
                            max_width: None,
                            align: TextAlign::Right,
                            vertical_align: TextVerticalAlign::Top,
                        });
                    }
                }

                ContextMenuItem::Separator => {
                    let sep_y = y + SEPARATOR_HEIGHT * 0.5;
                    draw.commands.push(DrawCommand::Line {
                        start: Vec2::new(item_rect.min.x + 4.0, sep_y),
                        end: Vec2::new(item_rect.max.x - 4.0, sep_y),
                        color: self.style.separator_color.with_alpha(alpha),
                        thickness: 1.0,
                    });
                }

                ContextMenuItem::SubMenu {
                    label, icon, enabled, ..
                } => {
                    let is_hovered = level.hovered == Some(i);

                    if is_hovered && *enabled {
                        draw.commands.push(DrawCommand::Rect {
                            rect: item_rect,
                            color: self.style.item_hover_color.with_alpha(alpha),
                            corner_radii: CornerRadii::all(2.0),
                            border: BorderSpec::default(),
                            shadow: None,
                        });
                    }

                    let text_color = if *enabled {
                        self.style.text_color
                    } else {
                        self.style.text_disabled_color
                    };

                    let mut text_x = item_rect.min.x + 6.0 + ICON_SIZE + ICON_PADDING;

                    if let Some(tex) = icon {
                        draw.commands.push(DrawCommand::Image {
                            rect: Rect::new(
                                Vec2::new(
                                    item_rect.min.x + 6.0,
                                    item_rect.min.y + (h - ICON_SIZE) * 0.5,
                                ),
                                Vec2::new(
                                    item_rect.min.x + 6.0 + ICON_SIZE,
                                    item_rect.min.y + (h + ICON_SIZE) * 0.5,
                                ),
                            ),
                            texture: *tex,
                            tint: Color::WHITE.with_alpha(alpha),
                            corner_radii: CornerRadii::ZERO,
                            scale_mode: crate::render_commands::ImageScaleMode::Fit,
                            uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
                        });
                    }

                    // Label.
                    draw.commands.push(DrawCommand::Text {
                        text: label.clone(),
                        position: Vec2::new(text_x, item_rect.min.y + 5.0),
                        font_size: self.style.font_size,
                        color: text_color.with_alpha(alpha),
                        font_id: self.style.font_id,
                        max_width: None,
                        align: TextAlign::Left,
                        vertical_align: TextVerticalAlign::Top,
                    });

                    // Submenu arrow.
                    let arrow_x = item_rect.max.x - 12.0;
                    let arrow_y = item_rect.min.y + h * 0.5;
                    draw.commands.push(DrawCommand::Triangle {
                        p0: Vec2::new(arrow_x, arrow_y - 4.0),
                        p1: Vec2::new(arrow_x + 6.0, arrow_y),
                        p2: Vec2::new(arrow_x, arrow_y + 4.0),
                        color: text_color.with_alpha(alpha),
                    });
                }

                ContextMenuItem::Header { label } => {
                    draw.commands.push(DrawCommand::Text {
                        text: label.clone(),
                        position: Vec2::new(item_rect.min.x + 6.0, item_rect.min.y + 5.0),
                        font_size: self.style.font_size - 1.0,
                        color: self.style.header_color.with_alpha(alpha),
                        font_id: self.style.font_id,
                        max_width: None,
                        align: TextAlign::Left,
                        vertical_align: TextVerticalAlign::Top,
                    });
                }
            }

            y += h;
        }
    }

    /// Handle an event. Returns the event reply.
    pub fn handle_event(&mut self, event: &UIEvent) -> EventReply {
        if !self.is_open {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::MouseMove { x, y, .. } => {
                let pos = Vec2::new(*x, *y);
                self.handle_mouse_move(pos)
            }

            UIEvent::MouseDown { x, y, button, .. } => {
                let pos = Vec2::new(*x, *y);
                if *button == MouseButton::Left || *button == MouseButton::Right {
                    self.handle_mouse_click(pos)
                } else {
                    EventReply::Unhandled
                }
            }

            UIEvent::KeyDown { key, modifiers, .. } => self.handle_key(*key, *modifiers),

            _ => EventReply::Unhandled,
        }
    }

    fn handle_mouse_move(&mut self, pos: Vec2) -> EventReply {
        // Check each level from deepest to shallowest.
        let depth = self.levels.len();
        for d in (0..depth).rev() {
            let level = &self.levels[d];
            if let Some(idx) = level.item_at(pos) {
                // Close deeper submenus if hovering a different item.
                if self.levels[d].hovered != Some(idx) {
                    self.levels[d].hover_time = 0.0;
                    self.levels[d].submenu_open = false;
                    // Close levels deeper than d+1.
                    while self.levels.len() > d + 1 {
                        self.levels.pop();
                    }
                }
                self.levels[d].hovered = Some(idx);
                self.keyboard_active = false;
                return EventReply::Handled;
            }
        }

        // Not over any menu level -- but keep menu open.
        // Clear hovered on top level only if mouse is outside all levels.
        let over_any = self
            .levels
            .iter()
            .any(|l| l.rect().contains(pos));
        if !over_any {
            if let Some(top) = self.levels.last_mut() {
                // Don't clear hover if submenu is open.
                if !top.submenu_open {
                    top.hovered = None;
                }
            }
        }

        EventReply::Handled // Capture mouse while menu is open.
    }

    fn handle_mouse_click(&mut self, pos: Vec2) -> EventReply {
        // Check each level.
        for d in (0..self.levels.len()).rev() {
            let level = &self.levels[d];
            if let Some(idx) = level.item_at(pos) {
                match &level.items[idx] {
                    ContextMenuItem::Action { action, enabled, .. } => {
                        if *enabled {
                            self.pending_action = Some(action.clone());
                            self.close();
                            return EventReply::Handled;
                        }
                    }
                    ContextMenuItem::SubMenu { .. } => {
                        // Clicking a submenu item -- just let hover handle it.
                        return EventReply::Handled;
                    }
                    _ => {}
                }
                return EventReply::Handled;
            }
        }

        // Click outside all menu levels -> close.
        let over_any = self.levels.iter().any(|l| l.rect().contains(pos));
        if !over_any {
            self.close();
        }
        EventReply::Handled
    }

    fn handle_key(&mut self, key: KeyCode, _modifiers: KeyModifiers) -> EventReply {
        self.keyboard_active = true;

        let depth = self.levels.len();
        if depth == 0 {
            return EventReply::Unhandled;
        }

        match key {
            KeyCode::Escape => {
                if depth > 1 {
                    // Close the deepest submenu.
                    self.levels.pop();
                    if let Some(level) = self.levels.last_mut() {
                        level.submenu_open = false;
                    }
                } else {
                    self.close();
                }
                EventReply::Handled
            }

            KeyCode::ArrowUp => {
                self.move_selection(-1);
                EventReply::Handled
            }

            KeyCode::ArrowDown => {
                self.move_selection(1);
                EventReply::Handled
            }

            KeyCode::ArrowRight => {
                // Open submenu.
                if let Some(level) = self.levels.last() {
                    if let Some(idx) = level.hovered {
                        if matches!(level.items.get(idx), Some(ContextMenuItem::SubMenu { .. })) {
                            // Force open submenu immediately.
                            let last = self.levels.len() - 1;
                            self.levels[last].hover_time = SUBMENU_DELAY + 1.0;
                            self.try_open_submenu();
                            // Select first item in new submenu.
                            if let Some(new_level) = self.levels.last_mut() {
                                new_level.hovered = self.first_interactive_index(&new_level.items);
                            }
                        }
                    }
                }
                EventReply::Handled
            }

            KeyCode::ArrowLeft => {
                // Close submenu.
                if depth > 1 {
                    self.levels.pop();
                    if let Some(level) = self.levels.last_mut() {
                        level.submenu_open = false;
                    }
                }
                EventReply::Handled
            }

            KeyCode::Enter => {
                if let Some(level) = self.levels.last() {
                    if let Some(idx) = level.hovered {
                        match &level.items[idx] {
                            ContextMenuItem::Action { action, enabled, .. } => {
                                if *enabled {
                                    self.pending_action = Some(action.clone());
                                    self.close();
                                }
                            }
                            ContextMenuItem::SubMenu { .. } => {
                                // Open submenu.
                                let last = self.levels.len() - 1;
                                self.levels[last].hover_time = SUBMENU_DELAY + 1.0;
                                self.try_open_submenu();
                            }
                            _ => {}
                        }
                    }
                }
                EventReply::Handled
            }

            _ => EventReply::Handled, // Consume all keys while menu is open.
        }
    }

    fn move_selection(&mut self, direction: i32) {
        if let Some(level) = self.levels.last_mut() {
            let count = level.items.len();
            if count == 0 {
                return;
            }

            let start = level.hovered.unwrap_or(if direction > 0 {
                count.wrapping_sub(1)
            } else {
                0
            });

            let mut idx = start;
            for _ in 0..count {
                idx = if direction > 0 {
                    (idx + 1) % count
                } else {
                    (idx + count - 1) % count
                };
                if level.items[idx].is_interactive() {
                    level.hovered = Some(idx);
                    level.hover_time = 0.0;
                    level.submenu_open = false;
                    return;
                }
            }
        }
    }

    fn first_interactive_index(&self, items: &[ContextMenuItem]) -> Option<usize> {
        items.iter().position(|i| i.is_interactive())
    }
}

impl Default for ContextMenu {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// ContextMenuManager
// =========================================================================

/// Manages multiple context menus (e.g. one per widget region).
/// Ensures only one menu is open at a time.
#[derive(Debug, Clone)]
pub struct ContextMenuManager {
    /// The active context menu.
    pub active: ContextMenu,
    /// ID of the widget that opened the current menu.
    pub owner_id: UIId,
}

impl ContextMenuManager {
    pub fn new() -> Self {
        Self {
            active: ContextMenu::new(),
            owner_id: UIId::INVALID,
        }
    }

    /// Open a context menu for a specific owner.
    pub fn open(
        &mut self,
        owner: UIId,
        items: Vec<ContextMenuItem>,
        position: Vec2,
        screen_size: Vec2,
    ) {
        self.active.close();
        self.owner_id = owner;
        self.active.open(items, position, screen_size);
    }

    /// Close the active menu.
    pub fn close(&mut self) {
        self.active.close();
        self.owner_id = UIId::INVALID;
    }

    /// Whether a menu is open.
    pub fn is_open(&self) -> bool {
        self.active.is_open
    }

    /// Whether the menu is owned by the given widget.
    pub fn is_owned_by(&self, id: UIId) -> bool {
        self.owner_id == id
    }

    /// Update animations.
    pub fn update(&mut self, dt: f32) {
        self.active.update(dt);
    }

    /// Paint.
    pub fn paint(&self, draw: &mut DrawList) {
        self.active.paint(draw);
    }

    /// Handle event.
    pub fn handle_event(&mut self, event: &UIEvent) -> EventReply {
        self.active.handle_event(event)
    }

    /// Take the pending action.
    pub fn take_action(&mut self) -> Option<MenuActionId> {
        self.active.take_action()
    }
}

impl Default for ContextMenuManager {
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
    fn test_menu_item_height() {
        assert_eq!(ContextMenuItem::action("Test", 1).height(), MENU_ITEM_HEIGHT);
        assert_eq!(ContextMenuItem::separator().height(), SEPARATOR_HEIGHT);
    }

    #[test]
    fn test_menu_open_close() {
        let mut menu = ContextMenu::new();
        let items = vec![
            ContextMenuItem::action("Cut", 1),
            ContextMenuItem::action("Copy", 2),
            ContextMenuItem::separator(),
            ContextMenuItem::action("Paste", 3),
        ];
        menu.open(items, Vec2::new(100.0, 100.0), Vec2::new(1920.0, 1080.0));
        assert!(menu.is_open);
        menu.close();
        assert!(!menu.is_open);
    }

    #[test]
    fn test_menu_clamping() {
        let mut menu = ContextMenu::new();
        let items = vec![ContextMenuItem::action("Test", 1)];
        // Open near bottom-right.
        menu.open(items, Vec2::new(1900.0, 1060.0), Vec2::new(1920.0, 1080.0));
        assert!(menu.is_open);
        // Menu position should be clamped.
        let level = &menu.levels[0];
        assert!(level.position.x + level.size.x <= 1920.0);
        assert!(level.position.y + level.size.y <= 1080.0);
    }

    #[test]
    fn test_submenu() {
        let items = vec![
            ContextMenuItem::action("Action 1", 1),
            ContextMenuItem::submenu(
                "More",
                vec![
                    ContextMenuItem::action("Sub 1", 10),
                    ContextMenuItem::action("Sub 2", 11),
                ],
            ),
        ];

        let mut menu = ContextMenu::new();
        menu.open(items, Vec2::new(100.0, 100.0), Vec2::new(1920.0, 1080.0));
        assert_eq!(menu.levels.len(), 1);
    }
}
