//! Scrollbar widget for the Genovo Slate UI.
//!
//! Provides vertical and horizontal scrollbars with:
//! - Proportional thumb sizing
//! - Drag interaction on the thumb
//! - Click-on-track for page up/down
//! - Auto-hide with fade animation
//! - Spring-based smooth scrolling
//! - Minimum thumb size
//! - Mouse wheel integration
//! - Full visual styling (track, thumb, hover, drag states)

use glam::Vec2;
use genovo_core::Rect;

use crate::core::{MouseButton, Padding, UIEvent, UIId};
use crate::render_commands::{
    Border as BorderSpec, Color, CornerRadii, DrawCommand, DrawList,
};
use crate::slate_widgets::EventReply;

// =========================================================================
// Orientation
// =========================================================================

/// Scrollbar orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScrollOrientation {
    Vertical,
    Horizontal,
}

impl Default for ScrollOrientation {
    fn default() -> Self {
        ScrollOrientation::Vertical
    }
}

// =========================================================================
// ScrollbarVisibility
// =========================================================================

/// Controls when a scrollbar is visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollbarVisibility {
    /// Always visible.
    AlwaysVisible,
    /// Visible only when content exceeds viewport.
    Auto,
    /// Visible only when scrolling or hovered, then fades out.
    AutoHide,
    /// Never visible (still functional for mouse wheel).
    Hidden,
}

impl Default for ScrollbarVisibility {
    fn default() -> Self {
        ScrollbarVisibility::Auto
    }
}

// =========================================================================
// ScrollbarStyle
// =========================================================================

/// Visual styling for a scrollbar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScrollbarAppearance {
    /// Track background colour (normal).
    pub track_color: Color,
    /// Track colour when hovered.
    pub track_hover_color: Color,
    /// Thumb colour (normal).
    pub thumb_color: Color,
    /// Thumb colour when hovered.
    pub thumb_hover_color: Color,
    /// Thumb colour when being dragged.
    pub thumb_drag_color: Color,
    /// Thumb corner radius.
    pub thumb_radius: f32,
    /// Track corner radius.
    pub track_radius: f32,
    /// Width/height of the scrollbar (thickness).
    pub thickness: f32,
    /// Minimum thumb length in pixels.
    pub min_thumb_size: f32,
    /// Inset from edges.
    pub margin: f32,
    /// Track border.
    pub track_border: BorderSpec,
    /// Thumb border.
    pub thumb_border: BorderSpec,
}

impl Default for ScrollbarAppearance {
    fn default() -> Self {
        Self {
            track_color: Color::new(0.12, 0.12, 0.12, 0.6),
            track_hover_color: Color::new(0.15, 0.15, 0.15, 0.8),
            thumb_color: Color::new(0.45, 0.45, 0.45, 0.7),
            thumb_hover_color: Color::new(0.55, 0.55, 0.55, 0.9),
            thumb_drag_color: Color::new(0.65, 0.65, 0.65, 1.0),
            thumb_radius: 4.0,
            track_radius: 4.0,
            thickness: 10.0,
            min_thumb_size: 24.0,
            margin: 2.0,
            track_border: BorderSpec::default(),
            thumb_border: BorderSpec::default(),
        }
    }
}

impl ScrollbarAppearance {
    /// Thin style (6px, for overlay scrollbars).
    pub fn thin() -> Self {
        Self {
            thickness: 6.0,
            min_thumb_size: 18.0,
            thumb_radius: 3.0,
            track_radius: 3.0,
            track_color: Color::TRANSPARENT,
            ..Default::default()
        }
    }

    /// Wide style (14px, for editor panels).
    pub fn wide() -> Self {
        Self {
            thickness: 14.0,
            min_thumb_size: 28.0,
            ..Default::default()
        }
    }
}

// =========================================================================
// SpringAnimation
// =========================================================================

/// Spring-based smooth scroll animation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScrollSpring {
    /// Current scroll offset.
    pub current: f32,
    /// Target scroll offset.
    pub target: f32,
    /// Current velocity.
    pub velocity: f32,
    /// Stiffness (spring constant).
    pub stiffness: f32,
    /// Damping ratio.
    pub damping: f32,
    /// Mass.
    pub mass: f32,
    /// Whether the spring is actively animating.
    pub active: bool,
    /// Threshold for considering the spring at rest.
    pub rest_threshold: f32,
}

impl Default for ScrollSpring {
    fn default() -> Self {
        Self {
            current: 0.0,
            target: 0.0,
            velocity: 0.0,
            stiffness: 300.0,
            damping: 25.0,
            mass: 1.0,
            active: false,
            rest_threshold: 0.1,
        }
    }
}

impl ScrollSpring {
    pub fn new(stiffness: f32, damping: f32) -> Self {
        Self {
            stiffness,
            damping,
            ..Default::default()
        }
    }

    /// Set the target, activating the spring.
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
        self.active = true;
    }

    /// Immediately jump to target.
    pub fn snap(&mut self, value: f32) {
        self.current = value;
        self.target = value;
        self.velocity = 0.0;
        self.active = false;
    }

    /// Advance the spring by `dt` seconds.
    pub fn update(&mut self, dt: f32) -> bool {
        if !self.active {
            return false;
        }

        let displacement = self.current - self.target;
        let spring_force = -self.stiffness * displacement;
        let damping_force = -self.damping * self.velocity;
        let acceleration = (spring_force + damping_force) / self.mass;

        self.velocity += acceleration * dt;
        self.current += self.velocity * dt;

        // Check if at rest.
        if displacement.abs() < self.rest_threshold && self.velocity.abs() < self.rest_threshold {
            self.current = self.target;
            self.velocity = 0.0;
            self.active = false;
        }

        true
    }

    /// Add an impulse velocity (for mouse wheel flick).
    pub fn impulse(&mut self, amount: f32) {
        self.target += amount;
        self.active = true;
    }

    /// Whether the spring is currently animating.
    pub fn is_animating(&self) -> bool {
        self.active
    }
}

// =========================================================================
// AutoHide state
// =========================================================================

/// Auto-hide state for the scrollbar fade animation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoHideState {
    /// Current opacity (0.0 = invisible, 1.0 = fully visible).
    pub opacity: f32,
    /// Target opacity.
    pub target_opacity: f32,
    /// Time since last scroll activity (seconds).
    pub idle_time: f32,
    /// Delay before starting to fade out (seconds).
    pub fade_delay: f32,
    /// Duration of the fade animation (seconds).
    pub fade_duration: f32,
    /// Whether the cursor is over the scrollbar region.
    pub hovered: bool,
}

impl Default for AutoHideState {
    fn default() -> Self {
        Self {
            opacity: 0.0,
            target_opacity: 0.0,
            idle_time: 0.0,
            fade_delay: 1.0,
            fade_duration: 0.3,
            hovered: false,
        }
    }
}

impl AutoHideState {
    /// Called when scrolling happens.
    pub fn on_scroll_activity(&mut self) {
        self.opacity = 1.0;
        self.target_opacity = 1.0;
        self.idle_time = 0.0;
    }

    /// Update the auto-hide state.
    pub fn update(&mut self, dt: f32) {
        if self.hovered {
            self.opacity = 1.0;
            self.target_opacity = 1.0;
            self.idle_time = 0.0;
            return;
        }

        self.idle_time += dt;

        if self.idle_time > self.fade_delay {
            self.target_opacity = 0.0;
        }

        if (self.opacity - self.target_opacity).abs() > 0.001 {
            let speed = 1.0 / self.fade_duration.max(0.01);
            if self.target_opacity > self.opacity {
                self.opacity = (self.opacity + speed * dt).min(self.target_opacity);
            } else {
                self.opacity = (self.opacity - speed * dt).max(self.target_opacity);
            }
        }
    }

    /// Whether the scrollbar should be rendered.
    pub fn is_visible(&self) -> bool {
        self.opacity > 0.001
    }
}

// =========================================================================
// DragState
// =========================================================================

/// Internal state for thumb dragging.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DragState {
    /// Is the thumb currently being dragged?
    dragging: bool,
    /// Mouse position at drag start.
    drag_start_mouse: f32,
    /// Scroll offset at drag start.
    drag_start_offset: f32,
}

impl Default for DragState {
    fn default() -> Self {
        Self {
            dragging: false,
            drag_start_mouse: 0.0,
            drag_start_offset: 0.0,
        }
    }
}

// =========================================================================
// Scrollbar
// =========================================================================

/// A scrollbar widget that works with any scrollable container.
///
/// The scrollbar manages its own interaction state (hover, drag, auto-hide)
/// and emits scroll offset changes. The parent widget is responsible for
/// actually scrolling its content.
#[derive(Debug, Clone)]
pub struct Scrollbar {
    /// Unique widget ID.
    pub id: UIId,
    /// Orientation (vertical or horizontal).
    pub orientation: ScrollOrientation,
    /// Visual appearance.
    pub appearance: ScrollbarAppearance,
    /// Visibility mode.
    pub visibility: ScrollbarVisibility,

    // --- scroll state ---
    /// Current scroll offset (0.0 = top/left).
    pub scroll_offset: f32,
    /// Total content size (in the scrolling direction).
    pub content_size: f32,
    /// Viewport size (visible area in the scrolling direction).
    pub viewport_size: f32,

    // --- interaction state ---
    /// Whether the mouse is over the track.
    pub track_hovered: bool,
    /// Whether the mouse is over the thumb.
    pub thumb_hovered: bool,
    /// Drag state.
    drag: DragState,
    /// Spring animation for smooth scrolling.
    pub spring: ScrollSpring,
    /// Auto-hide animation state.
    pub auto_hide: AutoHideState,

    // --- output ---
    /// Whether the scroll offset changed this frame.
    pub scroll_changed: bool,
    /// The new scroll offset (read after event handling).
    pub new_scroll_offset: f32,

    /// Lines to scroll per mouse wheel click.
    pub wheel_scroll_amount: f32,
    /// Page scroll amount (fraction of viewport).
    pub page_scroll_fraction: f32,
    /// Whether this scrollbar is enabled.
    pub enabled: bool,
    /// Whether this scrollbar is visible.
    pub visible: bool,
}

impl Scrollbar {
    /// Create a new vertical scrollbar.
    pub fn new_vertical() -> Self {
        Self {
            id: UIId::INVALID,
            orientation: ScrollOrientation::Vertical,
            appearance: ScrollbarAppearance::default(),
            visibility: ScrollbarVisibility::Auto,
            scroll_offset: 0.0,
            content_size: 0.0,
            viewport_size: 0.0,
            track_hovered: false,
            thumb_hovered: false,
            drag: DragState::default(),
            spring: ScrollSpring::default(),
            auto_hide: AutoHideState::default(),
            scroll_changed: false,
            new_scroll_offset: 0.0,
            wheel_scroll_amount: 48.0,
            page_scroll_fraction: 0.9,
            enabled: true,
            visible: true,
        }
    }

    /// Create a new horizontal scrollbar.
    pub fn new_horizontal() -> Self {
        let mut sb = Self::new_vertical();
        sb.orientation = ScrollOrientation::Horizontal;
        sb
    }

    /// Builder: set appearance.
    pub fn with_appearance(mut self, appearance: ScrollbarAppearance) -> Self {
        self.appearance = appearance;
        self
    }

    /// Builder: set visibility mode.
    pub fn with_visibility(mut self, vis: ScrollbarVisibility) -> Self {
        self.visibility = vis;
        self
    }

    /// Builder: set auto-hide fade delay.
    pub fn with_fade_delay(mut self, delay: f32) -> Self {
        self.auto_hide.fade_delay = delay;
        self
    }

    /// Builder: set spring parameters.
    pub fn with_spring(mut self, stiffness: f32, damping: f32) -> Self {
        self.spring.stiffness = stiffness;
        self.spring.damping = damping;
        self
    }

    /// Maximum scroll offset.
    pub fn max_scroll(&self) -> f32 {
        (self.content_size - self.viewport_size).max(0.0)
    }

    /// Whether scrolling is needed (content exceeds viewport).
    pub fn needs_scrolling(&self) -> bool {
        self.content_size > self.viewport_size
    }

    /// Thumb size as a fraction of the track.
    pub fn thumb_fraction(&self) -> f32 {
        if self.content_size <= 0.0 {
            return 1.0;
        }
        (self.viewport_size / self.content_size).clamp(0.0, 1.0)
    }

    /// Set the scroll state from the parent container.
    pub fn set_scroll_state(&mut self, offset: f32, content_size: f32, viewport_size: f32) {
        self.scroll_offset = offset;
        self.content_size = content_size;
        self.viewport_size = viewport_size;
        self.spring.current = offset;
        self.spring.target = offset;
    }

    /// Scroll to a specific offset (animated).
    pub fn scroll_to(&mut self, offset: f32) {
        let clamped = offset.clamp(0.0, self.max_scroll());
        self.spring.set_target(clamped);
        self.auto_hide.on_scroll_activity();
    }

    /// Scroll to a specific offset (instant, no animation).
    pub fn scroll_to_instant(&mut self, offset: f32) {
        let clamped = offset.clamp(0.0, self.max_scroll());
        self.spring.snap(clamped);
        self.scroll_offset = clamped;
        self.new_scroll_offset = clamped;
        self.scroll_changed = true;
    }

    /// Scroll by a delta (animated).
    pub fn scroll_by(&mut self, delta: f32) {
        let new_target = (self.spring.target + delta).clamp(0.0, self.max_scroll());
        self.spring.set_target(new_target);
        self.auto_hide.on_scroll_activity();
    }

    /// Page up.
    pub fn page_up(&mut self) {
        self.scroll_by(-self.viewport_size * self.page_scroll_fraction);
    }

    /// Page down.
    pub fn page_down(&mut self) {
        self.scroll_by(self.viewport_size * self.page_scroll_fraction);
    }

    /// Scroll to the top/left.
    pub fn scroll_to_start(&mut self) {
        self.scroll_to(0.0);
    }

    /// Scroll to the bottom/right.
    pub fn scroll_to_end(&mut self) {
        self.scroll_to(self.max_scroll());
    }

    /// Compute the track rectangle given the scrollbar's allocated rect.
    fn track_rect(&self, rect: Rect) -> Rect {
        let m = self.appearance.margin;
        Rect::new(
            Vec2::new(rect.min.x + m, rect.min.y + m),
            Vec2::new(rect.max.x - m, rect.max.y - m),
        )
    }

    /// Compute the thumb rectangle.
    fn thumb_rect(&self, rect: Rect) -> Rect {
        let track = self.track_rect(rect);
        let track_len = match self.orientation {
            ScrollOrientation::Vertical => track.height(),
            ScrollOrientation::Horizontal => track.width(),
        };

        let thumb_len = (track_len * self.thumb_fraction())
            .max(self.appearance.min_thumb_size)
            .min(track_len);
        let scroll_range = track_len - thumb_len;
        let max_scroll = self.max_scroll();
        let scroll_ratio = if max_scroll > 0.0 {
            self.scroll_offset / max_scroll
        } else {
            0.0
        };
        let thumb_offset = scroll_ratio * scroll_range;

        match self.orientation {
            ScrollOrientation::Vertical => Rect::new(
                Vec2::new(track.min.x, track.min.y + thumb_offset),
                Vec2::new(track.max.x, track.min.y + thumb_offset + thumb_len),
            ),
            ScrollOrientation::Horizontal => Rect::new(
                Vec2::new(track.min.x + thumb_offset, track.min.y),
                Vec2::new(track.min.x + thumb_offset + thumb_len, track.max.y),
            ),
        }
    }

    /// Whether the scrollbar should be rendered.
    fn should_render(&self) -> bool {
        if !self.visible || !self.enabled {
            return false;
        }
        match self.visibility {
            ScrollbarVisibility::AlwaysVisible => true,
            ScrollbarVisibility::Auto => self.needs_scrolling(),
            ScrollbarVisibility::AutoHide => self.needs_scrolling() && self.auto_hide.is_visible(),
            ScrollbarVisibility::Hidden => false,
        }
    }

    /// Update animation state. Call once per frame with delta time.
    pub fn update(&mut self, dt: f32) {
        // Update spring.
        if self.spring.update(dt) {
            self.scroll_offset = self.spring.current.clamp(0.0, self.max_scroll());
            self.new_scroll_offset = self.scroll_offset;
            self.scroll_changed = true;
        }

        // Update auto-hide.
        if self.visibility == ScrollbarVisibility::AutoHide {
            self.auto_hide.update(dt);
        }
    }

    /// Compute the desired size (thickness in the cross-axis direction).
    pub fn compute_desired_size(&self) -> Vec2 {
        let t = self.appearance.thickness + self.appearance.margin * 2.0;
        match self.orientation {
            ScrollOrientation::Vertical => Vec2::new(t, 0.0),
            ScrollOrientation::Horizontal => Vec2::new(0.0, t),
        }
    }

    /// Paint the scrollbar.
    pub fn paint(&self, rect: Rect, draw: &mut DrawList) {
        if !self.should_render() {
            return;
        }

        let opacity = match self.visibility {
            ScrollbarVisibility::AutoHide => self.auto_hide.opacity,
            _ => 1.0,
        };

        // Track.
        let track = self.track_rect(rect);
        let track_color = if self.track_hovered {
            self.appearance.track_hover_color
        } else {
            self.appearance.track_color
        };
        draw.commands.push(DrawCommand::Rect {
            rect: track,
            color: track_color.with_alpha(track_color.a * opacity),
            corner_radii: CornerRadii::all(self.appearance.track_radius),
            border: self.appearance.track_border,
            shadow: None,
        });

        // Thumb.
        let thumb = self.thumb_rect(rect);
        let thumb_color = if self.drag.dragging {
            self.appearance.thumb_drag_color
        } else if self.thumb_hovered {
            self.appearance.thumb_hover_color
        } else {
            self.appearance.thumb_color
        };
        draw.commands.push(DrawCommand::Rect {
            rect: thumb,
            color: thumb_color.with_alpha(thumb_color.a * opacity),
            corner_radii: CornerRadii::all(self.appearance.thumb_radius),
            border: self.appearance.thumb_border,
            shadow: None,
        });
    }

    /// Handle an event. Returns `EventReply` indicating consumption.
    pub fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled || !self.visible {
            return EventReply::Unhandled;
        }

        self.scroll_changed = false;

        match event {
            UIEvent::Hover { position } => {
                let pos = *position;
                let track = self.track_rect(rect);
                let thumb = self.thumb_rect(rect);

                let was_track_hovered = self.track_hovered;
                let was_thumb_hovered = self.thumb_hovered;

                self.track_hovered = track.contains(pos);
                self.thumb_hovered = thumb.contains(pos);

                self.auto_hide.hovered = self.track_hovered;

                if self.track_hovered != was_track_hovered
                    || self.thumb_hovered != was_thumb_hovered
                {
                    return EventReply::Handled;
                }
            }

            UIEvent::DragMove { position, .. } => {
                if self.drag.dragging {
                    let track = self.track_rect(rect);
                    let mouse_pos = match self.orientation {
                        ScrollOrientation::Vertical => position.y,
                        ScrollOrientation::Horizontal => position.x,
                    };
                    let delta_mouse = mouse_pos - self.drag.drag_start_mouse;

                    let track_len = match self.orientation {
                        ScrollOrientation::Vertical => track.height(),
                        ScrollOrientation::Horizontal => track.width(),
                    };
                    let thumb_len = (track_len * self.thumb_fraction())
                        .max(self.appearance.min_thumb_size)
                        .min(track_len);
                    let scroll_range = track_len - thumb_len;

                    if scroll_range > 0.0 {
                        let scroll_delta = (delta_mouse / scroll_range) * self.max_scroll();
                        let new_offset =
                            (self.drag.drag_start_offset + scroll_delta).clamp(0.0, self.max_scroll());
                        self.scroll_offset = new_offset;
                        self.spring.snap(new_offset);
                        self.new_scroll_offset = new_offset;
                        self.scroll_changed = true;
                        self.auto_hide.on_scroll_activity();
                    }

                    return EventReply::Handled;
                }
            }

            UIEvent::Click { position, button, .. } => {
                if *button != MouseButton::Left {
                    return EventReply::Unhandled;
                }
                let pos = *position;
                let track = self.track_rect(rect);
                let thumb = self.thumb_rect(rect);

                if thumb.contains(pos) {
                    // Start dragging the thumb.
                    self.drag.dragging = true;
                    self.drag.drag_start_mouse = match self.orientation {
                        ScrollOrientation::Vertical => pos.y,
                        ScrollOrientation::Horizontal => pos.x,
                    };
                    self.drag.drag_start_offset = self.scroll_offset;
                    return EventReply::CaptureMouse;
                } else if track.contains(pos) {
                    // Click on track -> page up/down.
                    let click_pos = match self.orientation {
                        ScrollOrientation::Vertical => pos.y - track.min.y,
                        ScrollOrientation::Horizontal => pos.x - track.min.x,
                    };
                    let thumb_center = {
                        let tr = self.thumb_rect(rect);
                        match self.orientation {
                            ScrollOrientation::Vertical => {
                                (tr.min.y + tr.max.y) * 0.5 - track.min.y
                            }
                            ScrollOrientation::Horizontal => {
                                (tr.min.x + tr.max.x) * 0.5 - track.min.x
                            }
                        }
                    };

                    if click_pos < thumb_center {
                        self.page_up();
                    } else {
                        self.page_down();
                    }
                    return EventReply::Handled;
                }
            }

            UIEvent::MouseUp { button, .. } => {
                if *button == MouseButton::Left && self.drag.dragging {
                    self.drag.dragging = false;
                    return EventReply::ReleaseMouse;
                }
            }

            UIEvent::Scroll { delta, .. } => {
                let scroll_amount = -delta.y * self.wheel_scroll_amount;
                self.scroll_by(scroll_amount);
                return EventReply::Handled;
            }

            _ => {}
        }

        EventReply::Unhandled
    }
}

impl Default for Scrollbar {
    fn default() -> Self {
        Self::new_vertical()
    }
}

// =========================================================================
// ScrollRegion
// =========================================================================

/// A scroll region that pairs two scrollbars (vertical + horizontal) with a
/// content area. This is the high-level widget for creating scrollable containers.
#[derive(Debug, Clone)]
pub struct ScrollRegion {
    /// Unique widget ID.
    pub id: UIId,
    /// Vertical scrollbar.
    pub vertical: Scrollbar,
    /// Horizontal scrollbar.
    pub horizontal: Scrollbar,
    /// Whether vertical scrolling is enabled.
    pub enable_vertical: bool,
    /// Whether horizontal scrolling is enabled.
    pub enable_horizontal: bool,
    /// Background colour.
    pub background: Color,
    /// Corner radius.
    pub corner_radii: CornerRadii,
    /// Content size (set by the user).
    pub content_size: Vec2,
    /// Whether to show the gutter square where scrollbars meet.
    pub show_corner_gutter: bool,
    /// Gutter colour.
    pub gutter_color: Color,
}

impl ScrollRegion {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            vertical: Scrollbar::new_vertical(),
            horizontal: Scrollbar::new_horizontal(),
            enable_vertical: true,
            enable_horizontal: false,
            background: Color::TRANSPARENT,
            corner_radii: CornerRadii::ZERO,
            content_size: Vec2::ZERO,
            show_corner_gutter: true,
            gutter_color: Color::new(0.12, 0.12, 0.12, 1.0),
        }
    }

    /// Set content size and update scrollbar states.
    pub fn set_content_size(&mut self, content: Vec2, viewport: Vec2) {
        self.content_size = content;
        self.vertical
            .set_scroll_state(self.vertical.scroll_offset, content.y, viewport.y);
        self.horizontal
            .set_scroll_state(self.horizontal.scroll_offset, content.x, viewport.x);
    }

    /// Current scroll offset.
    pub fn scroll_offset(&self) -> Vec2 {
        Vec2::new(self.horizontal.scroll_offset, self.vertical.scroll_offset)
    }

    /// Compute sub-rectangles for content, vertical bar, horizontal bar, gutter.
    pub fn layout(&self, rect: Rect) -> ScrollRegionLayout {
        let v_thick = if self.enable_vertical && self.vertical.needs_scrolling() {
            self.vertical.appearance.thickness + self.vertical.appearance.margin * 2.0
        } else {
            0.0
        };
        let h_thick = if self.enable_horizontal && self.horizontal.needs_scrolling() {
            self.horizontal.appearance.thickness + self.horizontal.appearance.margin * 2.0
        } else {
            0.0
        };

        let content = Rect::new(
            rect.min,
            Vec2::new(rect.max.x - v_thick, rect.max.y - h_thick),
        );
        let v_bar = Rect::new(
            Vec2::new(rect.max.x - v_thick, rect.min.y),
            Vec2::new(rect.max.x, rect.max.y - h_thick),
        );
        let h_bar = Rect::new(
            Vec2::new(rect.min.x, rect.max.y - h_thick),
            Vec2::new(rect.max.x - v_thick, rect.max.y),
        );
        let gutter = Rect::new(
            Vec2::new(rect.max.x - v_thick, rect.max.y - h_thick),
            rect.max,
        );

        ScrollRegionLayout {
            content,
            vertical_bar: v_bar,
            horizontal_bar: h_bar,
            gutter,
        }
    }

    /// Update animations.
    pub fn update(&mut self, dt: f32) {
        self.vertical.update(dt);
        self.horizontal.update(dt);
    }

    /// Paint the scroll region chrome (scrollbars, gutter).
    pub fn paint(&self, rect: Rect, draw: &mut DrawList) {
        let layout = self.layout(rect);

        // Background.
        if self.background.a > 0.0 {
            draw.commands.push(DrawCommand::Rect {
                rect,
                color: self.background,
                corner_radii: self.corner_radii,
                border: BorderSpec::default(),
                shadow: None,
            });
        }

        // Scrollbars.
        if self.enable_vertical {
            self.vertical.paint(layout.vertical_bar, draw);
        }
        if self.enable_horizontal {
            self.horizontal.paint(layout.horizontal_bar, draw);
        }

        // Gutter.
        if self.show_corner_gutter
            && self.enable_vertical
            && self.enable_horizontal
            && self.vertical.needs_scrolling()
            && self.horizontal.needs_scrolling()
        {
            draw.commands.push(DrawCommand::Rect {
                rect: layout.gutter,
                color: self.gutter_color,
                corner_radii: CornerRadii::ZERO,
                border: BorderSpec::default(),
                shadow: None,
            });
        }
    }

    /// Handle events, dispatching to the appropriate scrollbar.
    pub fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        let layout = self.layout(rect);

        // Vertical scrollbar.
        if self.enable_vertical {
            let reply = self.vertical.handle_event(event, layout.vertical_bar);
            if reply.is_handled() {
                return reply;
            }
        }

        // Horizontal scrollbar.
        if self.enable_horizontal {
            let reply = self.horizontal.handle_event(event, layout.horizontal_bar);
            if reply.is_handled() {
                return reply;
            }
        }

        // Mouse wheel in content area.
        if let UIEvent::Scroll { delta, .. } = event {
            if self.enable_vertical {
                self.vertical.scroll_by(-delta.y * self.vertical.wheel_scroll_amount);
                return EventReply::Handled;
            }
            if self.enable_horizontal {
                self.horizontal.scroll_by(-delta.x * self.horizontal.wheel_scroll_amount);
                return EventReply::Handled;
            }
        }

        EventReply::Unhandled
    }
}

impl Default for ScrollRegion {
    fn default() -> Self {
        Self::new()
    }
}

/// Layout rectangles computed by `ScrollRegion::layout`.
#[derive(Debug, Clone, Copy)]
pub struct ScrollRegionLayout {
    pub content: Rect,
    pub vertical_bar: Rect,
    pub horizontal_bar: Rect,
    pub gutter: Rect,
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scrollbar_thumb_fraction() {
        let mut sb = Scrollbar::new_vertical();
        sb.content_size = 1000.0;
        sb.viewport_size = 250.0;
        assert!((sb.thumb_fraction() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_scrollbar_max_scroll() {
        let mut sb = Scrollbar::new_vertical();
        sb.content_size = 1000.0;
        sb.viewport_size = 250.0;
        assert_eq!(sb.max_scroll(), 750.0);
    }

    #[test]
    fn test_spring_update() {
        let mut spring = ScrollSpring::default();
        spring.set_target(100.0);
        for _ in 0..1000 {
            spring.update(1.0 / 60.0);
        }
        assert!((spring.current - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_auto_hide() {
        let mut ah = AutoHideState::default();
        ah.on_scroll_activity();
        assert!(ah.is_visible());

        // Simulate time passing beyond fade delay.
        for _ in 0..200 {
            ah.update(0.016);
        }
        assert!(!ah.is_visible());
    }

    #[test]
    fn test_scroll_region_layout() {
        let sr = ScrollRegion::new();
        let rect = Rect::new(Vec2::ZERO, Vec2::new(400.0, 300.0));
        let layout = sr.layout(rect);
        assert!(layout.content.width() > 0.0);
        assert!(layout.content.height() > 0.0);
    }
}
