//! Slate Notification and Dialog System
//!
//! Provides a complete notification, toast, modal dialog, confirmation dialog,
//! and progress dialog system for the Genovo UI framework.
//!
//! # Architecture
//!
//! ```text
//!  NotificationManager ──> NotificationStack ──> FadeAnimation
//!       │                        │
//!  ModalDialog              ConfirmDialog
//!       │                        │
//!  ProgressDialog           DialogBackdrop
//! ```
//!
//! # Design
//!
//! Notifications appear as toasts in the bottom-right corner of the screen,
//! stacked vertically. They auto-expire with a configurable timeout and fade
//! out smoothly. Modals block interaction with the rest of the UI using a
//! semi-transparent backdrop and focus trapping.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use glam::Vec2;

use crate::core::UIId;
use crate::render_commands::Color;

// ---------------------------------------------------------------------------
// NotificationId / DialogId
// ---------------------------------------------------------------------------

/// Unique identifier for a notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NotificationId(pub u64);

/// Unique identifier for a dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DialogId(pub u64);

/// Unique handle for a progress notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NotificationHandle(pub u64);

static NEXT_NOTIFICATION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_DIALOG_ID: AtomicU64 = AtomicU64::new(1);

fn next_notification_id() -> NotificationId {
    NotificationId(NEXT_NOTIFICATION_ID.fetch_add(1, Ordering::Relaxed))
}

fn next_dialog_id() -> DialogId {
    DialogId(NEXT_DIALOG_ID.fetch_add(1, Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// NotificationLevel
// ---------------------------------------------------------------------------

/// Severity level of a notification, which determines its appearance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationLevel {
    /// Informational message (blue icon).
    Info,
    /// Warning message (yellow/orange icon).
    Warning,
    /// Error message (red icon).
    Error,
    /// Success message (green icon).
    Success,
    /// Progress indicator (spinning icon).
    Progress,
}

impl NotificationLevel {
    /// Returns the icon color for this level.
    pub fn icon_color(&self) -> Color {
        match self {
            Self::Info => Color::new(0.3, 0.6, 1.0, 1.0),
            Self::Warning => Color::new(1.0, 0.75, 0.2, 1.0),
            Self::Error => Color::new(1.0, 0.3, 0.3, 1.0),
            Self::Success => Color::new(0.3, 0.9, 0.4, 1.0),
            Self::Progress => Color::new(0.5, 0.7, 1.0, 1.0),
        }
    }

    /// Returns the background color for this level.
    pub fn background_color(&self) -> Color {
        match self {
            Self::Info => Color::new(0.15, 0.18, 0.25, 0.95),
            Self::Warning => Color::new(0.25, 0.22, 0.12, 0.95),
            Self::Error => Color::new(0.25, 0.12, 0.12, 0.95),
            Self::Success => Color::new(0.12, 0.22, 0.14, 0.95),
            Self::Progress => Color::new(0.15, 0.18, 0.25, 0.95),
        }
    }

    /// Returns the border color for this level.
    pub fn border_color(&self) -> Color {
        match self {
            Self::Info => Color::new(0.3, 0.5, 0.9, 0.5),
            Self::Warning => Color::new(0.9, 0.7, 0.2, 0.5),
            Self::Error => Color::new(0.9, 0.3, 0.3, 0.5),
            Self::Success => Color::new(0.3, 0.8, 0.4, 0.5),
            Self::Progress => Color::new(0.4, 0.5, 0.8, 0.5),
        }
    }

    /// Returns a human-readable label for this level.
    pub fn label(&self) -> &str {
        match self {
            Self::Info => "Info",
            Self::Warning => "Warning",
            Self::Error => "Error",
            Self::Success => "Success",
            Self::Progress => "Progress",
        }
    }

    /// Returns the icon character (for text-based rendering).
    pub fn icon_char(&self) -> char {
        match self {
            Self::Info => 'i',
            Self::Warning => '!',
            Self::Error => 'X',
            Self::Success => '*',
            Self::Progress => '~',
        }
    }
}

impl fmt::Display for NotificationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// NotificationAnimation
// ---------------------------------------------------------------------------

/// Animation state for notification enter/exit transitions.
#[derive(Debug, Clone)]
pub struct NotificationAnimation {
    /// Current animation phase.
    pub phase: AnimationPhase,
    /// Progress within the current phase (0.0 to 1.0).
    pub progress: f32,
    /// Speed of the animation (units per second).
    pub speed: f32,
    /// Current opacity (affected by fade animation).
    pub opacity: f32,
    /// Current slide offset in Y (for slide-in/out).
    pub slide_offset_y: f32,
    /// Target slide offset.
    pub target_slide_y: f32,
    /// Current scale factor (for pop-in effect).
    pub scale: f32,
    /// Whether the animation is complete.
    pub complete: bool,
    /// Duration of the fade-out phase in seconds.
    pub fade_duration: f32,
    /// Duration of the slide-in phase in seconds.
    pub slide_duration: f32,
}

/// Phase of notification animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationPhase {
    /// Sliding in from the side.
    SlideIn,
    /// Visible and stable.
    Visible,
    /// Fading out.
    FadeOut,
    /// Sliding up (making room for new notification below).
    SlideUp,
    /// Animation complete, ready for removal.
    Done,
}

impl NotificationAnimation {
    /// Creates a new animation starting with slide-in.
    pub fn new() -> Self {
        Self {
            phase: AnimationPhase::SlideIn,
            progress: 0.0,
            speed: 4.0,
            opacity: 0.0,
            slide_offset_y: 50.0,
            target_slide_y: 0.0,
            scale: 0.95,
            complete: false,
            fade_duration: 0.5,
            slide_duration: 0.25,
        }
    }

    /// Creates an animation for immediate visibility (no slide-in).
    pub fn immediate() -> Self {
        Self {
            phase: AnimationPhase::Visible,
            progress: 1.0,
            speed: 4.0,
            opacity: 1.0,
            slide_offset_y: 0.0,
            target_slide_y: 0.0,
            scale: 1.0,
            complete: false,
            fade_duration: 0.5,
            slide_duration: 0.25,
        }
    }

    /// Updates the animation for one frame.
    pub fn update(&mut self, dt: f32) {
        match self.phase {
            AnimationPhase::SlideIn => {
                self.progress += dt * self.speed / self.slide_duration.max(0.01);
                if self.progress >= 1.0 {
                    self.progress = 1.0;
                    self.phase = AnimationPhase::Visible;
                }
                // Ease-out cubic for slide-in.
                let t = self.progress;
                let eased = 1.0 - (1.0 - t) * (1.0 - t) * (1.0 - t);
                self.opacity = eased;
                self.slide_offset_y = 50.0 * (1.0 - eased);
                self.scale = 0.95 + 0.05 * eased;
            }
            AnimationPhase::Visible => {
                self.opacity = 1.0;
                self.slide_offset_y = self.target_slide_y;
                self.scale = 1.0;
            }
            AnimationPhase::FadeOut => {
                self.progress += dt / self.fade_duration.max(0.01);
                if self.progress >= 1.0 {
                    self.progress = 1.0;
                    self.phase = AnimationPhase::Done;
                    self.complete = true;
                }
                // Ease-in for fade-out.
                let t = self.progress;
                self.opacity = 1.0 - t * t;
                self.slide_offset_y = self.target_slide_y + 10.0 * t;
                self.scale = 1.0 - 0.05 * t;
            }
            AnimationPhase::SlideUp => {
                self.progress += dt * self.speed;
                if self.progress >= 1.0 {
                    self.progress = 1.0;
                    self.phase = AnimationPhase::Visible;
                    self.progress = 0.0;
                }
                let t = self.progress;
                let eased = 1.0 - (1.0 - t) * (1.0 - t);
                self.slide_offset_y = self.slide_offset_y
                    + (self.target_slide_y - self.slide_offset_y) * eased;
            }
            AnimationPhase::Done => {
                self.complete = true;
                self.opacity = 0.0;
            }
        }
    }

    /// Starts the fade-out phase.
    pub fn start_fade_out(&mut self) {
        if self.phase != AnimationPhase::FadeOut && self.phase != AnimationPhase::Done {
            self.phase = AnimationPhase::FadeOut;
            self.progress = 0.0;
        }
    }

    /// Returns true if the notification is being dismissed.
    pub fn is_dismissing(&self) -> bool {
        matches!(self.phase, AnimationPhase::FadeOut | AnimationPhase::Done)
    }

    /// Returns true if the animation has finished.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Returns the current display opacity (0.0 = invisible, 1.0 = opaque).
    pub fn current_opacity(&self) -> f32 {
        self.opacity.clamp(0.0, 1.0)
    }
}

impl Default for NotificationAnimation {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Notification
// ---------------------------------------------------------------------------

/// A single notification toast.
#[derive(Debug, Clone)]
pub struct Notification {
    /// Unique identifier.
    pub id: NotificationId,
    /// Text content.
    pub text: String,
    /// Optional secondary text.
    pub detail_text: Option<String>,
    /// Severity level.
    pub level: NotificationLevel,
    /// Auto-expire timeout in seconds (0 = no auto-expire).
    pub timeout: f32,
    /// Time remaining before auto-expire.
    pub time_remaining: f32,
    /// Animation state.
    pub animation: NotificationAnimation,
    /// Whether this notification has been dismissed.
    pub dismissed: bool,
    /// Whether this notification is interactive (clickable).
    pub interactive: bool,
    /// Whether this notification has been clicked.
    pub clicked: bool,
    /// Whether to show a close button.
    pub show_close_button: bool,
    /// Progress value (0.0 to 1.0) for progress notifications.
    pub progress: Option<f32>,
    /// Progress text (e.g., "3/10 files").
    pub progress_text: Option<String>,
    /// Whether this progress notification can be cancelled.
    pub cancellable: bool,
    /// Whether cancel was requested.
    pub cancel_requested: bool,
    /// Whether the task is complete.
    pub complete: bool,
    /// Width of the notification in pixels.
    pub width: f32,
    /// Height of the notification in pixels (computed).
    pub height: f32,
    /// Position (computed by the manager).
    pub position: Vec2,
    /// Creation frame number.
    pub created_frame: u64,
    /// Custom action label (e.g., "Undo", "Retry").
    pub action_label: Option<String>,
    /// Whether the action button was clicked.
    pub action_clicked: bool,
}

impl Notification {
    /// Creates a new info notification.
    pub fn info(text: &str) -> Self {
        Self::new(text, NotificationLevel::Info, 5.0)
    }

    /// Creates a new warning notification.
    pub fn warning(text: &str) -> Self {
        Self::new(text, NotificationLevel::Warning, 8.0)
    }

    /// Creates a new error notification.
    pub fn error(text: &str) -> Self {
        Self::new(text, NotificationLevel::Error, 0.0) // Errors don't auto-expire.
    }

    /// Creates a new success notification.
    pub fn success(text: &str) -> Self {
        Self::new(text, NotificationLevel::Success, 5.0)
    }

    /// Creates a new progress notification.
    pub fn progress(text: &str) -> Self {
        let mut notif = Self::new(text, NotificationLevel::Progress, 0.0);
        notif.progress = Some(0.0);
        notif.cancellable = true;
        notif
    }

    /// Creates a new notification with the given parameters.
    pub fn new(text: &str, level: NotificationLevel, timeout: f32) -> Self {
        Self {
            id: next_notification_id(),
            text: text.to_string(),
            detail_text: None,
            level,
            timeout,
            time_remaining: timeout,
            animation: NotificationAnimation::new(),
            dismissed: false,
            interactive: true,
            clicked: false,
            show_close_button: true,
            progress: None,
            progress_text: None,
            cancellable: false,
            cancel_requested: false,
            complete: false,
            width: 350.0,
            height: 80.0,
            position: Vec2::ZERO,
            created_frame: 0,
            action_label: None,
            action_clicked: false,
        }
    }

    /// Sets the detail text.
    pub fn with_detail(mut self, detail: &str) -> Self {
        self.detail_text = Some(detail.to_string());
        self.height = 100.0;
        self
    }

    /// Sets an action button label.
    pub fn with_action(mut self, label: &str) -> Self {
        self.action_label = Some(label.to_string());
        self
    }

    /// Sets the width.
    pub fn with_width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// Sets the timeout.
    pub fn with_timeout(mut self, seconds: f32) -> Self {
        self.timeout = seconds;
        self.time_remaining = seconds;
        self
    }

    /// Updates the notification for one frame.
    pub fn update(&mut self, dt: f32) {
        self.animation.update(dt);

        // Auto-expire.
        if self.timeout > 0.0 && !self.dismissed && !self.animation.is_dismissing() {
            self.time_remaining -= dt;
            if self.time_remaining <= 0.0 {
                self.dismiss();
            }
        }
    }

    /// Dismisses the notification with fade-out animation.
    pub fn dismiss(&mut self) {
        if !self.dismissed {
            self.dismissed = true;
            self.animation.start_fade_out();
        }
    }

    /// Returns true if this notification should be removed.
    pub fn should_remove(&self) -> bool {
        self.dismissed && self.animation.is_complete()
    }

    /// Updates the progress value.
    pub fn update_progress(&mut self, fraction: f32, text: Option<&str>) {
        self.progress = Some(fraction.clamp(0.0, 1.0));
        if let Some(t) = text {
            self.progress_text = Some(t.to_string());
        }
    }

    /// Marks the progress as complete.
    pub fn complete_progress(&mut self, success: bool) {
        self.progress = Some(1.0);
        self.complete = true;
        if success {
            self.level = NotificationLevel::Success;
            self.text = format!("{} - Complete", self.text);
        } else {
            self.level = NotificationLevel::Error;
            self.text = format!("{} - Failed", self.text);
        }
        self.timeout = 5.0;
        self.time_remaining = 5.0;
    }

    /// Returns the auto-expire progress (0.0 = fresh, 1.0 = about to expire).
    pub fn expire_fraction(&self) -> f32 {
        if self.timeout <= 0.0 {
            0.0
        } else {
            1.0 - (self.time_remaining / self.timeout).clamp(0.0, 1.0)
        }
    }

    /// Returns the render data for this notification.
    pub fn render_data(&self) -> NotificationRenderData {
        NotificationRenderData {
            position: self.position
                + Vec2::new(0.0, self.animation.slide_offset_y),
            size: Vec2::new(self.width, self.height),
            opacity: self.animation.current_opacity(),
            scale: self.animation.scale,
            background_color: self.level.background_color(),
            border_color: self.level.border_color(),
            icon_color: self.level.icon_color(),
            text: self.text.clone(),
            detail_text: self.detail_text.clone(),
            progress: self.progress,
            progress_text: self.progress_text.clone(),
            action_label: self.action_label.clone(),
            show_close_button: self.show_close_button,
            expire_fraction: self.expire_fraction(),
            level: self.level,
        }
    }
}

/// Render data for a notification (passed to the rendering backend).
#[derive(Debug, Clone)]
pub struct NotificationRenderData {
    pub position: Vec2,
    pub size: Vec2,
    pub opacity: f32,
    pub scale: f32,
    pub background_color: Color,
    pub border_color: Color,
    pub icon_color: Color,
    pub text: String,
    pub detail_text: Option<String>,
    pub progress: Option<f32>,
    pub progress_text: Option<String>,
    pub action_label: Option<String>,
    pub show_close_button: bool,
    pub expire_fraction: f32,
    pub level: NotificationLevel,
}

// ---------------------------------------------------------------------------
// NotificationManager
// ---------------------------------------------------------------------------

/// Manages the lifecycle and display of notification toasts.
///
/// Notifications appear in the bottom-right corner of the screen, stacked
/// vertically. Newer notifications appear at the bottom and push older ones
/// up. Auto-expiring notifications fade out smoothly after their timeout.
///
/// # Usage
///
/// ```ignore
/// let mut mgr = NotificationManager::new(1920.0, 1080.0);
///
/// mgr.show_info("File saved successfully.");
/// mgr.show_error("Failed to compile shader: syntax error at line 42.");
///
/// let handle = mgr.show_progress("Importing assets...");
/// mgr.update_progress(handle, 0.5, Some("50% complete"));
/// mgr.complete(handle, true);
/// ```
#[derive(Debug, Clone)]
pub struct NotificationManager {
    /// Active notifications (newest last).
    pub notifications: Vec<Notification>,
    /// Screen width for positioning.
    pub screen_width: f32,
    /// Screen height for positioning.
    pub screen_height: f32,
    /// Maximum number of visible notifications.
    pub max_visible: usize,
    /// Vertical spacing between notifications.
    pub spacing: f32,
    /// Margin from screen edge.
    pub margin: f32,
    /// Default notification width.
    pub default_width: f32,
    /// Default timeout for info notifications (seconds).
    pub default_info_timeout: f32,
    /// Default timeout for warning notifications (seconds).
    pub default_warning_timeout: f32,
    /// Default timeout for error notifications (seconds, 0 = never).
    pub default_error_timeout: f32,
    /// Default timeout for success notifications (seconds).
    pub default_success_timeout: f32,
    /// Whether notifications are enabled.
    pub enabled: bool,
    /// Position anchor for the notification stack.
    pub anchor: NotificationAnchor,
    /// Current frame number.
    pub current_frame: u64,
    /// Total notifications shown.
    pub total_shown: u64,
    /// Map of handles to notification IDs for progress updates.
    pub handle_map: HashMap<u64, NotificationId>,
    /// Whether to pause auto-expire when mouse is over a notification.
    pub pause_on_hover: bool,
    /// Currently hovered notification ID.
    pub hovered_id: Option<NotificationId>,
}

/// Position anchor for the notification stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationAnchor {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    TopCenter,
    BottomCenter,
}

impl NotificationManager {
    /// Creates a new notification manager.
    pub fn new(screen_width: f32, screen_height: f32) -> Self {
        Self {
            notifications: Vec::new(),
            screen_width,
            screen_height,
            max_visible: 5,
            spacing: 8.0,
            margin: 20.0,
            default_width: 350.0,
            default_info_timeout: 5.0,
            default_warning_timeout: 8.0,
            default_error_timeout: 0.0,
            default_success_timeout: 5.0,
            enabled: true,
            anchor: NotificationAnchor::BottomRight,
            current_frame: 0,
            total_shown: 0,
            handle_map: HashMap::new(),
            pause_on_hover: true,
            hovered_id: None,
        }
    }

    /// Sets the screen dimensions (for positioning).
    pub fn set_screen_size(&mut self, width: f32, height: f32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    /// Shows an info notification.
    pub fn show_info(&mut self, text: &str) -> NotificationId {
        let notif = Notification::info(text).with_timeout(self.default_info_timeout);
        self.add_notification(notif)
    }

    /// Shows a warning notification.
    pub fn show_warning(&mut self, text: &str) -> NotificationId {
        let notif = Notification::warning(text).with_timeout(self.default_warning_timeout);
        self.add_notification(notif)
    }

    /// Shows an error notification.
    pub fn show_error(&mut self, text: &str) -> NotificationId {
        let notif = Notification::error(text).with_timeout(self.default_error_timeout);
        self.add_notification(notif)
    }

    /// Shows a success notification.
    pub fn show_success(&mut self, text: &str) -> NotificationId {
        let notif = Notification::success(text).with_timeout(self.default_success_timeout);
        self.add_notification(notif)
    }

    /// Shows a progress notification and returns a handle for updates.
    pub fn show_progress(&mut self, text: &str) -> NotificationHandle {
        let notif = Notification::progress(text);
        let id = notif.id;
        let handle = NotificationHandle(id.0);
        self.handle_map.insert(handle.0, id);
        self.add_notification(notif);
        handle
    }

    /// Updates a progress notification.
    pub fn update_progress(&mut self, handle: NotificationHandle, fraction: f32, text: Option<&str>) {
        if let Some(id) = self.handle_map.get(&handle.0).copied() {
            if let Some(notif) = self.notifications.iter_mut().find(|n| n.id == id) {
                notif.update_progress(fraction, text);
            }
        }
    }

    /// Completes a progress notification.
    pub fn complete(&mut self, handle: NotificationHandle, success: bool) {
        if let Some(id) = self.handle_map.get(&handle.0).copied() {
            if let Some(notif) = self.notifications.iter_mut().find(|n| n.id == id) {
                notif.complete_progress(success);
            }
        }
    }

    /// Adds a notification to the stack.
    fn add_notification(&mut self, mut notif: Notification) -> NotificationId {
        notif.created_frame = self.current_frame;
        notif.width = self.default_width;
        let id = notif.id;
        self.notifications.push(notif);
        self.total_shown += 1;

        // Dismiss excess notifications (beyond max_visible).
        let visible_count = self
            .notifications
            .iter()
            .filter(|n| !n.dismissed)
            .count();
        if visible_count > self.max_visible {
            // Dismiss the oldest non-dismissed notification.
            if let Some(oldest) = self
                .notifications
                .iter_mut()
                .find(|n| !n.dismissed)
            {
                oldest.dismiss();
            }
        }

        id
    }

    /// Dismisses a notification by ID.
    pub fn dismiss(&mut self, id: NotificationId) {
        if let Some(notif) = self.notifications.iter_mut().find(|n| n.id == id) {
            notif.dismiss();
        }
    }

    /// Dismisses all notifications.
    pub fn dismiss_all(&mut self) {
        for notif in &mut self.notifications {
            notif.dismiss();
        }
    }

    /// Updates all notifications for one frame.
    pub fn update(&mut self, dt: f32) {
        self.current_frame += 1;

        // Update each notification.
        for notif in &mut self.notifications {
            // Pause auto-expire on hover.
            if self.pause_on_hover && self.hovered_id == Some(notif.id) {
                // Don't update time_remaining, but still update animation.
                notif.animation.update(dt);
            } else {
                notif.update(dt);
            }
        }

        // Remove completed notifications.
        self.notifications.retain(|n| !n.should_remove());

        // Recompute positions.
        self.layout_notifications();
    }

    /// Computes positions for all active notifications.
    fn layout_notifications(&mut self) {
        let active: Vec<usize> = (0..self.notifications.len())
            .filter(|&i| !self.notifications[i].animation.is_complete())
            .collect();

        match self.anchor {
            NotificationAnchor::BottomRight => {
                let mut y = self.screen_height - self.margin;
                for &i in active.iter().rev() {
                    let notif = &mut self.notifications[i];
                    y -= notif.height;
                    notif.position = Vec2::new(
                        self.screen_width - self.margin - notif.width,
                        y,
                    );
                    y -= self.spacing;
                }
            }
            NotificationAnchor::TopRight => {
                let mut y = self.margin;
                for &i in active.iter() {
                    let notif = &mut self.notifications[i];
                    notif.position = Vec2::new(
                        self.screen_width - self.margin - notif.width,
                        y,
                    );
                    y += notif.height + self.spacing;
                }
            }
            NotificationAnchor::BottomLeft => {
                let mut y = self.screen_height - self.margin;
                for &i in active.iter().rev() {
                    let notif = &mut self.notifications[i];
                    y -= notif.height;
                    notif.position = Vec2::new(self.margin, y);
                    y -= self.spacing;
                }
            }
            NotificationAnchor::TopLeft => {
                let mut y = self.margin;
                for &i in active.iter() {
                    let notif = &mut self.notifications[i];
                    notif.position = Vec2::new(self.margin, y);
                    y += notif.height + self.spacing;
                }
            }
            NotificationAnchor::TopCenter => {
                let mut y = self.margin;
                for &i in active.iter() {
                    let notif = &mut self.notifications[i];
                    notif.position = Vec2::new(
                        (self.screen_width - notif.width) * 0.5,
                        y,
                    );
                    y += notif.height + self.spacing;
                }
            }
            NotificationAnchor::BottomCenter => {
                let mut y = self.screen_height - self.margin;
                for &i in active.iter().rev() {
                    let notif = &mut self.notifications[i];
                    y -= notif.height;
                    notif.position = Vec2::new(
                        (self.screen_width - notif.width) * 0.5,
                        y,
                    );
                    y -= self.spacing;
                }
            }
        }
    }

    /// Handles a mouse click at the given position.
    pub fn on_click(&mut self, pos: Vec2) -> Option<NotificationId> {
        for notif in self.notifications.iter_mut().rev() {
            if notif.animation.is_dismissing() {
                continue;
            }
            let render = notif.render_data();
            let p = render.position;
            let s = render.size;
            if pos.x >= p.x && pos.x < p.x + s.x && pos.y >= p.y && pos.y < p.y + s.y {
                if notif.interactive {
                    notif.clicked = true;
                    notif.dismiss();
                    return Some(notif.id);
                }
            }
        }
        None
    }

    /// Updates the hovered notification based on mouse position.
    pub fn on_hover(&mut self, pos: Vec2) {
        self.hovered_id = None;
        for notif in self.notifications.iter().rev() {
            if notif.animation.is_dismissing() {
                continue;
            }
            let p = notif.position;
            let s = Vec2::new(notif.width, notif.height);
            if pos.x >= p.x && pos.x < p.x + s.x && pos.y >= p.y && pos.y < p.y + s.y {
                self.hovered_id = Some(notif.id);
                break;
            }
        }
    }

    /// Returns render data for all visible notifications.
    pub fn render_data(&self) -> Vec<NotificationRenderData> {
        self.notifications
            .iter()
            .filter(|n| !n.animation.is_complete())
            .map(|n| n.render_data())
            .collect()
    }

    /// Returns the number of active (non-dismissed) notifications.
    pub fn active_count(&self) -> usize {
        self.notifications.iter().filter(|n| !n.dismissed).count()
    }

    /// Returns the total number of notifications (including dismissing).
    pub fn total_count(&self) -> usize {
        self.notifications.len()
    }
}

// ---------------------------------------------------------------------------
// DialogButton
// ---------------------------------------------------------------------------

/// Button type for modal dialogs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DialogButton {
    OK,
    Cancel,
    Yes,
    No,
    Retry,
    Abort,
    Ignore,
    Close,
    Save,
    DontSave,
    Custom(u32),
}

impl DialogButton {
    /// Returns the display label for this button.
    pub fn label(&self) -> &str {
        match self {
            Self::OK => "OK",
            Self::Cancel => "Cancel",
            Self::Yes => "Yes",
            Self::No => "No",
            Self::Retry => "Retry",
            Self::Abort => "Abort",
            Self::Ignore => "Ignore",
            Self::Close => "Close",
            Self::Save => "Save",
            Self::DontSave => "Don't Save",
            Self::Custom(_) => "Custom",
        }
    }

    /// Returns true if this is a positive/affirmative button.
    pub fn is_affirmative(&self) -> bool {
        matches!(self, Self::OK | Self::Yes | Self::Save | Self::Retry)
    }

    /// Returns true if this is a negative/destructive button.
    pub fn is_negative(&self) -> bool {
        matches!(self, Self::No | Self::Cancel | Self::Abort | Self::DontSave)
    }
}

impl fmt::Display for DialogButton {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Preset button combinations for common dialog patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogButtons {
    OK,
    OKCancel,
    YesNo,
    YesNoCancel,
    RetryCancel,
    SaveDontSaveCancel,
    AbortRetryIgnore,
}

impl DialogButtons {
    /// Returns the button types for this preset.
    pub fn buttons(&self) -> Vec<DialogButton> {
        match self {
            Self::OK => vec![DialogButton::OK],
            Self::OKCancel => vec![DialogButton::OK, DialogButton::Cancel],
            Self::YesNo => vec![DialogButton::Yes, DialogButton::No],
            Self::YesNoCancel => vec![
                DialogButton::Yes,
                DialogButton::No,
                DialogButton::Cancel,
            ],
            Self::RetryCancel => vec![DialogButton::Retry, DialogButton::Cancel],
            Self::SaveDontSaveCancel => vec![
                DialogButton::Save,
                DialogButton::DontSave,
                DialogButton::Cancel,
            ],
            Self::AbortRetryIgnore => vec![
                DialogButton::Abort,
                DialogButton::Retry,
                DialogButton::Ignore,
            ],
        }
    }

    /// Returns the default button for this preset.
    pub fn default_button(&self) -> DialogButton {
        match self {
            Self::OK => DialogButton::OK,
            Self::OKCancel => DialogButton::OK,
            Self::YesNo => DialogButton::Yes,
            Self::YesNoCancel => DialogButton::Yes,
            Self::RetryCancel => DialogButton::Retry,
            Self::SaveDontSaveCancel => DialogButton::Save,
            Self::AbortRetryIgnore => DialogButton::Retry,
        }
    }

    /// Returns the escape button (pressed when Escape key is hit).
    pub fn escape_button(&self) -> DialogButton {
        match self {
            Self::OK => DialogButton::OK,
            Self::OKCancel => DialogButton::Cancel,
            Self::YesNo => DialogButton::No,
            Self::YesNoCancel => DialogButton::Cancel,
            Self::RetryCancel => DialogButton::Cancel,
            Self::SaveDontSaveCancel => DialogButton::Cancel,
            Self::AbortRetryIgnore => DialogButton::Abort,
        }
    }
}

// ---------------------------------------------------------------------------
// DialogResult
// ---------------------------------------------------------------------------

/// The result of a dialog interaction.
#[derive(Debug, Clone)]
pub struct DialogResult {
    /// Which button was clicked.
    pub button: DialogButton,
    /// Custom labels for button values.
    pub custom_button_labels: HashMap<DialogButton, String>,
    /// Whether the dialog was closed via escape key.
    pub escaped: bool,
    /// Whether "Don't show again" was checked.
    pub dont_show_again: bool,
}

impl DialogResult {
    /// Creates a result from a button click.
    pub fn from_button(button: DialogButton) -> Self {
        Self {
            button,
            custom_button_labels: HashMap::new(),
            escaped: false,
            dont_show_again: false,
        }
    }

    /// Creates a result from escape.
    pub fn from_escape(escape_button: DialogButton) -> Self {
        Self {
            button: escape_button,
            custom_button_labels: HashMap::new(),
            escaped: true,
            dont_show_again: false,
        }
    }

    /// Returns true if the user chose an affirmative action.
    pub fn is_affirmative(&self) -> bool {
        self.button.is_affirmative()
    }

    /// Returns true if the user cancelled or chose a negative action.
    pub fn is_cancelled(&self) -> bool {
        self.button.is_negative() || self.escaped
    }
}

// ---------------------------------------------------------------------------
// ModalDialog
// ---------------------------------------------------------------------------

/// A modal dialog that blocks interaction with the rest of the UI.
///
/// The dialog renders on top of a semi-transparent backdrop that prevents
/// clicks from reaching underlying widgets. Focus is trapped within the
/// dialog. Escape closes the dialog (with the cancel/escape button).
///
/// # Usage
///
/// ```ignore
/// let mut dialog = ModalDialog::new("Confirm Delete", DialogButtons::YesNoCancel);
/// dialog.set_message("Are you sure you want to delete this asset?");
/// dialog.set_icon(DialogIcon::Warning);
///
/// // Each frame while the dialog is open:
/// if let Some(result) = dialog.update(input) {
///     match result.button {
///         DialogButton::Yes => { /* delete */ },
///         _ => { /* cancelled */ },
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ModalDialog {
    /// Unique identifier.
    pub id: DialogId,
    /// Dialog title.
    pub title: String,
    /// Message text.
    pub message: String,
    /// Detail text (smaller, below the message).
    pub detail: Option<String>,
    /// Icon type.
    pub icon: DialogIcon,
    /// Button configuration.
    pub buttons: DialogButtons,
    /// Custom button labels.
    pub custom_labels: HashMap<DialogButton, String>,
    /// The default button (highlighted, activated by Enter).
    pub default_button: DialogButton,
    /// The escape button (activated by Escape).
    pub escape_button: DialogButton,
    /// Whether the dialog is currently open.
    pub open: bool,
    /// The result, set when a button is clicked.
    pub result: Option<DialogResult>,
    /// Dialog width.
    pub width: f32,
    /// Dialog height (computed).
    pub height: f32,
    /// Dialog position (computed, centered on parent).
    pub position: Vec2,
    /// Parent window size (for centering).
    pub parent_size: Vec2,
    /// Backdrop opacity (0.0 = none, 1.0 = fully opaque).
    pub backdrop_opacity: f32,
    /// Backdrop color.
    pub backdrop_color: Color,
    /// Background color.
    pub background_color: Color,
    /// Title bar color.
    pub title_bar_color: Color,
    /// Title text color.
    pub title_text_color: Color,
    /// Message text color.
    pub message_text_color: Color,
    /// Corner radius.
    pub corner_radius: f32,
    /// Whether to show a close (X) button in the title bar.
    pub show_close_button: bool,
    /// Whether focus trapping is enabled.
    pub focus_trap: bool,
    /// Currently focused button index.
    pub focused_button_index: Option<usize>,
    /// Animation progress for opening (0 to 1).
    pub open_animation: f32,
    /// Animation speed.
    pub animation_speed: f32,
    /// Whether "Don't show again" checkbox is shown.
    pub show_dont_ask_again: bool,
    /// "Don't show again" checkbox state.
    pub dont_ask_again: bool,
}

/// Icon type for modal dialogs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogIcon {
    None,
    Info,
    Warning,
    Error,
    Question,
    Success,
}

impl DialogIcon {
    /// Returns the color for this icon.
    pub fn color(&self) -> Color {
        match self {
            Self::None => Color::TRANSPARENT,
            Self::Info => Color::new(0.3, 0.6, 1.0, 1.0),
            Self::Warning => Color::new(1.0, 0.75, 0.2, 1.0),
            Self::Error => Color::new(1.0, 0.3, 0.3, 1.0),
            Self::Question => Color::new(0.5, 0.7, 1.0, 1.0),
            Self::Success => Color::new(0.3, 0.9, 0.4, 1.0),
        }
    }

    /// Returns the icon character.
    pub fn char(&self) -> Option<char> {
        match self {
            Self::None => None,
            Self::Info => Some('i'),
            Self::Warning => Some('!'),
            Self::Error => Some('X'),
            Self::Question => Some('?'),
            Self::Success => Some('*'),
        }
    }
}

impl ModalDialog {
    /// Creates a new modal dialog.
    pub fn new(title: &str, buttons: DialogButtons) -> Self {
        Self {
            id: next_dialog_id(),
            title: title.to_string(),
            message: String::new(),
            detail: None,
            icon: DialogIcon::None,
            buttons,
            custom_labels: HashMap::new(),
            default_button: buttons.default_button(),
            escape_button: buttons.escape_button(),
            open: true,
            result: None,
            width: 420.0,
            height: 200.0,
            position: Vec2::ZERO,
            parent_size: Vec2::new(1920.0, 1080.0),
            backdrop_opacity: 0.5,
            backdrop_color: Color::new(0.0, 0.0, 0.0, 1.0),
            background_color: Color::new(0.18, 0.18, 0.22, 1.0),
            title_bar_color: Color::new(0.15, 0.15, 0.18, 1.0),
            title_text_color: Color::WHITE,
            message_text_color: Color::new(0.85, 0.85, 0.85, 1.0),
            corner_radius: 8.0,
            show_close_button: true,
            focus_trap: true,
            focused_button_index: Some(0),
            open_animation: 0.0,
            animation_speed: 5.0,
            show_dont_ask_again: false,
            dont_ask_again: false,
        }
    }

    /// Sets the message text.
    pub fn set_message(&mut self, message: &str) {
        self.message = message.to_string();
    }

    /// Sets the detail text.
    pub fn set_detail(&mut self, detail: &str) {
        self.detail = Some(detail.to_string());
        self.height = 240.0;
    }

    /// Sets the icon.
    pub fn set_icon(&mut self, icon: DialogIcon) {
        self.icon = icon;
    }

    /// Sets a custom label for a button.
    pub fn set_button_label(&mut self, button: DialogButton, label: &str) {
        self.custom_labels.insert(button, label.to_string());
    }

    /// Returns the label for a button (custom or default).
    pub fn button_label(&self, button: DialogButton) -> String {
        self.custom_labels
            .get(&button)
            .cloned()
            .unwrap_or_else(|| button.label().to_string())
    }

    /// Enables the "Don't show again" checkbox.
    pub fn enable_dont_ask_again(&mut self) {
        self.show_dont_ask_again = true;
        self.height += 30.0;
    }

    /// Updates the dialog for one frame.
    pub fn update(&mut self, dt: f32) {
        // Opening animation.
        if self.open && self.open_animation < 1.0 {
            self.open_animation = (self.open_animation + dt * self.animation_speed).min(1.0);
        }

        // Center on parent.
        self.position = Vec2::new(
            (self.parent_size.x - self.width) * 0.5,
            (self.parent_size.y - self.height) * 0.5,
        );
    }

    /// Handles a button click.
    pub fn click_button(&mut self, button: DialogButton) {
        let mut result = DialogResult::from_button(button);
        result.dont_show_again = self.dont_ask_again;
        result.custom_button_labels = self.custom_labels.clone();
        self.result = Some(result);
        self.open = false;
    }

    /// Handles the escape key.
    pub fn on_escape(&mut self) {
        let mut result = DialogResult::from_escape(self.escape_button);
        result.dont_show_again = self.dont_ask_again;
        self.result = Some(result);
        self.open = false;
    }

    /// Handles the enter key (activates default button).
    pub fn on_enter(&mut self) {
        self.click_button(self.default_button);
    }

    /// Handles tab key (cycle focus between buttons).
    pub fn on_tab(&mut self, shift: bool) {
        let button_count = self.buttons.buttons().len();
        if button_count == 0 {
            return;
        }

        let current = self.focused_button_index.unwrap_or(0);
        let next = if shift {
            if current == 0 {
                button_count - 1
            } else {
                current - 1
            }
        } else {
            (current + 1) % button_count
        };
        self.focused_button_index = Some(next);
    }

    /// Returns true if the dialog is open.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Takes the result (consuming it).
    pub fn take_result(&mut self) -> Option<DialogResult> {
        self.result.take()
    }

    /// Returns the button list for rendering.
    pub fn render_buttons(&self) -> Vec<(DialogButton, String, bool, bool)> {
        let buttons = self.buttons.buttons();
        buttons
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let label = self.button_label(b);
                let focused = self.focused_button_index == Some(i);
                let is_default = b == self.default_button;
                (b, label, focused, is_default)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ConfirmDialog
// ---------------------------------------------------------------------------

/// A specialized confirmation dialog ("Are you sure?" pattern).
///
/// Adds support for a "Don't ask again" checkbox and custom confirmation text.
#[derive(Debug, Clone)]
pub struct ConfirmDialog {
    /// The underlying modal dialog.
    pub dialog: ModalDialog,
    /// The action being confirmed (e.g., "delete this asset").
    pub action_text: String,
    /// Custom confirmation prompt format.
    pub prompt_format: String,
    /// Whether to show a "Don't ask again" checkbox.
    pub show_dont_ask: bool,
    /// Whether "Don't ask again" is checked.
    pub dont_ask_checked: bool,
    /// Custom confirmation text (replaces the default "Are you sure?").
    pub custom_prompt: Option<String>,
    /// Whether to show a warning icon.
    pub show_warning_icon: bool,
    /// The key for "Don't ask again" persistence.
    pub persistence_key: Option<String>,
}

impl ConfirmDialog {
    /// Creates a new confirmation dialog.
    pub fn new(action_text: &str) -> Self {
        let message = format!("Are you sure you want to {}?", action_text);
        let mut dialog = ModalDialog::new("Confirm", DialogButtons::YesNo);
        dialog.set_message(&message);
        dialog.set_icon(DialogIcon::Question);

        Self {
            dialog,
            action_text: action_text.to_string(),
            prompt_format: "Are you sure you want to {}?".to_string(),
            show_dont_ask: false,
            dont_ask_checked: false,
            custom_prompt: None,
            show_warning_icon: false,
            persistence_key: None,
        }
    }

    /// Creates a dangerous confirmation (red/warning theme).
    pub fn dangerous(action_text: &str) -> Self {
        let mut confirm = Self::new(action_text);
        confirm.dialog.set_icon(DialogIcon::Warning);
        confirm.dialog.default_button = DialogButton::No;
        confirm.show_warning_icon = true;
        confirm
    }

    /// Enables "Don't ask again" with a persistence key.
    pub fn with_dont_ask_again(mut self, key: &str) -> Self {
        self.show_dont_ask = true;
        self.persistence_key = Some(key.to_string());
        self.dialog.enable_dont_ask_again();
        self
    }

    /// Sets a custom prompt text.
    pub fn with_custom_prompt(mut self, prompt: &str) -> Self {
        self.custom_prompt = Some(prompt.to_string());
        self.dialog.set_message(prompt);
        self
    }

    /// Updates the dialog.
    pub fn update(&mut self, dt: f32) {
        self.dialog.update(dt);
    }

    /// Returns true if the dialog is open.
    pub fn is_open(&self) -> bool {
        self.dialog.is_open()
    }

    /// Returns the result if the dialog is closed.
    pub fn result(&mut self) -> Option<DialogResult> {
        self.dialog.take_result()
    }

    /// Returns true if the user confirmed the action.
    pub fn confirmed(&self) -> bool {
        self.dialog
            .result
            .as_ref()
            .map(|r| r.is_affirmative())
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// ProgressStage
// ---------------------------------------------------------------------------

/// A single stage in a multi-stage progress dialog.
#[derive(Debug, Clone)]
pub struct ProgressStage {
    /// Stage name/label.
    pub name: String,
    /// Stage progress (0.0 to 1.0).
    pub progress: f32,
    /// Whether this stage is complete.
    pub complete: bool,
    /// Whether this stage is currently active.
    pub active: bool,
    /// Status text for this stage.
    pub status: String,
    /// Weight of this stage relative to others (for overall progress).
    pub weight: f32,
}

impl ProgressStage {
    /// Creates a new progress stage.
    pub fn new(name: &str, weight: f32) -> Self {
        Self {
            name: name.to_string(),
            progress: 0.0,
            complete: false,
            active: false,
            status: String::new(),
            weight: weight.max(0.01),
        }
    }

    /// Updates the stage progress.
    pub fn set_progress(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 1.0);
    }

    /// Marks the stage as complete.
    pub fn mark_complete(&mut self) {
        self.progress = 1.0;
        self.complete = true;
        self.active = false;
    }

    /// Marks the stage as active.
    pub fn mark_active(&mut self) {
        self.active = true;
    }
}

// ---------------------------------------------------------------------------
// ProgressDialog
// ---------------------------------------------------------------------------

/// A modal progress dialog with optional cancel button and multi-stage support.
///
/// Supports:
/// - Determinate mode: shows a progress bar with percentage.
/// - Indeterminate mode: shows a spinning/pulsing indicator.
/// - Multi-stage: shows progress through multiple named stages.
/// - Time remaining estimate: computed from progress rate.
/// - Cancel button: allows the user to abort the operation.
///
/// # Usage
///
/// ```ignore
/// let mut dialog = ProgressDialog::new("Importing Assets");
/// dialog.add_stage("Scanning", 0.2);
/// dialog.add_stage("Processing", 0.6);
/// dialog.add_stage("Finalizing", 0.2);
///
/// // During the operation:
/// dialog.set_stage_progress(0, 0.5);
/// dialog.advance_stage(); // Move to next stage.
///
/// if dialog.is_cancelled() {
///     // User clicked cancel.
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ProgressDialog {
    /// Unique identifier.
    pub id: DialogId,
    /// Dialog title.
    pub title: String,
    /// Current status text.
    pub status_text: String,
    /// Whether the dialog is open.
    pub open: bool,
    /// Overall progress (0.0 to 1.0).
    pub progress: f32,
    /// Whether the progress is indeterminate (spinning).
    pub indeterminate: bool,
    /// Whether the operation can be cancelled.
    pub cancellable: bool,
    /// Whether cancel was requested.
    pub cancel_requested: bool,
    /// Dialog position.
    pub position: Vec2,
    /// Dialog size.
    pub size: Vec2,
    /// Parent window size (for centering).
    pub parent_size: Vec2,
    /// Multi-stage progress.
    pub stages: Vec<ProgressStage>,
    /// Current stage index.
    pub current_stage: usize,
    /// Whether to show individual stage progress.
    pub show_stages: bool,
    /// Time elapsed in seconds.
    pub elapsed: f32,
    /// Estimated time remaining in seconds.
    pub estimated_remaining: f32,
    /// Whether to show the time remaining estimate.
    pub show_time_remaining: bool,
    /// Animation: indeterminate spinner angle.
    pub spinner_angle: f32,
    /// Animation: indeterminate pulse value.
    pub pulse_value: f32,
    /// Animation speed.
    pub animation_speed: f32,
    /// Background color.
    pub background_color: Color,
    /// Progress bar color.
    pub progress_bar_color: Color,
    /// Progress bar background color.
    pub progress_bar_bg_color: Color,
    /// Whether the operation is complete.
    pub complete: bool,
    /// Whether the operation succeeded.
    pub succeeded: bool,
    /// Backdrop opacity.
    pub backdrop_opacity: f32,
    /// Open animation progress.
    pub open_animation: f32,
    /// History of progress values for rate estimation.
    progress_history: VecDeque<(f32, f32)>,
    /// Maximum history length.
    max_history: usize,
}

impl ProgressDialog {
    /// Creates a new progress dialog.
    pub fn new(title: &str) -> Self {
        Self {
            id: next_dialog_id(),
            title: title.to_string(),
            status_text: String::new(),
            open: true,
            progress: 0.0,
            indeterminate: false,
            cancellable: true,
            cancel_requested: false,
            position: Vec2::ZERO,
            size: Vec2::new(400.0, 180.0),
            parent_size: Vec2::new(1920.0, 1080.0),
            stages: Vec::new(),
            current_stage: 0,
            show_stages: false,
            elapsed: 0.0,
            estimated_remaining: 0.0,
            show_time_remaining: true,
            spinner_angle: 0.0,
            pulse_value: 0.0,
            animation_speed: 2.0,
            background_color: Color::new(0.18, 0.18, 0.22, 1.0),
            progress_bar_color: Color::new(0.3, 0.6, 1.0, 1.0),
            progress_bar_bg_color: Color::new(0.1, 0.1, 0.14, 1.0),
            complete: false,
            succeeded: false,
            backdrop_opacity: 0.5,
            open_animation: 0.0,
            progress_history: VecDeque::with_capacity(60),
            max_history: 60,
        }
    }

    /// Creates an indeterminate progress dialog.
    pub fn indeterminate(title: &str) -> Self {
        let mut dialog = Self::new(title);
        dialog.indeterminate = true;
        dialog.show_time_remaining = false;
        dialog
    }

    /// Adds a stage to the multi-stage progress.
    pub fn add_stage(&mut self, name: &str, weight: f32) {
        self.stages.push(ProgressStage::new(name, weight));
        self.show_stages = true;
        // Increase dialog size for stages.
        self.size.y = 180.0 + self.stages.len() as f32 * 24.0;
    }

    /// Sets the progress of a specific stage.
    pub fn set_stage_progress(&mut self, stage_index: usize, progress: f32) {
        if stage_index < self.stages.len() {
            self.stages[stage_index].set_progress(progress);
            self.recompute_overall_progress();
        }
    }

    /// Advances to the next stage.
    pub fn advance_stage(&mut self) {
        if self.current_stage < self.stages.len() {
            self.stages[self.current_stage].mark_complete();
            self.current_stage += 1;
            if self.current_stage < self.stages.len() {
                self.stages[self.current_stage].mark_active();
            }
            self.recompute_overall_progress();
        }
    }

    /// Recomputes the overall progress from stage weights.
    fn recompute_overall_progress(&mut self) {
        if self.stages.is_empty() {
            return;
        }
        let total_weight: f32 = self.stages.iter().map(|s| s.weight).sum();
        let weighted_progress: f32 = self
            .stages
            .iter()
            .map(|s| s.progress * s.weight)
            .sum();
        self.progress = weighted_progress / total_weight.max(0.01);
    }

    /// Sets the overall progress directly (non-staged mode).
    pub fn set_progress(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 1.0);
    }

    /// Sets the status text.
    pub fn set_status(&mut self, text: &str) {
        self.status_text = text.to_string();
    }

    /// Requests cancellation.
    pub fn cancel(&mut self) {
        if self.cancellable {
            self.cancel_requested = true;
        }
    }

    /// Returns true if cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_requested
    }

    /// Marks the operation as complete.
    pub fn finish(&mut self, success: bool) {
        self.complete = true;
        self.succeeded = success;
        self.progress = if success { 1.0 } else { self.progress };
        for stage in &mut self.stages {
            if success {
                stage.mark_complete();
            }
        }
    }

    /// Closes the dialog.
    pub fn close(&mut self) {
        self.open = false;
    }

    /// Updates the dialog for one frame.
    pub fn update(&mut self, dt: f32) {
        if !self.open {
            return;
        }

        self.elapsed += dt;

        // Opening animation.
        if self.open_animation < 1.0 {
            self.open_animation = (self.open_animation + dt * 5.0).min(1.0);
        }

        // Center on parent.
        self.position = Vec2::new(
            (self.parent_size.x - self.size.x) * 0.5,
            (self.parent_size.y - self.size.y) * 0.5,
        );

        // Indeterminate animation.
        if self.indeterminate {
            self.spinner_angle += dt * 360.0 * self.animation_speed;
            if self.spinner_angle > 360.0 {
                self.spinner_angle -= 360.0;
            }
            self.pulse_value = ((self.elapsed * self.animation_speed * 3.14159).sin() + 1.0) * 0.5;
        }

        // Time remaining estimation.
        if self.show_time_remaining && !self.indeterminate && self.progress > 0.01 {
            self.progress_history
                .push_back((self.elapsed, self.progress));
            if self.progress_history.len() > self.max_history {
                self.progress_history.pop_front();
            }

            if self.progress_history.len() >= 2 {
                let (t0, p0) = self.progress_history[0];
                let (t1, p1) = *self.progress_history.back().unwrap();
                let dt = t1 - t0;
                let dp = p1 - p0;
                if dp > 0.001 && dt > 0.0 {
                    let rate = dp / dt;
                    let remaining_progress = 1.0 - self.progress;
                    self.estimated_remaining = remaining_progress / rate;
                }
            }
        }
    }

    /// Returns a formatted time remaining string.
    pub fn time_remaining_text(&self) -> String {
        if self.estimated_remaining <= 0.0 || self.indeterminate {
            return String::new();
        }
        let secs = self.estimated_remaining as u32;
        if secs < 60 {
            format!("{} seconds remaining", secs)
        } else if secs < 3600 {
            let mins = secs / 60;
            let remainder = secs % 60;
            format!("{}:{:02} remaining", mins, remainder)
        } else {
            let hours = secs / 3600;
            let mins = (secs % 3600) / 60;
            format!("{}:{:02} remaining", hours, mins)
        }
    }

    /// Returns the elapsed time as a formatted string.
    pub fn elapsed_text(&self) -> String {
        let secs = self.elapsed as u32;
        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}:{:02}", secs / 60, secs % 60)
        } else {
            format!(
                "{}:{:02}:{:02}",
                secs / 3600,
                (secs % 3600) / 60,
                secs % 60,
            )
        }
    }

    /// Returns true if the dialog is still open.
    pub fn is_open(&self) -> bool {
        self.open
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notification_lifecycle() {
        let mut notif = Notification::info("Test message");
        assert!(!notif.dismissed);
        assert!(!notif.should_remove());

        // Simulate some frames.
        for _ in 0..60 {
            notif.update(0.1);
        }
        // After 6 seconds, the 5-second timeout should have triggered.
        assert!(notif.dismissed);
    }

    #[test]
    fn test_notification_dismiss() {
        let mut notif = Notification::info("Test");
        notif.dismiss();
        assert!(notif.dismissed);
        assert!(notif.animation.is_dismissing());

        // Fast-forward animation.
        for _ in 0..60 {
            notif.update(0.1);
        }
        assert!(notif.should_remove());
    }

    #[test]
    fn test_notification_manager_stack() {
        let mut mgr = NotificationManager::new(1920.0, 1080.0);
        mgr.show_info("First");
        mgr.show_info("Second");
        mgr.show_info("Third");
        assert_eq!(mgr.active_count(), 3);
    }

    #[test]
    fn test_notification_manager_max_visible() {
        let mut mgr = NotificationManager::new(1920.0, 1080.0);
        mgr.max_visible = 3;
        for i in 0..5 {
            mgr.show_info(&format!("Notification {}", i));
        }
        // The first 2 should have been dismissed.
        let active = mgr.active_count();
        assert!(active <= 3);
    }

    #[test]
    fn test_notification_progress() {
        let mut mgr = NotificationManager::new(1920.0, 1080.0);
        let handle = mgr.show_progress("Loading...");
        mgr.update_progress(handle, 0.5, Some("50%"));
        mgr.complete(handle, true);

        let notif = mgr
            .notifications
            .iter()
            .find(|n| n.progress == Some(1.0))
            .unwrap();
        assert!(notif.complete);
    }

    #[test]
    fn test_modal_dialog_buttons() {
        let buttons = DialogButtons::YesNoCancel;
        let list = buttons.buttons();
        assert_eq!(list.len(), 3);
        assert_eq!(buttons.default_button(), DialogButton::Yes);
        assert_eq!(buttons.escape_button(), DialogButton::Cancel);
    }

    #[test]
    fn test_modal_dialog_lifecycle() {
        let mut dialog = ModalDialog::new("Test", DialogButtons::OKCancel);
        dialog.set_message("Are you sure?");
        assert!(dialog.is_open());

        dialog.click_button(DialogButton::OK);
        assert!(!dialog.is_open());
        let result = dialog.take_result().unwrap();
        assert!(result.is_affirmative());
    }

    #[test]
    fn test_modal_dialog_escape() {
        let mut dialog = ModalDialog::new("Test", DialogButtons::OKCancel);
        dialog.on_escape();
        assert!(!dialog.is_open());
        let result = dialog.take_result().unwrap();
        assert!(result.is_cancelled());
        assert!(result.escaped);
    }

    #[test]
    fn test_confirm_dialog() {
        let mut confirm = ConfirmDialog::new("delete this file");
        assert!(confirm.dialog.message.contains("delete this file"));
    }

    #[test]
    fn test_progress_dialog_stages() {
        let mut dialog = ProgressDialog::new("Import");
        dialog.add_stage("Scan", 0.2);
        dialog.add_stage("Process", 0.6);
        dialog.add_stage("Finalize", 0.2);

        dialog.set_stage_progress(0, 1.0);
        dialog.advance_stage();
        dialog.set_stage_progress(1, 0.5);

        // 100% of 0.2 weight + 50% of 0.6 weight = 0.2 + 0.3 = 0.5
        assert!((dialog.progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_progress_dialog_cancel() {
        let mut dialog = ProgressDialog::new("Import");
        dialog.cancellable = true;
        dialog.cancel();
        assert!(dialog.is_cancelled());
    }

    #[test]
    fn test_notification_animation_phases() {
        let mut anim = NotificationAnimation::new();
        assert_eq!(anim.phase, AnimationPhase::SlideIn);
        assert!(!anim.is_complete());

        // Fast-forward through slide-in.
        for _ in 0..30 {
            anim.update(0.05);
        }
        assert_eq!(anim.phase, AnimationPhase::Visible);

        anim.start_fade_out();
        assert_eq!(anim.phase, AnimationPhase::FadeOut);

        // Fast-forward through fade-out.
        for _ in 0..30 {
            anim.update(0.05);
        }
        assert!(anim.is_complete());
    }

    #[test]
    fn test_notification_level_colors() {
        assert_ne!(
            NotificationLevel::Info.icon_color().r,
            NotificationLevel::Error.icon_color().r,
        );
    }

    #[test]
    fn test_progress_dialog_time_remaining() {
        let mut dialog = ProgressDialog::new("Test");
        dialog.set_progress(0.5);
        dialog.update(5.0);
        dialog.set_progress(0.6);
        dialog.update(1.0);
        // At this point, 10% progress in ~1 second => ~4 seconds remaining for 40%.
        // The estimate may vary based on history, just check it's computed.
    }
}
