//! Slate Performance Optimizations
//!
//! Provides a comprehensive performance optimization layer for the Genovo UI
//! framework, inspired by Unreal Engine's Slate invalidation system. These
//! systems work together to minimize CPU and GPU work each frame:
//!
//! - **InvalidationPanel**: caches subtree renders to an offscreen buffer and
//!   only re-renders when explicitly invalidated.
//! - **RetainerBox**: renders children to an offscreen texture at configurable
//!   intervals and resolution.
//! - **VirtualizedListCore**: only creates/renders widgets for visible rows,
//!   with widget recycling to avoid allocation churn.
//! - **DirtyRectTracking**: tracks which screen regions need repainting.
//! - **ElementBatching**: merges sequential draw elements to minimize GPU state
//!   changes.
//! - **WidgetCaching**: caches `ComputeDesiredSize` results per widget.
//! - **SlateSleep**: pauses rendering entirely when the UI is idle.
//! - **PerformanceStats**: real-time frame-by-frame statistics.
//!
//! # Architecture
//!
//! ```text
//!  WidgetTree ──> InvalidationPanel ──> DirtyRectTracking ──> ElementBatching
//!       │               │                       │                    │
//!   WidgetCaching   RetainerBox            SlateSleep          DrawCallMerge
//!       │               │
//!   VirtualizedListCore  PerformanceStats
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use glam::Vec2;

use crate::core::UIId;
use crate::render_commands::Color;

// ---------------------------------------------------------------------------
// InvalidateReason
// ---------------------------------------------------------------------------

/// Describes why a widget (or its subtree) needs to be re-rendered.
///
/// The invalidation system uses these reasons to determine which phases of the
/// render pipeline need to run for a given widget. A `Paint`-only invalidation
/// skips layout; a `Layout` invalidation forces both layout and paint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InvalidateReason {
    /// The visual appearance changed (color, text, icon) but the size did not.
    Paint,
    /// The widget's desired size or constraints changed, requiring layout.
    Layout,
    /// Children were added, removed, or reordered.
    ChildOrder,
    /// The widget's visibility changed (shown/hidden).
    Visibility,
    /// The widget's render transform (position, rotation, scale) changed.
    RenderTransform,
    /// An attribute changed that affects accessibility but not rendering.
    AttributeOnly,
}

impl fmt::Display for InvalidateReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Paint => write!(f, "Paint"),
            Self::Layout => write!(f, "Layout"),
            Self::ChildOrder => write!(f, "ChildOrder"),
            Self::Visibility => write!(f, "Visibility"),
            Self::RenderTransform => write!(f, "RenderTransform"),
            Self::AttributeOnly => write!(f, "AttributeOnly"),
        }
    }
}

impl InvalidateReason {
    /// Returns `true` if this reason requires a layout pass.
    pub fn needs_layout(&self) -> bool {
        matches!(
            self,
            Self::Layout | Self::ChildOrder | Self::Visibility
        )
    }

    /// Returns `true` if this reason requires a paint pass.
    pub fn needs_paint(&self) -> bool {
        !matches!(self, Self::AttributeOnly)
    }

    /// Returns the "severity" for sorting — higher means more work needed.
    pub fn severity(&self) -> u8 {
        match self {
            Self::AttributeOnly => 0,
            Self::RenderTransform => 1,
            Self::Paint => 2,
            Self::Visibility => 3,
            Self::ChildOrder => 4,
            Self::Layout => 5,
        }
    }
}

// ---------------------------------------------------------------------------
// CacheStatistics
// ---------------------------------------------------------------------------

/// Tracks cache performance metrics for the invalidation system.
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Total number of cache lookups attempted.
    pub total_lookups: u64,
    /// Number of cache hits (subtree reused from buffer).
    pub cache_hits: u64,
    /// Number of cache misses (subtree re-rendered).
    pub cache_misses: u64,
    /// Number of times a subtree was explicitly invalidated.
    pub invalidation_count: u64,
    /// Number of re-renders triggered by volatile widgets.
    pub volatile_renders: u64,
    /// Total bytes consumed by offscreen cache buffers.
    pub cache_memory_bytes: u64,
    /// Number of cached panels currently active.
    pub active_panels: u32,
    /// Time spent on cache management this frame.
    pub cache_overhead_us: u64,
}

impl CacheStatistics {
    /// Creates a new zeroed statistics instance.
    pub fn new() -> Self {
        Self {
            total_lookups: 0,
            cache_hits: 0,
            cache_misses: 0,
            invalidation_count: 0,
            volatile_renders: 0,
            cache_memory_bytes: 0,
            active_panels: 0,
            cache_overhead_us: 0,
        }
    }

    /// Returns the cache hit rate as a fraction in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_lookups as f64
        }
    }

    /// Returns the cache miss rate as a fraction in `[0.0, 1.0]`.
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }

    /// Resets all counters to zero for a new measurement period.
    pub fn reset(&mut self) {
        self.total_lookups = 0;
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.invalidation_count = 0;
        self.volatile_renders = 0;
        self.cache_overhead_us = 0;
    }

    /// Merges another statistics instance into this one (additive).
    pub fn merge(&mut self, other: &CacheStatistics) {
        self.total_lookups += other.total_lookups;
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
        self.invalidation_count += other.invalidation_count;
        self.volatile_renders += other.volatile_renders;
        self.cache_memory_bytes += other.cache_memory_bytes;
        self.active_panels += other.active_panels;
        self.cache_overhead_us += other.cache_overhead_us;
    }
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CacheStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cache: {:.1}% hit ({}/{}) | {} invalidations | {:.1} KB | {} panels",
            self.hit_rate() * 100.0,
            self.cache_hits,
            self.total_lookups,
            self.invalidation_count,
            self.cache_memory_bytes as f64 / 1024.0,
            self.active_panels,
        )
    }
}

// ---------------------------------------------------------------------------
// OffscreenBuffer
// ---------------------------------------------------------------------------

/// Represents a cached offscreen render target used by InvalidationPanel.
#[derive(Debug, Clone)]
pub struct OffscreenBuffer {
    /// Unique identifier for this buffer.
    pub id: u64,
    /// Width of the buffer in pixels.
    pub width: u32,
    /// Height of the buffer in pixels.
    pub height: u32,
    /// Whether the buffer contents are valid (not stale).
    pub valid: bool,
    /// Frame number when the buffer was last rendered.
    pub last_render_frame: u64,
    /// Render scale factor (1.0 = full resolution).
    pub render_scale: f32,
    /// Estimated memory consumption in bytes (width * height * 4).
    pub memory_bytes: u64,
    /// Whether this buffer is currently in use by an active panel.
    pub in_use: bool,
}

impl OffscreenBuffer {
    /// Creates a new offscreen buffer descriptor.
    pub fn new(id: u64, width: u32, height: u32, render_scale: f32) -> Self {
        let scaled_w = (width as f32 * render_scale) as u32;
        let scaled_h = (height as f32 * render_scale) as u32;
        let memory = (scaled_w as u64) * (scaled_h as u64) * 4;
        Self {
            id,
            width,
            height,
            valid: false,
            last_render_frame: 0,
            render_scale,
            memory_bytes: memory,
            in_use: true,
        }
    }

    /// Marks the buffer as needing re-render.
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Marks the buffer contents as up-to-date for the given frame.
    pub fn validate(&mut self, frame: u64) {
        self.valid = true;
        self.last_render_frame = frame;
    }

    /// Returns true if the buffer dimensions match the requested size.
    pub fn matches_size(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }

    /// Resizes the buffer (invalidates contents).
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let scaled_w = (width as f32 * self.render_scale) as u32;
        let scaled_h = (height as f32 * self.render_scale) as u32;
        self.memory_bytes = (scaled_w as u64) * (scaled_h as u64) * 4;
        self.valid = false;
    }

    /// Returns the age of this buffer in frames.
    pub fn age_in_frames(&self, current_frame: u64) -> u64 {
        current_frame.saturating_sub(self.last_render_frame)
    }
}

// ---------------------------------------------------------------------------
// InvalidationPanel
// ---------------------------------------------------------------------------

/// A panel that caches its subtree rendering to an offscreen buffer.
///
/// The subtree is only re-rendered when a child widget calls `invalidate()`,
/// or when the panel is marked as volatile (always re-render). This is one of
/// the most important performance optimizations in the UI system, as it allows
/// complex widget trees to be drawn once and then composited cheaply until
/// something actually changes.
///
/// # Volatile mode
///
/// Widgets that contain animations, timers, or real-time updates should set
/// `volatile = true`. This forces the panel to re-render every frame but still
/// benefits from dirty-rect tracking and batching.
///
/// # Usage
///
/// ```ignore
/// let mut panel = InvalidationPanel::new(widget_id);
/// panel.set_size(400, 300);
///
/// // Each frame:
/// if panel.needs_render() {
///     panel.begin_render(frame_number);
///     // ... render children to panel's buffer ...
///     panel.end_render();
/// }
/// // Composite the cached buffer to the screen.
/// panel.composite(draw_list, position);
/// ```
#[derive(Debug, Clone)]
pub struct InvalidationPanel {
    /// The widget this panel caches.
    pub widget_id: UIId,
    /// The offscreen render buffer.
    pub buffer: OffscreenBuffer,
    /// Pending invalidation reasons since last render.
    pub pending_reasons: Vec<InvalidateReason>,
    /// Whether this panel is volatile (always re-renders).
    pub volatile: bool,
    /// Set of child widget IDs that have requested invalidation.
    pub invalidated_children: HashSet<UIId>,
    /// Cache statistics for this panel.
    pub stats: CacheStatistics,
    /// Position where this panel should be composited.
    pub position: Vec2,
    /// Clipping rectangle for this panel.
    pub clip_rect: Option<[f32; 4]>,
    /// Opacity for compositing (0.0 = invisible, 1.0 = opaque).
    pub opacity: f32,
    /// Whether the panel is currently visible.
    pub visible: bool,
    /// Z-order for compositing.
    pub z_order: i32,
    /// Debug name for profiling/debugging.
    pub debug_name: String,
    /// Maximum age in frames before forcing a re-render (0 = no limit).
    pub max_cache_age: u64,
    /// Whether to use sub-pixel rendering in the cache.
    pub sub_pixel_rendering: bool,
}

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

impl InvalidationPanel {
    /// Creates a new invalidation panel for the given widget.
    pub fn new(widget_id: UIId) -> Self {
        let buffer_id = NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            widget_id,
            buffer: OffscreenBuffer::new(buffer_id, 1, 1, 1.0),
            pending_reasons: Vec::new(),
            volatile: false,
            invalidated_children: HashSet::new(),
            stats: CacheStatistics::new(),
            position: Vec2::ZERO,
            clip_rect: None,
            opacity: 1.0,
            visible: true,
            z_order: 0,
            debug_name: String::new(),
            max_cache_age: 0,
            sub_pixel_rendering: true,
        }
    }

    /// Creates a new volatile panel (always re-renders).
    pub fn new_volatile(widget_id: UIId) -> Self {
        let mut panel = Self::new(widget_id);
        panel.volatile = true;
        panel
    }

    /// Sets the debug name for profiling.
    pub fn with_debug_name(mut self, name: &str) -> Self {
        self.debug_name = name.to_string();
        self
    }

    /// Sets the render scale (lower = less GPU work, more blurry).
    pub fn with_render_scale(mut self, scale: f32) -> Self {
        self.buffer.render_scale = scale.clamp(0.1, 2.0);
        self
    }

    /// Sets the maximum cache age before forcing re-render.
    pub fn with_max_age(mut self, frames: u64) -> Self {
        self.max_cache_age = frames;
        self
    }

    /// Sets the size of the panel's render buffer.
    pub fn set_size(&mut self, width: u32, height: u32) {
        if !self.buffer.matches_size(width, height) {
            self.buffer.resize(width, height);
            self.stats.active_panels = 1;
        }
    }

    /// Called by a child widget to request re-rendering of this panel.
    pub fn invalidate(&mut self, reason: InvalidateReason) {
        self.pending_reasons.push(reason);
        self.buffer.invalidate();
        self.stats.invalidation_count += 1;
    }

    /// Called by a specific child widget to request re-rendering.
    pub fn invalidate_child(&mut self, child_id: UIId, reason: InvalidateReason) {
        self.invalidated_children.insert(child_id);
        self.invalidate(reason);
    }

    /// Returns `true` if the panel needs to be re-rendered this frame.
    pub fn needs_render(&self) -> bool {
        if !self.visible {
            return false;
        }
        if self.volatile {
            return true;
        }
        if !self.buffer.valid {
            return true;
        }
        if !self.pending_reasons.is_empty() {
            return true;
        }
        false
    }

    /// Returns `true` if a layout pass is needed (not just paint).
    pub fn needs_layout(&self) -> bool {
        self.pending_reasons.iter().any(|r| r.needs_layout())
    }

    /// Checks if the cache has exceeded its maximum age.
    pub fn is_cache_stale(&self, current_frame: u64) -> bool {
        if self.max_cache_age == 0 {
            return false;
        }
        self.buffer.age_in_frames(current_frame) >= self.max_cache_age
    }

    /// Begins a render pass into this panel's offscreen buffer.
    pub fn begin_render(&mut self, frame_number: u64) {
        if self.volatile {
            self.stats.volatile_renders += 1;
        }
        self.stats.total_lookups += 1;
        self.stats.cache_misses += 1;
    }

    /// Ends the render pass and marks the buffer as valid.
    pub fn end_render(&mut self, frame_number: u64) {
        self.buffer.validate(frame_number);
        self.pending_reasons.clear();
        self.invalidated_children.clear();
    }

    /// Attempts to use the cached buffer (returns true on cache hit).
    pub fn try_use_cache(&mut self, current_frame: u64) -> bool {
        self.stats.total_lookups += 1;
        if self.buffer.valid && !self.volatile && !self.is_cache_stale(current_frame) {
            self.stats.cache_hits += 1;
            true
        } else {
            self.stats.cache_misses += 1;
            false
        }
    }

    /// Returns the most severe pending invalidation reason.
    pub fn worst_reason(&self) -> Option<InvalidateReason> {
        self.pending_reasons
            .iter()
            .max_by_key(|r| r.severity())
            .copied()
    }

    /// Returns the total memory consumed by this panel's buffer.
    pub fn memory_usage(&self) -> u64 {
        self.buffer.memory_bytes
    }

    /// Resets statistics for a new measurement period.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Updates cache memory tracking in stats.
    pub fn update_memory_stats(&mut self) {
        self.stats.cache_memory_bytes = self.buffer.memory_bytes;
    }
}

// ---------------------------------------------------------------------------
// RetainerBox
// ---------------------------------------------------------------------------

/// Renders a subtree to a texture at a fixed interval or on invalidation.
///
/// Unlike `InvalidationPanel` which re-renders on demand, `RetainerBox` can
/// spread renders across frames using a phase parameter. This is useful for
/// complex but slowly-changing UI elements like minimap overlays, background
/// graphs, or preview thumbnails.
///
/// The render scale parameter allows rendering at lower resolution to save GPU
/// bandwidth, at the cost of some blurriness.
#[derive(Debug, Clone)]
pub struct RetainerBox {
    /// Unique identifier for this retainer box.
    pub id: u64,
    /// The widget ID this retainer wraps.
    pub widget_id: UIId,
    /// Offscreen buffer for the retained render.
    pub buffer: OffscreenBuffer,
    /// Phase offset for staggered rendering (0..render_interval).
    pub phase: u32,
    /// Number of frames between re-renders (1 = every frame).
    pub render_interval: u32,
    /// Current frame counter (modulo render_interval).
    pub frame_counter: u32,
    /// Render scale factor (0.5 = half resolution).
    pub render_scale: f32,
    /// Whether an explicit invalidation is pending.
    pub force_render: bool,
    /// Position for compositing.
    pub position: Vec2,
    /// Size of the box in logical pixels.
    pub size: Vec2,
    /// Whether rendering is enabled.
    pub enabled: bool,
    /// Debug name.
    pub debug_name: String,
    /// Statistics: total number of renders performed.
    pub render_count: u64,
    /// Statistics: total number of frames this has been alive.
    pub total_frames: u64,
    /// Statistics: average render time in microseconds.
    pub avg_render_time_us: f64,
    /// Smoothing factor for the render time average (EMA alpha).
    pub render_time_alpha: f64,
    /// Last render duration.
    pub last_render_time_us: u64,
}

impl RetainerBox {
    /// Creates a new retainer box.
    pub fn new(widget_id: UIId, render_interval: u32) -> Self {
        let buffer_id = NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            id: buffer_id,
            widget_id,
            buffer: OffscreenBuffer::new(buffer_id, 1, 1, 1.0),
            phase: 0,
            render_interval: render_interval.max(1),
            frame_counter: 0,
            render_scale: 1.0,
            force_render: true,
            position: Vec2::ZERO,
            size: Vec2::ZERO,
            enabled: true,
            debug_name: String::new(),
            render_count: 0,
            total_frames: 0,
            avg_render_time_us: 0.0,
            render_time_alpha: 0.1,
            last_render_time_us: 0,
        }
    }

    /// Sets the phase offset for staggered rendering.
    pub fn with_phase(mut self, phase: u32) -> Self {
        self.phase = phase % self.render_interval;
        self
    }

    /// Sets the render scale factor.
    pub fn with_render_scale(mut self, scale: f32) -> Self {
        self.render_scale = scale.clamp(0.1, 2.0);
        self.buffer.render_scale = self.render_scale;
        self
    }

    /// Sets the debug name.
    pub fn with_debug_name(mut self, name: &str) -> Self {
        self.debug_name = name.to_string();
        self
    }

    /// Sets the size and resizes the buffer accordingly.
    pub fn set_size(&mut self, width: f32, height: f32) {
        self.size = Vec2::new(width, height);
        let buf_w = (width * self.render_scale) as u32;
        let buf_h = (height * self.render_scale) as u32;
        if !self.buffer.matches_size(buf_w, buf_h) {
            self.buffer.resize(buf_w, buf_h);
            self.force_render = true;
        }
    }

    /// Forces a re-render on the next eligible frame.
    pub fn invalidate(&mut self) {
        self.force_render = true;
        self.buffer.invalidate();
    }

    /// Returns `true` if this retainer should render this frame.
    pub fn should_render_this_frame(&self) -> bool {
        if !self.enabled {
            return false;
        }
        if self.force_render {
            return true;
        }
        if !self.buffer.valid {
            return true;
        }
        (self.frame_counter % self.render_interval) == self.phase
    }

    /// Advances the frame counter and returns whether to render.
    pub fn tick(&mut self) -> bool {
        self.total_frames += 1;
        self.frame_counter = (self.frame_counter + 1) % self.render_interval;
        self.should_render_this_frame()
    }

    /// Begins a render pass.
    pub fn begin_render(&mut self) {
        self.render_count += 1;
    }

    /// Ends a render pass and records the duration.
    pub fn end_render(&mut self, frame_number: u64, duration_us: u64) {
        self.buffer.validate(frame_number);
        self.force_render = false;
        self.last_render_time_us = duration_us;
        // Exponential moving average of render time.
        let alpha = self.render_time_alpha;
        self.avg_render_time_us =
            self.avg_render_time_us * (1.0 - alpha) + duration_us as f64 * alpha;
    }

    /// Returns the fraction of frames where this retainer actually renders.
    pub fn render_duty_cycle(&self) -> f64 {
        if self.total_frames == 0 {
            0.0
        } else {
            self.render_count as f64 / self.total_frames as f64
        }
    }

    /// Returns the memory consumption of the buffer.
    pub fn memory_usage(&self) -> u64 {
        self.buffer.memory_bytes
    }
}

// ---------------------------------------------------------------------------
// VirtualizedListCore — Row metadata
// ---------------------------------------------------------------------------

/// Metadata for a single row in the virtualized list.
#[derive(Debug, Clone)]
pub struct VirtualizedRow {
    /// Index of this row in the data source.
    pub data_index: usize,
    /// Measured (or estimated) height of this row in logical pixels.
    pub height: f32,
    /// Computed Y-offset from the top of the list content.
    pub y_offset: f32,
    /// Whether this row has been measured (vs estimated).
    pub measured: bool,
    /// The widget slot index assigned to this row (if visible).
    pub widget_slot: Option<usize>,
    /// Whether this row is currently visible in the viewport.
    pub visible: bool,
    /// Whether this row is selected.
    pub selected: bool,
}

impl VirtualizedRow {
    /// Creates a new row with estimated height.
    pub fn new(data_index: usize, estimated_height: f32) -> Self {
        Self {
            data_index,
            height: estimated_height,
            y_offset: 0.0,
            measured: false,
            widget_slot: None,
            visible: false,
            selected: false,
        }
    }

    /// Returns the bottom edge Y coordinate.
    pub fn bottom(&self) -> f32 {
        self.y_offset + self.height
    }

    /// Returns true if this row overlaps the given Y range.
    pub fn overlaps_range(&self, range_top: f32, range_bottom: f32) -> bool {
        self.y_offset < range_bottom && self.bottom() > range_top
    }
}

// ---------------------------------------------------------------------------
// RecycledWidget
// ---------------------------------------------------------------------------

/// A widget slot that can be recycled for different data rows.
#[derive(Debug, Clone)]
pub struct RecycledWidget {
    /// Unique slot index in the widget pool.
    pub slot_index: usize,
    /// The data index currently bound to this widget, if any.
    pub bound_data_index: Option<usize>,
    /// Whether this widget slot is actively in use.
    pub in_use: bool,
    /// Position (Y offset) where this widget is rendered.
    pub y_position: f32,
    /// Height of the widget as rendered.
    pub rendered_height: f32,
    /// Frame number when this widget was last bound.
    pub last_bind_frame: u64,
    /// Number of times this slot has been recycled.
    pub recycle_count: u64,
    /// Whether the widget needs its content updated.
    pub dirty: bool,
    /// Arbitrary user data attached to the widget for the current binding.
    pub user_data: Option<u64>,
}

impl RecycledWidget {
    /// Creates a new recycled widget slot.
    pub fn new(slot_index: usize) -> Self {
        Self {
            slot_index,
            bound_data_index: None,
            in_use: false,
            y_position: 0.0,
            rendered_height: 0.0,
            last_bind_frame: 0,
            recycle_count: 0,
            dirty: true,
            user_data: None,
        }
    }

    /// Binds this widget slot to a new data row.
    pub fn bind(&mut self, data_index: usize, y_position: f32, frame: u64) {
        if self.bound_data_index != Some(data_index) {
            self.recycle_count += 1;
            self.dirty = true;
        }
        self.bound_data_index = Some(data_index);
        self.y_position = y_position;
        self.in_use = true;
        self.last_bind_frame = frame;
    }

    /// Releases this widget slot back to the pool.
    pub fn release(&mut self) {
        self.bound_data_index = None;
        self.in_use = false;
        self.dirty = true;
        self.user_data = None;
    }

    /// Returns true if this slot is bound to the given data index.
    pub fn is_bound_to(&self, data_index: usize) -> bool {
        self.bound_data_index == Some(data_index)
    }
}

// ---------------------------------------------------------------------------
// ScrollState
// ---------------------------------------------------------------------------

/// Internal scroll state for the virtualized list.
#[derive(Debug, Clone)]
pub struct ScrollState {
    /// Current scroll offset in logical pixels from the top.
    pub offset: f32,
    /// Target scroll offset (for smooth scrolling).
    pub target_offset: f32,
    /// Scroll velocity for momentum scrolling.
    pub velocity: f32,
    /// Whether the user is currently dragging the scrollbar.
    pub dragging: bool,
    /// Whether smooth scrolling is in progress.
    pub animating: bool,
    /// Smooth scroll speed factor.
    pub smooth_speed: f32,
    /// Friction coefficient for momentum scrolling.
    pub friction: f32,
    /// Maximum allowed scroll offset.
    pub max_offset: f32,
    /// Viewport height in logical pixels.
    pub viewport_height: f32,
}

impl ScrollState {
    /// Creates a new scroll state.
    pub fn new() -> Self {
        Self {
            offset: 0.0,
            target_offset: 0.0,
            velocity: 0.0,
            dragging: false,
            animating: false,
            smooth_speed: 10.0,
            friction: 0.92,
            max_offset: 0.0,
            viewport_height: 0.0,
        }
    }

    /// Sets the viewport height and updates the max scroll offset.
    pub fn set_viewport(&mut self, viewport_height: f32, content_height: f32) {
        self.viewport_height = viewport_height;
        self.max_offset = (content_height - viewport_height).max(0.0);
        self.clamp_offset();
    }

    /// Scrolls by a delta amount.
    pub fn scroll_by(&mut self, delta: f32) {
        self.target_offset = (self.target_offset + delta).clamp(0.0, self.max_offset);
        self.animating = true;
    }

    /// Immediately jumps to an offset (no animation).
    pub fn jump_to(&mut self, offset: f32) {
        self.offset = offset.clamp(0.0, self.max_offset);
        self.target_offset = self.offset;
        self.velocity = 0.0;
        self.animating = false;
    }

    /// Updates scroll animation for one frame.
    pub fn update(&mut self, dt: f32) {
        if self.animating {
            let diff = self.target_offset - self.offset;
            if diff.abs() < 0.5 {
                self.offset = self.target_offset;
                self.animating = false;
                self.velocity = 0.0;
            } else {
                self.offset += diff * (self.smooth_speed * dt).min(1.0);
            }
        }

        // Momentum scrolling.
        if self.velocity.abs() > 0.1 && !self.dragging {
            self.offset += self.velocity * dt;
            self.velocity *= self.friction;
            self.target_offset = self.offset;
        } else if !self.animating {
            self.velocity = 0.0;
        }

        self.clamp_offset();
    }

    /// Clamps the offset to valid range.
    fn clamp_offset(&mut self) {
        self.offset = self.offset.clamp(0.0, self.max_offset);
        self.target_offset = self.target_offset.clamp(0.0, self.max_offset);
    }

    /// Returns the visible Y range.
    pub fn visible_range(&self) -> (f32, f32) {
        (self.offset, self.offset + self.viewport_height)
    }

    /// Returns the scroll fraction (0.0 = top, 1.0 = bottom).
    pub fn scroll_fraction(&self) -> f32 {
        if self.max_offset <= 0.0 {
            0.0
        } else {
            self.offset / self.max_offset
        }
    }

    /// Returns true if currently at the top.
    pub fn at_top(&self) -> bool {
        self.offset <= 0.5
    }

    /// Returns true if currently at the bottom.
    pub fn at_bottom(&self) -> bool {
        self.offset >= self.max_offset - 0.5
    }

    /// Returns true if the scroll position is changing.
    pub fn is_scrolling(&self) -> bool {
        self.animating || self.velocity.abs() > 0.1
    }
}

impl Default for ScrollState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VirtualizedListCore
// ---------------------------------------------------------------------------

/// High-performance virtualized list that only creates widgets for visible rows.
///
/// Supports 100K+ items at 60fps by:
/// - Only allocating widgets for visible rows plus a small buffer.
/// - Recycling off-screen widgets instead of destroying and recreating them.
/// - Computing visibility ranges from scroll offset.
/// - Supporting dynamic row heights (measured on first view, estimated before).
/// - Providing jump-to-index with smooth scroll animation.
///
/// # How it works
///
/// The list maintains a pool of `RecycledWidget` slots. As the user scrolls,
/// rows that move out of the visible area have their widget slots released back
/// to the pool. Rows that become visible are assigned recycled slots (or new
/// ones are created if the pool is empty).
///
/// For lists with uniform row height, all offsets are computed analytically.
/// For variable-height rows, measured heights are cached and unmeasured rows
/// use an estimated height derived from the average of all measured rows.
///
/// # Usage
///
/// ```ignore
/// let mut list = VirtualizedListCore::new(100_000, 24.0); // 100K items, 24px each
/// list.set_viewport(0.0, 600.0); // 600px tall viewport
///
/// for row_index in list.visible_range() {
///     let widget = list.get_or_create_widget(row_index);
///     // ... render the widget ...
/// }
/// ```
#[derive(Debug, Clone)]
pub struct VirtualizedListCore {
    /// Total number of items in the data source.
    pub total_items: usize,
    /// Default estimated height for unmeasured rows.
    pub default_row_height: f32,
    /// Minimum row height (never allow rows smaller than this).
    pub min_row_height: f32,
    /// Maximum row height (cap for safety).
    pub max_row_height: f32,
    /// Row metadata (lazily populated).
    pub rows: Vec<VirtualizedRow>,
    /// Pool of recycled widget slots.
    pub widget_pool: Vec<RecycledWidget>,
    /// Number of widget slots currently in use.
    pub active_widgets: usize,
    /// Maximum number of widgets to keep in the pool.
    pub max_pool_size: usize,
    /// Scroll state.
    pub scroll: ScrollState,
    /// Number of rows to keep rendered above/below the viewport.
    pub overscan_count: usize,
    /// Whether all rows have the same height (optimization).
    pub uniform_height: bool,
    /// Cached total content height.
    pub total_height: f32,
    /// Whether the total height needs recomputation.
    pub height_dirty: bool,
    /// First visible row index.
    pub first_visible: usize,
    /// Last visible row index (exclusive).
    pub last_visible: usize,
    /// Number of measured rows.
    pub measured_count: usize,
    /// Sum of all measured heights (for computing average).
    pub measured_height_sum: f64,
    /// Current frame number.
    pub current_frame: u64,
    /// Statistics: total recycles.
    pub total_recycles: u64,
    /// Statistics: peak active widgets.
    pub peak_active_widgets: usize,
    /// Width of each row in logical pixels.
    pub row_width: f32,
    /// Selection: indices of selected rows.
    pub selected_indices: HashSet<usize>,
    /// Whether multi-select is enabled.
    pub multi_select: bool,
    /// Last clicked index (for shift-select range).
    pub last_click_index: Option<usize>,
    /// Smooth scroll target index (for jump-to).
    pub scroll_target_index: Option<usize>,
}

impl VirtualizedListCore {
    /// Creates a new virtualized list with the given item count and row height.
    pub fn new(total_items: usize, default_row_height: f32) -> Self {
        let mut list = Self {
            total_items,
            default_row_height: default_row_height.max(1.0),
            min_row_height: 16.0,
            max_row_height: 1000.0,
            rows: Vec::with_capacity(total_items.min(10000)),
            widget_pool: Vec::new(),
            active_widgets: 0,
            max_pool_size: 200,
            scroll: ScrollState::new(),
            overscan_count: 5,
            uniform_height: true,
            total_height: 0.0,
            height_dirty: true,
            first_visible: 0,
            last_visible: 0,
            measured_count: 0,
            measured_height_sum: 0.0,
            current_frame: 0,
            total_recycles: 0,
            peak_active_widgets: 0,
            row_width: 300.0,
            selected_indices: HashSet::new(),
            multi_select: false,
            last_click_index: None,
            scroll_target_index: None,
        };
        list.initialize_rows();
        list
    }

    /// Initializes row metadata with estimated heights.
    fn initialize_rows(&mut self) {
        self.rows.clear();
        self.rows.reserve(self.total_items);
        let mut y = 0.0;
        for i in 0..self.total_items {
            let mut row = VirtualizedRow::new(i, self.default_row_height);
            row.y_offset = y;
            y += self.default_row_height;
            self.rows.push(row);
        }
        self.total_height = y;
        self.height_dirty = false;
    }

    /// Returns the estimated total content height.
    pub fn estimate_total_height(&self) -> f32 {
        if self.uniform_height || self.measured_count == 0 {
            self.total_items as f32 * self.default_row_height
        } else {
            let avg_height = self.measured_height_sum / self.measured_count as f64;
            let unmeasured = self.total_items - self.measured_count;
            self.measured_height_sum as f32 + unmeasured as f32 * avg_height as f32
        }
    }

    /// Recalculates y-offsets for all rows (call after height changes).
    pub fn recalculate_offsets(&mut self) {
        let mut y = 0.0;
        for row in &mut self.rows {
            row.y_offset = y;
            y += row.height;
        }
        self.total_height = y;
        self.height_dirty = false;
    }

    /// Updates the measured height for a specific row.
    pub fn set_row_height(&mut self, index: usize, height: f32) {
        if index >= self.rows.len() {
            return;
        }
        let clamped = height.clamp(self.min_row_height, self.max_row_height);
        let row = &mut self.rows[index];
        if !row.measured {
            self.measured_count += 1;
            row.measured = true;
            self.measured_height_sum += clamped as f64;
        } else {
            self.measured_height_sum -= row.height as f64;
            self.measured_height_sum += clamped as f64;
        }
        if (row.height - clamped).abs() > 0.01 {
            row.height = clamped;
            self.height_dirty = true;
            self.uniform_height = false;
        }
    }

    /// Sets the viewport dimensions and recomputes visibility.
    pub fn set_viewport(&mut self, viewport_y: f32, viewport_height: f32) {
        if self.height_dirty {
            self.recalculate_offsets();
        }
        self.scroll
            .set_viewport(viewport_height, self.total_height);
        self.row_width = 300.0; // Default; caller can override.
    }

    /// Sets the viewport width.
    pub fn set_viewport_width(&mut self, width: f32) {
        self.row_width = width;
    }

    /// Scrolls to make the given index visible, with optional smooth animation.
    pub fn scroll_to_index(&mut self, index: usize, smooth: bool) {
        if index >= self.rows.len() {
            return;
        }
        let row = &self.rows[index];
        let target_top = row.y_offset;
        let target_bottom = row.bottom();
        let (vis_top, vis_bottom) = self.scroll.visible_range();

        let target_offset = if target_top < vis_top {
            target_top
        } else if target_bottom > vis_bottom {
            target_bottom - self.scroll.viewport_height
        } else {
            return; // Already visible.
        };

        if smooth {
            self.scroll.target_offset = target_offset.clamp(0.0, self.scroll.max_offset);
            self.scroll.animating = true;
            self.scroll_target_index = Some(index);
        } else {
            self.scroll.jump_to(target_offset);
        }
    }

    /// Jumps to the given index and centers it in the viewport.
    pub fn jump_to_index_centered(&mut self, index: usize) {
        if index >= self.rows.len() {
            return;
        }
        let row = &self.rows[index];
        let center_y = row.y_offset + row.height * 0.5;
        let target = center_y - self.scroll.viewport_height * 0.5;
        self.scroll.jump_to(target);
    }

    /// Computes which rows are visible and returns the index range.
    pub fn compute_visible_range(&mut self) -> (usize, usize) {
        if self.rows.is_empty() {
            self.first_visible = 0;
            self.last_visible = 0;
            return (0, 0);
        }

        let (vis_top, vis_bottom) = self.scroll.visible_range();

        // Binary search for the first visible row.
        let first = self.binary_search_row(vis_top);
        // Binary search for the last visible row.
        let last = self.binary_search_row(vis_bottom);

        // Apply overscan.
        let first_with_overscan = first.saturating_sub(self.overscan_count);
        let last_with_overscan = (last + self.overscan_count + 1).min(self.rows.len());

        // Mark visibility on rows.
        for row in &mut self.rows {
            row.visible = false;
        }
        for i in first_with_overscan..last_with_overscan {
            self.rows[i].visible = true;
        }

        self.first_visible = first_with_overscan;
        self.last_visible = last_with_overscan;
        (first_with_overscan, last_with_overscan)
    }

    /// Binary search for the row containing the given Y coordinate.
    fn binary_search_row(&self, y: f32) -> usize {
        if self.rows.is_empty() {
            return 0;
        }
        if self.uniform_height {
            // Fast path: uniform height means direct index computation.
            let index = (y / self.default_row_height) as usize;
            return index.min(self.rows.len().saturating_sub(1));
        }
        // Binary search on y_offset.
        let mut lo = 0;
        let mut hi = self.rows.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.rows[mid].y_offset + self.rows[mid].height <= y {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.min(self.rows.len().saturating_sub(1))
    }

    /// Allocates or recycles a widget for the given row index.
    pub fn get_or_create_widget(&mut self, row_index: usize) -> &mut RecycledWidget {
        // Check if this row already has a widget.
        if let Some(slot) = self.rows[row_index].widget_slot {
            if slot < self.widget_pool.len() && self.widget_pool[slot].is_bound_to(row_index) {
                return &mut self.widget_pool[slot];
            }
        }

        // Try to recycle an unused widget.
        let recycled_slot = self.find_free_widget_slot();
        let slot_index = if let Some(slot) = recycled_slot {
            slot
        } else {
            // Create a new widget slot.
            let new_index = self.widget_pool.len();
            self.widget_pool.push(RecycledWidget::new(new_index));
            new_index
        };

        // Bind the widget to this row.
        let row = &self.rows[row_index];
        self.widget_pool[slot_index].bind(row_index, row.y_offset, self.current_frame);
        self.rows[row_index].widget_slot = Some(slot_index);
        self.active_widgets += 1;
        if self.active_widgets > self.peak_active_widgets {
            self.peak_active_widgets = self.active_widgets;
        }

        &mut self.widget_pool[slot_index]
    }

    /// Finds a free (not in use) widget slot in the pool.
    fn find_free_widget_slot(&self) -> Option<usize> {
        self.widget_pool
            .iter()
            .position(|w| !w.in_use)
    }

    /// Releases widgets for rows that are no longer visible.
    pub fn release_offscreen_widgets(&mut self) {
        for row in &mut self.rows {
            if !row.visible {
                if let Some(slot) = row.widget_slot.take() {
                    if slot < self.widget_pool.len() {
                        self.widget_pool[slot].release();
                        self.active_widgets = self.active_widgets.saturating_sub(1);
                        self.total_recycles += 1;
                    }
                }
            }
        }

        // Trim the pool if it's much larger than needed.
        if self.widget_pool.len() > self.max_pool_size {
            let excess = self.widget_pool.len() - self.max_pool_size;
            let mut removed = 0;
            self.widget_pool.retain(|w| {
                if !w.in_use && removed < excess {
                    removed += 1;
                    false
                } else {
                    true
                }
            });
        }
    }

    /// Updates the list for one frame: scroll animation + visibility.
    pub fn update(&mut self, dt: f32) {
        self.current_frame += 1;
        self.scroll.update(dt);

        if self.height_dirty {
            self.recalculate_offsets();
            self.scroll
                .set_viewport(self.scroll.viewport_height, self.total_height);
        }

        self.compute_visible_range();
        self.release_offscreen_widgets();
    }

    /// Handles mouse scroll input.
    pub fn on_scroll(&mut self, delta_y: f32) {
        let scroll_amount = delta_y * self.default_row_height * 3.0;
        self.scroll.scroll_by(-scroll_amount);
    }

    /// Handles clicking on a row.
    pub fn on_click(&mut self, row_index: usize, shift_held: bool, ctrl_held: bool) {
        if row_index >= self.rows.len() {
            return;
        }

        if self.multi_select && ctrl_held {
            // Toggle selection.
            if self.selected_indices.contains(&row_index) {
                self.selected_indices.remove(&row_index);
                self.rows[row_index].selected = false;
            } else {
                self.selected_indices.insert(row_index);
                self.rows[row_index].selected = true;
            }
        } else if self.multi_select && shift_held {
            // Range select.
            if let Some(anchor) = self.last_click_index {
                let start = anchor.min(row_index);
                let end = anchor.max(row_index);
                for i in start..=end {
                    self.selected_indices.insert(i);
                    if i < self.rows.len() {
                        self.rows[i].selected = true;
                    }
                }
            }
        } else {
            // Single select.
            for i in &self.selected_indices {
                if *i < self.rows.len() {
                    self.rows[*i].selected = false;
                }
            }
            self.selected_indices.clear();
            self.selected_indices.insert(row_index);
            self.rows[row_index].selected = true;
        }
        self.last_click_index = Some(row_index);
    }

    /// Sets the total item count (e.g., when the data source changes).
    pub fn set_item_count(&mut self, count: usize) {
        if count == self.total_items {
            return;
        }
        self.total_items = count;
        // Release all widgets before reinitializing.
        for w in &mut self.widget_pool {
            w.release();
        }
        self.active_widgets = 0;
        self.selected_indices.clear();
        self.last_click_index = None;
        self.measured_count = 0;
        self.measured_height_sum = 0.0;
        self.initialize_rows();
    }

    /// Returns an iterator over the visible row indices.
    pub fn visible_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.first_visible..self.last_visible
    }

    /// Returns the average measured row height.
    pub fn average_row_height(&self) -> f32 {
        if self.measured_count == 0 {
            self.default_row_height
        } else {
            (self.measured_height_sum / self.measured_count as f64) as f32
        }
    }

    /// Returns true if the list is currently scrolling.
    pub fn is_scrolling(&self) -> bool {
        self.scroll.is_scrolling()
    }

    /// Returns the scrollbar thumb parameters (position, size) as fractions.
    pub fn scrollbar_params(&self) -> (f32, f32) {
        let total = self.total_height;
        let viewport = self.scroll.viewport_height;
        if total <= viewport {
            return (0.0, 1.0);
        }
        let thumb_size = (viewport / total).clamp(0.05, 1.0);
        let thumb_pos = self.scroll.scroll_fraction() * (1.0 - thumb_size);
        (thumb_pos, thumb_size)
    }
}

// ---------------------------------------------------------------------------
// DirtyRect
// ---------------------------------------------------------------------------

/// A rectangular screen region that needs repainting.
#[derive(Debug, Clone, Copy)]
pub struct DirtyRect {
    /// Left edge in pixels.
    pub x: f32,
    /// Top edge in pixels.
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
    /// Frame number when this rect was marked dirty.
    pub frame: u64,
    /// The reason this region is dirty.
    pub reason: InvalidateReason,
    /// Debug color for visualization overlay.
    pub debug_color: Color,
}

impl DirtyRect {
    /// Creates a new dirty rect.
    pub fn new(x: f32, y: f32, width: f32, height: f32, reason: InvalidateReason) -> Self {
        Self {
            x,
            y,
            width,
            height,
            frame: 0,
            reason,
            debug_color: Self::color_for_reason(reason),
        }
    }

    /// Returns a debug color based on the invalidation reason.
    fn color_for_reason(reason: InvalidateReason) -> Color {
        match reason {
            InvalidateReason::Paint => Color::new(1.0, 0.2, 0.2, 0.3),
            InvalidateReason::Layout => Color::new(0.2, 0.2, 1.0, 0.3),
            InvalidateReason::ChildOrder => Color::new(0.2, 1.0, 0.2, 0.3),
            InvalidateReason::Visibility => Color::new(1.0, 1.0, 0.2, 0.3),
            InvalidateReason::RenderTransform => Color::new(1.0, 0.5, 0.0, 0.3),
            InvalidateReason::AttributeOnly => Color::new(0.5, 0.5, 0.5, 0.2),
        }
    }

    /// Returns the right edge.
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// Returns the bottom edge.
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// Returns the area in square pixels.
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Returns true if this rect overlaps another.
    pub fn overlaps(&self, other: &DirtyRect) -> bool {
        self.x < other.right()
            && self.right() > other.x
            && self.y < other.bottom()
            && self.bottom() > other.y
    }

    /// Merges another rect into this one (union bounding box).
    pub fn merge(&mut self, other: &DirtyRect) {
        let new_x = self.x.min(other.x);
        let new_y = self.y.min(other.y);
        let new_right = self.right().max(other.right());
        let new_bottom = self.bottom().max(other.bottom());
        self.x = new_x;
        self.y = new_y;
        self.width = new_right - new_x;
        self.height = new_bottom - new_y;
    }

    /// Returns true if this rect fully contains another.
    pub fn contains(&self, other: &DirtyRect) -> bool {
        self.x <= other.x
            && self.y <= other.y
            && self.right() >= other.right()
            && self.bottom() >= other.bottom()
    }

    /// Returns true if a point is inside this rect.
    pub fn contains_point(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }

    /// Expands the rect by the given margin on all sides.
    pub fn expand(&mut self, margin: f32) {
        self.x -= margin;
        self.y -= margin;
        self.width += margin * 2.0;
        self.height += margin * 2.0;
    }
}

// ---------------------------------------------------------------------------
// DirtyRectTracking
// ---------------------------------------------------------------------------

/// Tracks which screen regions need repainting each frame.
///
/// Overlapping dirty rects are merged to reduce the number of clip/scissor
/// regions sent to the GPU. Clean regions are skipped entirely, which can
/// dramatically reduce paint work for UIs where only small areas change.
#[derive(Debug, Clone)]
pub struct DirtyRectTracking {
    /// Dirty rects accumulated for the current frame.
    pub current_rects: Vec<DirtyRect>,
    /// Dirty rects from the previous frame (for debug visualization).
    pub previous_rects: Vec<DirtyRect>,
    /// Screen width in pixels.
    pub screen_width: f32,
    /// Screen height in pixels.
    pub screen_height: f32,
    /// Whether the entire screen is dirty (full repaint).
    pub full_repaint: bool,
    /// Current frame number.
    pub current_frame: u64,
    /// Statistics: total dirty rect area this frame.
    pub dirty_area: f32,
    /// Statistics: total screen area.
    pub screen_area: f32,
    /// Maximum number of dirty rects before forcing a full repaint.
    pub max_rects: usize,
    /// Whether debug overlay visualization is enabled.
    pub debug_overlay_enabled: bool,
    /// Merge threshold: rects closer than this are merged.
    pub merge_distance: f32,
    /// History of dirty fractions for the last N frames.
    pub dirty_fraction_history: VecDeque<f32>,
    /// Maximum history length.
    pub max_history: usize,
}

impl DirtyRectTracking {
    /// Creates a new dirty rect tracker.
    pub fn new(screen_width: f32, screen_height: f32) -> Self {
        Self {
            current_rects: Vec::with_capacity(64),
            previous_rects: Vec::new(),
            screen_width,
            screen_height,
            full_repaint: true,
            current_frame: 0,
            dirty_area: 0.0,
            screen_area: screen_width * screen_height,
            max_rects: 32,
            debug_overlay_enabled: false,
            merge_distance: 8.0,
            dirty_fraction_history: VecDeque::with_capacity(120),
            max_history: 120,
        }
    }

    /// Sets the screen dimensions.
    pub fn set_screen_size(&mut self, width: f32, height: f32) {
        if (self.screen_width - width).abs() > 0.01
            || (self.screen_height - height).abs() > 0.01
        {
            self.screen_width = width;
            self.screen_height = height;
            self.screen_area = width * height;
            self.mark_full_repaint();
        }
    }

    /// Marks the entire screen as dirty.
    pub fn mark_full_repaint(&mut self) {
        self.full_repaint = true;
        self.current_rects.clear();
    }

    /// Adds a dirty rect to the current frame.
    pub fn add_dirty_rect(&mut self, rect: DirtyRect) {
        if self.full_repaint {
            return;
        }
        let mut new_rect = rect;
        new_rect.frame = self.current_frame;

        // Clamp to screen bounds.
        new_rect.x = new_rect.x.max(0.0);
        new_rect.y = new_rect.y.max(0.0);
        if new_rect.right() > self.screen_width {
            new_rect.width = self.screen_width - new_rect.x;
        }
        if new_rect.bottom() > self.screen_height {
            new_rect.height = self.screen_height - new_rect.y;
        }

        if new_rect.width <= 0.0 || new_rect.height <= 0.0 {
            return;
        }

        self.current_rects.push(new_rect);

        if self.current_rects.len() > self.max_rects {
            self.mark_full_repaint();
        }
    }

    /// Marks a widget's region as dirty.
    pub fn mark_widget_dirty(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        reason: InvalidateReason,
    ) {
        self.add_dirty_rect(DirtyRect::new(x, y, width, height, reason));
    }

    /// Merges overlapping dirty rects to reduce draw calls.
    pub fn merge_overlapping(&mut self) {
        if self.full_repaint || self.current_rects.len() <= 1 {
            return;
        }

        let mut merged = true;
        while merged {
            merged = false;
            let mut i = 0;
            while i < self.current_rects.len() {
                let mut j = i + 1;
                while j < self.current_rects.len() {
                    // Expand for proximity check.
                    let mut expanded_a = self.current_rects[i];
                    expanded_a.expand(self.merge_distance);

                    if expanded_a.overlaps(&self.current_rects[j]) {
                        let other = self.current_rects[j];
                        self.current_rects[i].merge(&other);
                        self.current_rects.swap_remove(j);
                        merged = true;
                    } else {
                        j += 1;
                    }
                }
                i += 1;
            }
        }
    }

    /// Prepares for a new frame: archives current rects and clears.
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
        self.previous_rects = self.current_rects.clone();
        self.current_rects.clear();
        self.full_repaint = false;
    }

    /// Finalizes the frame: merges rects and computes statistics.
    pub fn end_frame(&mut self) {
        self.merge_overlapping();
        self.dirty_area = if self.full_repaint {
            self.screen_area
        } else {
            self.current_rects.iter().map(|r| r.area()).sum()
        };

        let fraction = if self.screen_area > 0.0 {
            (self.dirty_area / self.screen_area).min(1.0)
        } else {
            0.0
        };
        self.dirty_fraction_history.push_back(fraction);
        if self.dirty_fraction_history.len() > self.max_history {
            self.dirty_fraction_history.pop_front();
        }
    }

    /// Returns the fraction of the screen that is dirty this frame.
    pub fn dirty_fraction(&self) -> f32 {
        if self.screen_area <= 0.0 {
            0.0
        } else {
            (self.dirty_area / self.screen_area).min(1.0)
        }
    }

    /// Returns the average dirty fraction over the history window.
    pub fn average_dirty_fraction(&self) -> f32 {
        if self.dirty_fraction_history.is_empty() {
            0.0
        } else {
            let sum: f32 = self.dirty_fraction_history.iter().sum();
            sum / self.dirty_fraction_history.len() as f32
        }
    }

    /// Returns true if a given rectangle intersects any dirty rect.
    pub fn is_region_dirty(&self, x: f32, y: f32, w: f32, h: f32) -> bool {
        if self.full_repaint {
            return true;
        }
        let query = DirtyRect::new(x, y, w, h, InvalidateReason::Paint);
        self.current_rects.iter().any(|r| r.overlaps(&query))
    }

    /// Returns true if nothing is dirty this frame.
    pub fn is_clean(&self) -> bool {
        !self.full_repaint && self.current_rects.is_empty()
    }

    /// Returns the number of dirty rects (after merging).
    pub fn rect_count(&self) -> usize {
        if self.full_repaint {
            1
        } else {
            self.current_rects.len()
        }
    }
}

// ---------------------------------------------------------------------------
// DrawElementBatch
// ---------------------------------------------------------------------------

/// Represents a batch of draw elements that share the same GPU state.
#[derive(Debug, Clone)]
pub struct DrawElementBatch {
    /// Texture ID used by all elements in this batch (0 = no texture).
    pub texture_id: u64,
    /// Clip rect applied to all elements ([x, y, w, h]).
    pub clip_rect: [f32; 4],
    /// Number of elements in this batch.
    pub element_count: u32,
    /// Number of vertices in this batch.
    pub vertex_count: u32,
    /// Number of indices in this batch.
    pub index_count: u32,
    /// Start index in the global vertex buffer.
    pub vertex_offset: u32,
    /// Start index in the global index buffer.
    pub index_offset: u32,
    /// Blend mode for this batch (0 = normal, 1 = additive, 2 = multiply).
    pub blend_mode: u8,
    /// Whether this batch uses SDF rendering.
    pub sdf: bool,
}

impl DrawElementBatch {
    /// Creates a new empty batch with the given state.
    pub fn new(texture_id: u64, clip_rect: [f32; 4]) -> Self {
        Self {
            texture_id,
            clip_rect,
            element_count: 0,
            vertex_count: 0,
            index_count: 0,
            vertex_offset: 0,
            index_offset: 0,
            blend_mode: 0,
            sdf: false,
        }
    }

    /// Returns true if another element can be added to this batch.
    pub fn can_merge(&self, texture_id: u64, clip_rect: &[f32; 4], blend_mode: u8, sdf: bool) -> bool {
        self.texture_id == texture_id
            && self.clip_rect == *clip_rect
            && self.blend_mode == blend_mode
            && self.sdf == sdf
    }

    /// Adds an element to this batch.
    pub fn add_element(&mut self, vertices: u32, indices: u32) {
        self.element_count += 1;
        self.vertex_count += vertices;
        self.index_count += indices;
    }
}

// ---------------------------------------------------------------------------
// BatchStatistics
// ---------------------------------------------------------------------------

/// Statistics about draw call batching.
#[derive(Debug, Clone, Default)]
pub struct BatchStatistics {
    /// Total draw calls before batching.
    pub draw_calls_before: u32,
    /// Total draw calls after batching.
    pub draw_calls_after: u32,
    /// Number of batches formed.
    pub batch_count: u32,
    /// Total elements across all batches.
    pub total_elements: u32,
    /// Total vertices across all batches.
    pub total_vertices: u32,
    /// Total indices across all batches.
    pub total_indices: u32,
    /// Number of GPU state changes.
    pub state_changes: u32,
    /// Number of texture binds.
    pub texture_binds: u32,
    /// Number of clip rect changes.
    pub clip_changes: u32,
    /// Time spent on batching in microseconds.
    pub batch_time_us: u64,
}

impl BatchStatistics {
    /// Returns the batch efficiency (1.0 = perfect batching, no wasted calls).
    pub fn efficiency(&self) -> f32 {
        if self.draw_calls_before == 0 {
            1.0
        } else {
            1.0 - (self.draw_calls_after as f32 / self.draw_calls_before as f32)
        }
    }

    /// Returns the average batch size (elements per draw call).
    pub fn average_batch_size(&self) -> f32 {
        if self.batch_count == 0 {
            0.0
        } else {
            self.total_elements as f32 / self.batch_count as f32
        }
    }
}

impl fmt::Display for BatchStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Batches: {} (from {} calls, {:.0}% reduction) | {} elements, {} verts | {} state changes",
            self.batch_count,
            self.draw_calls_before,
            self.efficiency() * 100.0,
            self.total_elements,
            self.total_vertices,
            self.state_changes,
        )
    }
}

// ---------------------------------------------------------------------------
// ElementBatching
// ---------------------------------------------------------------------------

/// Merges sequential draw elements that share the same GPU state into batches.
///
/// This reduces the number of draw calls submitted to the GPU, which is one of
/// the most impactful performance optimizations for UI rendering. Elements are
/// also sorted by texture to minimize texture bind calls.
#[derive(Debug, Clone)]
pub struct ElementBatching {
    /// Current list of batches.
    pub batches: Vec<DrawElementBatch>,
    /// Statistics for the current frame.
    pub stats: BatchStatistics,
    /// Whether to sort by texture before batching.
    pub sort_by_texture: bool,
    /// Maximum vertices per batch (GPU buffer size limit).
    pub max_vertices_per_batch: u32,
    /// Maximum indices per batch.
    pub max_indices_per_batch: u32,
    /// Whether batching is enabled.
    pub enabled: bool,
    /// History of batch counts for profiling.
    pub batch_count_history: VecDeque<u32>,
    /// Maximum history length.
    pub max_history: usize,
}

impl ElementBatching {
    /// Creates a new element batching system.
    pub fn new() -> Self {
        Self {
            batches: Vec::with_capacity(64),
            stats: BatchStatistics::default(),
            sort_by_texture: true,
            max_vertices_per_batch: 65536,
            max_indices_per_batch: 65536 * 3,
            enabled: true,
            batch_count_history: VecDeque::with_capacity(120),
            max_history: 120,
        }
    }

    /// Begins a new batching pass (clears previous results).
    pub fn begin(&mut self) {
        self.batches.clear();
        self.stats = BatchStatistics::default();
    }

    /// Submits a draw element for batching.
    pub fn submit_element(
        &mut self,
        texture_id: u64,
        clip_rect: [f32; 4],
        vertex_count: u32,
        index_count: u32,
        blend_mode: u8,
        sdf: bool,
    ) {
        self.stats.draw_calls_before += 1;

        if !self.enabled {
            let mut batch = DrawElementBatch::new(texture_id, clip_rect);
            batch.blend_mode = blend_mode;
            batch.sdf = sdf;
            batch.add_element(vertex_count, index_count);
            self.batches.push(batch);
            return;
        }

        // Try to extend the last batch.
        if let Some(last) = self.batches.last_mut() {
            if last.can_merge(texture_id, &clip_rect, blend_mode, sdf)
                && last.vertex_count + vertex_count <= self.max_vertices_per_batch
                && last.index_count + index_count <= self.max_indices_per_batch
            {
                last.add_element(vertex_count, index_count);
                return;
            }
        }

        // Start a new batch.
        let mut batch = DrawElementBatch::new(texture_id, clip_rect);
        batch.blend_mode = blend_mode;
        batch.sdf = sdf;
        batch.add_element(vertex_count, index_count);

        // Track state changes.
        if let Some(last) = self.batches.last() {
            if last.texture_id != texture_id {
                self.stats.texture_binds += 1;
            }
            if last.clip_rect != clip_rect {
                self.stats.clip_changes += 1;
            }
            self.stats.state_changes += 1;
        }

        self.batches.push(batch);
    }

    /// Finalizes the batching pass and computes statistics.
    pub fn finish(&mut self) {
        self.stats.batch_count = self.batches.len() as u32;
        self.stats.draw_calls_after = self.stats.batch_count;
        self.stats.total_elements = self.batches.iter().map(|b| b.element_count).sum();
        self.stats.total_vertices = self.batches.iter().map(|b| b.vertex_count).sum();
        self.stats.total_indices = self.batches.iter().map(|b| b.index_count).sum();

        self.batch_count_history
            .push_back(self.stats.batch_count);
        if self.batch_count_history.len() > self.max_history {
            self.batch_count_history.pop_front();
        }
    }

    /// Returns the average batch count over the history window.
    pub fn average_batch_count(&self) -> f32 {
        if self.batch_count_history.is_empty() {
            0.0
        } else {
            let sum: u32 = self.batch_count_history.iter().sum();
            sum as f32 / self.batch_count_history.len() as f32
        }
    }
}

impl Default for ElementBatching {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DesiredSizeCache
// ---------------------------------------------------------------------------

/// Cached desired-size result for a single widget.
#[derive(Debug, Clone)]
pub struct DesiredSizeEntry {
    /// The cached desired size.
    pub size: Vec2,
    /// The constraints used to compute this size.
    pub constraints: Vec2,
    /// Frame number when this entry was computed.
    pub frame: u64,
    /// Whether this entry is valid.
    pub valid: bool,
    /// Number of times this entry has been read without recomputation.
    pub hit_count: u64,
}

impl DesiredSizeEntry {
    /// Creates a new cache entry.
    pub fn new(size: Vec2, constraints: Vec2, frame: u64) -> Self {
        Self {
            size,
            constraints,
            frame,
            valid: true,
            hit_count: 0,
        }
    }

    /// Invalidates this entry.
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Returns true if this entry is usable for the given constraints.
    pub fn matches(&self, constraints: Vec2) -> bool {
        self.valid
            && (self.constraints.x - constraints.x).abs() < 0.01
            && (self.constraints.y - constraints.y).abs() < 0.01
    }
}

// ---------------------------------------------------------------------------
// WidgetCaching
// ---------------------------------------------------------------------------

/// Caches `ComputeDesiredSize` results for widgets to avoid redundant layout
/// computations.
///
/// In a two-pass layout system, each widget's desired size may be queried
/// multiple times per frame (once per layout pass). This cache stores the
/// result keyed by widget ID and constraints, and invalidates entries when
/// a `Layout` invalidation reason is received.
#[derive(Debug, Clone)]
pub struct WidgetCaching {
    /// Cached desired sizes keyed by widget ID index.
    pub cache: HashMap<u32, DesiredSizeEntry>,
    /// Statistics: cache lookups.
    pub lookups: u64,
    /// Statistics: cache hits.
    pub hits: u64,
    /// Statistics: cache misses.
    pub misses: u64,
    /// Statistics: invalidations.
    pub invalidations: u64,
    /// Maximum cache entries (LRU eviction beyond this).
    pub max_entries: usize,
    /// Whether caching is enabled.
    pub enabled: bool,
    /// Whether prepass caching is enabled (two-pass layout).
    pub prepass_enabled: bool,
    /// Prepass cache (separate from the main pass).
    pub prepass_cache: HashMap<u32, DesiredSizeEntry>,
}

impl WidgetCaching {
    /// Creates a new widget caching system.
    pub fn new() -> Self {
        Self {
            cache: HashMap::with_capacity(256),
            lookups: 0,
            hits: 0,
            misses: 0,
            invalidations: 0,
            max_entries: 4096,
            enabled: true,
            prepass_enabled: false,
            prepass_cache: HashMap::with_capacity(256),
        }
    }

    /// Looks up the cached desired size for a widget.
    pub fn get(&mut self, widget_index: u32, constraints: Vec2) -> Option<Vec2> {
        if !self.enabled {
            return None;
        }
        self.lookups += 1;
        if let Some(entry) = self.cache.get_mut(&widget_index) {
            if entry.matches(constraints) {
                self.hits += 1;
                entry.hit_count += 1;
                return Some(entry.size);
            }
        }
        self.misses += 1;
        None
    }

    /// Stores a computed desired size in the cache.
    pub fn put(&mut self, widget_index: u32, size: Vec2, constraints: Vec2, frame: u64) {
        if !self.enabled {
            return;
        }
        if self.cache.len() >= self.max_entries {
            self.evict_oldest();
        }
        self.cache
            .insert(widget_index, DesiredSizeEntry::new(size, constraints, frame));
    }

    /// Looks up the prepass cached desired size for a widget.
    pub fn get_prepass(&mut self, widget_index: u32, constraints: Vec2) -> Option<Vec2> {
        if !self.prepass_enabled {
            return None;
        }
        self.lookups += 1;
        if let Some(entry) = self.prepass_cache.get_mut(&widget_index) {
            if entry.matches(constraints) {
                self.hits += 1;
                entry.hit_count += 1;
                return Some(entry.size);
            }
        }
        self.misses += 1;
        None
    }

    /// Stores a prepass computed desired size in the cache.
    pub fn put_prepass(&mut self, widget_index: u32, size: Vec2, constraints: Vec2, frame: u64) {
        if !self.prepass_enabled {
            return;
        }
        self.prepass_cache
            .insert(widget_index, DesiredSizeEntry::new(size, constraints, frame));
    }

    /// Invalidates a specific widget's cached size.
    pub fn invalidate(&mut self, widget_index: u32) {
        self.invalidations += 1;
        if let Some(entry) = self.cache.get_mut(&widget_index) {
            entry.invalidate();
        }
        if let Some(entry) = self.prepass_cache.get_mut(&widget_index) {
            entry.invalidate();
        }
    }

    /// Invalidates all cached sizes.
    pub fn invalidate_all(&mut self) {
        for entry in self.cache.values_mut() {
            entry.invalidate();
        }
        for entry in self.prepass_cache.values_mut() {
            entry.invalidate();
        }
    }

    /// Removes the oldest cache entries to make room.
    fn evict_oldest(&mut self) {
        let target = self.max_entries / 2;
        let mut entries: Vec<(u32, u64)> = self
            .cache
            .iter()
            .map(|(k, v)| (*k, v.frame))
            .collect();
        entries.sort_by_key(|(_, frame)| *frame);
        let to_remove = entries.len().saturating_sub(target);
        for (key, _) in entries.into_iter().take(to_remove) {
            self.cache.remove(&key);
        }
    }

    /// Returns the cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.lookups as f64
        }
    }

    /// Resets statistics.
    pub fn reset_stats(&mut self) {
        self.lookups = 0;
        self.hits = 0;
        self.misses = 0;
        self.invalidations = 0;
    }

    /// Returns the total number of cached entries.
    pub fn entry_count(&self) -> usize {
        self.cache.len() + self.prepass_cache.len()
    }
}

impl Default for WidgetCaching {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WakeReason
// ---------------------------------------------------------------------------

/// Describes why the UI system woke from sleep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeReason {
    /// Mouse moved.
    MouseMove,
    /// Mouse button pressed.
    MouseButton,
    /// Key pressed.
    KeyPress,
    /// A timer fired.
    TimerFire,
    /// A widget was invalidated.
    Invalidation,
    /// Window was resized.
    WindowResize,
    /// Window gained focus.
    WindowFocus,
    /// External event (e.g., data update).
    External,
    /// Forced wake by application.
    Forced,
}

impl fmt::Display for WakeReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MouseMove => write!(f, "MouseMove"),
            Self::MouseButton => write!(f, "MouseButton"),
            Self::KeyPress => write!(f, "KeyPress"),
            Self::TimerFire => write!(f, "TimerFire"),
            Self::Invalidation => write!(f, "Invalidation"),
            Self::WindowResize => write!(f, "WindowResize"),
            Self::WindowFocus => write!(f, "WindowFocus"),
            Self::External => write!(f, "External"),
            Self::Forced => write!(f, "Forced"),
        }
    }
}

// ---------------------------------------------------------------------------
// SlateSleep
// ---------------------------------------------------------------------------

/// Manages the UI sleep state to reduce CPU usage when idle.
///
/// When nothing is animating, no input is occurring, and no timers are pending,
/// the UI system can skip rendering entirely. This reduces idle CPU usage to
/// near zero, which is important for editor applications that may be open for
/// hours at a time.
///
/// The sleep system tracks:
/// - Active animations (any animation = stay awake).
/// - Pending invalidations (any invalidation = stay awake).
/// - Input activity (mouse/key events = stay awake + grace period).
/// - Active timers (any pending timer = stay awake).
///
/// After a configurable idle timeout, the system enters sleep mode and stops
/// calling the render pipeline until a wake event occurs.
#[derive(Debug, Clone)]
pub struct SlateSleep {
    /// Whether the system is currently sleeping.
    pub sleeping: bool,
    /// Number of frames since the last wake event.
    pub idle_frames: u64,
    /// Number of frames to wait before entering sleep.
    pub idle_threshold: u64,
    /// Number of active animations.
    pub active_animations: u32,
    /// Number of pending invalidations.
    pub pending_invalidations: u32,
    /// Number of active timers.
    pub active_timers: u32,
    /// Whether there was input this frame.
    pub had_input: bool,
    /// Number of frames to stay awake after last input.
    pub input_grace_frames: u64,
    /// Frame counter since last input.
    pub frames_since_input: u64,
    /// The reason for the most recent wake.
    pub last_wake_reason: Option<WakeReason>,
    /// Total frames spent sleeping.
    pub total_sleep_frames: u64,
    /// Total frames spent awake.
    pub total_awake_frames: u64,
    /// Whether sleep mode is enabled.
    pub enabled: bool,
    /// Minimum frames to stay awake after waking.
    pub min_awake_frames: u64,
    /// Frame counter since last wake.
    pub frames_since_wake: u64,
    /// Callback-like flag: set when wake should trigger a full repaint.
    pub repaint_on_wake: bool,
}

impl SlateSleep {
    /// Creates a new sleep manager.
    pub fn new() -> Self {
        Self {
            sleeping: false,
            idle_frames: 0,
            idle_threshold: 60,
            active_animations: 0,
            pending_invalidations: 0,
            active_timers: 0,
            had_input: false,
            input_grace_frames: 30,
            frames_since_input: 0,
            last_wake_reason: None,
            total_sleep_frames: 0,
            total_awake_frames: 0,
            enabled: true,
            min_awake_frames: 10,
            frames_since_wake: 0,
            repaint_on_wake: true,
        }
    }

    /// Called each frame to update the sleep state.
    pub fn update(&mut self) {
        if !self.enabled {
            self.sleeping = false;
            return;
        }

        if self.had_input {
            self.frames_since_input = 0;
            self.had_input = false;
        } else {
            self.frames_since_input += 1;
        }

        if self.sleeping {
            self.total_sleep_frames += 1;
        } else {
            self.total_awake_frames += 1;
            self.frames_since_wake += 1;
        }

        // Determine if we should sleep.
        let should_be_awake = self.active_animations > 0
            || self.pending_invalidations > 0
            || self.active_timers > 0
            || self.frames_since_input < self.input_grace_frames
            || self.frames_since_wake < self.min_awake_frames;

        if should_be_awake {
            self.sleeping = false;
            self.idle_frames = 0;
        } else {
            self.idle_frames += 1;
            if self.idle_frames >= self.idle_threshold {
                self.sleeping = true;
            }
        }
    }

    /// Wakes the system with the given reason.
    pub fn wake(&mut self, reason: WakeReason) {
        if self.sleeping {
            self.last_wake_reason = Some(reason);
            self.frames_since_wake = 0;
        }
        self.sleeping = false;
        self.idle_frames = 0;

        match reason {
            WakeReason::MouseMove | WakeReason::MouseButton | WakeReason::KeyPress => {
                self.had_input = true;
            }
            _ => {}
        }
    }

    /// Notifies the system that an animation started.
    pub fn on_animation_start(&mut self) {
        self.active_animations += 1;
        if self.sleeping {
            self.wake(WakeReason::Invalidation);
        }
    }

    /// Notifies the system that an animation ended.
    pub fn on_animation_end(&mut self) {
        self.active_animations = self.active_animations.saturating_sub(1);
    }

    /// Notifies the system of a pending invalidation.
    pub fn on_invalidation(&mut self) {
        self.pending_invalidations += 1;
        if self.sleeping {
            self.wake(WakeReason::Invalidation);
        }
    }

    /// Clears the pending invalidation count (after processing).
    pub fn clear_invalidations(&mut self) {
        self.pending_invalidations = 0;
    }

    /// Sets the active timer count.
    pub fn set_active_timers(&mut self, count: u32) {
        self.active_timers = count;
        if count > 0 && self.sleeping {
            self.wake(WakeReason::TimerFire);
        }
    }

    /// Returns the fraction of time spent sleeping.
    pub fn sleep_fraction(&self) -> f64 {
        let total = self.total_sleep_frames + self.total_awake_frames;
        if total == 0 {
            0.0
        } else {
            self.total_sleep_frames as f64 / total as f64
        }
    }

    /// Returns true if the render pipeline should be skipped this frame.
    pub fn should_skip_render(&self) -> bool {
        self.sleeping && self.enabled
    }

    /// Resets statistics.
    pub fn reset_stats(&mut self) {
        self.total_sleep_frames = 0;
        self.total_awake_frames = 0;
    }
}

impl Default for SlateSleep {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FrameTimings
// ---------------------------------------------------------------------------

/// Timing data for a single frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameTimings {
    /// Total frame time in microseconds.
    pub total_us: u64,
    /// Time spent on layout in microseconds.
    pub layout_us: u64,
    /// Time spent on painting in microseconds.
    pub paint_us: u64,
    /// Time spent on event dispatch in microseconds.
    pub event_dispatch_us: u64,
    /// Time spent on batching in microseconds.
    pub batch_us: u64,
    /// Time spent on GPU submission in microseconds.
    pub gpu_submit_us: u64,
}

impl FrameTimings {
    /// Returns the overhead (total - layout - paint - events).
    pub fn overhead_us(&self) -> u64 {
        self.total_us
            .saturating_sub(self.layout_us)
            .saturating_sub(self.paint_us)
            .saturating_sub(self.event_dispatch_us)
            .saturating_sub(self.batch_us)
    }
}

// ---------------------------------------------------------------------------
// PerformanceStats
// ---------------------------------------------------------------------------

/// Comprehensive real-time performance statistics for the UI system.
///
/// Collects per-frame timings, widget counts, draw call information, cache hit
/// rates, and memory usage. Maintains a rolling history of 120 frames for
/// sparkline / graph visualization.
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Rolling history of frame timings.
    pub frame_history: VecDeque<FrameTimings>,
    /// Maximum history length.
    pub max_history: usize,
    /// Current frame number.
    pub frame_number: u64,
    /// Total widget count.
    pub widget_count: u32,
    /// Visible widget count.
    pub visible_widget_count: u32,
    /// Invalidated widget count this frame.
    pub invalidated_widget_count: u32,
    /// Cached widget count (layout cache hits).
    pub cached_widget_count: u32,
    /// Total draw calls.
    pub draw_call_count: u32,
    /// Total vertices submitted.
    pub vertex_count: u32,
    /// Total elements submitted.
    pub element_count: u32,
    /// Layout cache hit rate.
    pub layout_cache_hit_rate: f64,
    /// Render cache hit rate.
    pub render_cache_hit_rate: f64,
    /// Total memory used by caches in bytes.
    pub cache_memory_bytes: u64,
    /// Total memory used by widgets in bytes (estimated).
    pub widget_memory_bytes: u64,
    /// Whether the system is sleeping.
    pub is_sleeping: bool,
    /// Dirty fraction of the screen.
    pub dirty_fraction: f32,
    /// Current FPS estimate.
    pub fps: f32,
    /// Time of the last frame start (for FPS calculation).
    pub last_frame_time: Option<Instant>,
    /// Exponential moving average of frame time.
    pub avg_frame_time_us: f64,
}

impl PerformanceStats {
    /// Creates a new performance stats tracker.
    pub fn new() -> Self {
        Self {
            frame_history: VecDeque::with_capacity(120),
            max_history: 120,
            frame_number: 0,
            widget_count: 0,
            visible_widget_count: 0,
            invalidated_widget_count: 0,
            cached_widget_count: 0,
            draw_call_count: 0,
            vertex_count: 0,
            element_count: 0,
            layout_cache_hit_rate: 0.0,
            render_cache_hit_rate: 0.0,
            cache_memory_bytes: 0,
            widget_memory_bytes: 0,
            is_sleeping: false,
            dirty_fraction: 0.0,
            fps: 0.0,
            last_frame_time: None,
            avg_frame_time_us: 0.0,
        }
    }

    /// Records a new frame's timings.
    pub fn record_frame(&mut self, timings: FrameTimings) {
        self.frame_number += 1;
        self.frame_history.push_back(timings);
        if self.frame_history.len() > self.max_history {
            self.frame_history.pop_front();
        }

        // Update FPS estimate.
        let now = Instant::now();
        if let Some(last) = self.last_frame_time {
            let dt = now.duration_since(last).as_secs_f32();
            if dt > 0.0 {
                self.fps = self.fps * 0.9 + (1.0 / dt) * 0.1;
            }
        }
        self.last_frame_time = Some(now);

        // Update average frame time.
        self.avg_frame_time_us = self.avg_frame_time_us * 0.95 + timings.total_us as f64 * 0.05;
    }

    /// Returns the average layout time over the history window.
    pub fn avg_layout_us(&self) -> f64 {
        if self.frame_history.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.frame_history.iter().map(|f| f.layout_us).sum();
        sum as f64 / self.frame_history.len() as f64
    }

    /// Returns the average paint time over the history window.
    pub fn avg_paint_us(&self) -> f64 {
        if self.frame_history.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.frame_history.iter().map(|f| f.paint_us).sum();
        sum as f64 / self.frame_history.len() as f64
    }

    /// Returns the average event dispatch time.
    pub fn avg_event_us(&self) -> f64 {
        if self.frame_history.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.frame_history.iter().map(|f| f.event_dispatch_us).sum();
        sum as f64 / self.frame_history.len() as f64
    }

    /// Returns the peak frame time in the history.
    pub fn peak_frame_us(&self) -> u64 {
        self.frame_history.iter().map(|f| f.total_us).max().unwrap_or(0)
    }

    /// Returns the minimum frame time in the history.
    pub fn min_frame_us(&self) -> u64 {
        self.frame_history
            .iter()
            .map(|f| f.total_us)
            .min()
            .unwrap_or(0)
    }

    /// Returns a formatted summary string.
    pub fn summary(&self) -> String {
        format!(
            "FPS: {:.0} | Frame: {:.1}ms avg, {:.1}ms peak | Layout: {:.1}us | Paint: {:.1}us | \
             Widgets: {}/{} | Draws: {} | Verts: {} | Cache: {:.0}%/{:.0}% | Dirty: {:.0}%",
            self.fps,
            self.avg_frame_time_us / 1000.0,
            self.peak_frame_us() as f64 / 1000.0,
            self.avg_layout_us(),
            self.avg_paint_us(),
            self.visible_widget_count,
            self.widget_count,
            self.draw_call_count,
            self.vertex_count,
            self.layout_cache_hit_rate * 100.0,
            self.render_cache_hit_rate * 100.0,
            self.dirty_fraction * 100.0,
        )
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.frame_history.clear();
        self.frame_number = 0;
        self.fps = 0.0;
        self.avg_frame_time_us = 0.0;
        self.last_frame_time = None;
    }
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PerformanceManager
// ---------------------------------------------------------------------------

/// Top-level manager that coordinates all performance subsystems.
///
/// This is the main entry point for performance optimization in the UI system.
/// It owns and coordinates the invalidation panels, retainer boxes, dirty rect
/// tracking, element batching, widget caching, and sleep system.
#[derive(Debug, Clone)]
pub struct PerformanceManager {
    /// Active invalidation panels.
    pub invalidation_panels: HashMap<u32, InvalidationPanel>,
    /// Active retainer boxes.
    pub retainer_boxes: HashMap<u64, RetainerBox>,
    /// Dirty rect tracker.
    pub dirty_rects: DirtyRectTracking,
    /// Element batching system.
    pub batching: ElementBatching,
    /// Widget size cache.
    pub widget_cache: WidgetCaching,
    /// Sleep manager.
    pub sleep: SlateSleep,
    /// Performance statistics.
    pub stats: PerformanceStats,
    /// Current frame number.
    pub frame_number: u64,
    /// Whether performance tracking is enabled.
    pub enabled: bool,
}

impl PerformanceManager {
    /// Creates a new performance manager.
    pub fn new(screen_width: f32, screen_height: f32) -> Self {
        Self {
            invalidation_panels: HashMap::new(),
            retainer_boxes: HashMap::new(),
            dirty_rects: DirtyRectTracking::new(screen_width, screen_height),
            batching: ElementBatching::new(),
            widget_cache: WidgetCaching::new(),
            sleep: SlateSleep::new(),
            stats: PerformanceStats::new(),
            frame_number: 0,
            enabled: true,
        }
    }

    /// Begins a new frame.
    pub fn begin_frame(&mut self) {
        self.frame_number += 1;
        self.dirty_rects.begin_frame();
        self.batching.begin();
        self.sleep.update();
    }

    /// Ends the current frame and collects statistics.
    pub fn end_frame(&mut self, timings: FrameTimings) {
        self.dirty_rects.end_frame();
        self.batching.finish();
        self.sleep.clear_invalidations();

        self.stats.draw_call_count = self.batching.stats.draw_calls_after;
        self.stats.vertex_count = self.batching.stats.total_vertices;
        self.stats.element_count = self.batching.stats.total_elements;
        self.stats.layout_cache_hit_rate = self.widget_cache.hit_rate();
        self.stats.dirty_fraction = self.dirty_rects.dirty_fraction();
        self.stats.is_sleeping = self.sleep.sleeping;

        self.stats.record_frame(timings);
    }

    /// Returns true if rendering should be skipped this frame.
    pub fn should_skip_render(&self) -> bool {
        self.sleep.should_skip_render()
    }

    /// Registers a new invalidation panel.
    pub fn register_panel(&mut self, widget_index: u32, panel: InvalidationPanel) {
        self.invalidation_panels.insert(widget_index, panel);
    }

    /// Removes an invalidation panel.
    pub fn unregister_panel(&mut self, widget_index: u32) {
        self.invalidation_panels.remove(&widget_index);
    }

    /// Registers a new retainer box.
    pub fn register_retainer(&mut self, id: u64, retainer: RetainerBox) {
        self.retainer_boxes.insert(id, retainer);
    }

    /// Removes a retainer box.
    pub fn unregister_retainer(&mut self, id: u64) {
        self.retainer_boxes.remove(&id);
    }

    /// Invalidates a widget, propagating to its invalidation panel.
    pub fn invalidate_widget(&mut self, widget_index: u32, reason: InvalidateReason) {
        self.widget_cache.invalidate(widget_index);
        if let Some(panel) = self.invalidation_panels.get_mut(&widget_index) {
            panel.invalidate(reason);
        }
        self.sleep.on_invalidation();
    }

    /// Sets the screen size (triggers full repaint).
    pub fn set_screen_size(&mut self, width: f32, height: f32) {
        self.dirty_rects.set_screen_size(width, height);
    }

    /// Returns total memory used by all caches.
    pub fn total_cache_memory(&self) -> u64 {
        let panel_mem: u64 = self
            .invalidation_panels
            .values()
            .map(|p| p.memory_usage())
            .sum();
        let retainer_mem: u64 = self
            .retainer_boxes
            .values()
            .map(|r| r.memory_usage())
            .sum();
        panel_mem + retainer_mem
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalidate_reason_severity() {
        assert!(InvalidateReason::Layout.severity() > InvalidateReason::Paint.severity());
        assert!(InvalidateReason::Paint.severity() > InvalidateReason::AttributeOnly.severity());
    }

    #[test]
    fn test_cache_statistics_hit_rate() {
        let mut stats = CacheStatistics::new();
        stats.total_lookups = 100;
        stats.cache_hits = 80;
        assert!((stats.hit_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_offscreen_buffer_lifecycle() {
        let mut buf = OffscreenBuffer::new(1, 100, 100, 1.0);
        assert!(!buf.valid);
        buf.validate(1);
        assert!(buf.valid);
        buf.invalidate();
        assert!(!buf.valid);
    }

    #[test]
    fn test_virtualized_list_uniform() {
        let mut list = VirtualizedListCore::new(10000, 24.0);
        list.set_viewport(0.0, 600.0);
        list.update(0.016);
        // With 600px viewport and 24px rows, about 25 rows visible + overscan.
        assert!(list.last_visible - list.first_visible > 20);
        assert!(list.last_visible - list.first_visible < 40);
    }

    #[test]
    fn test_virtualized_list_scroll_to_index() {
        let mut list = VirtualizedListCore::new(1000, 24.0);
        list.set_viewport(0.0, 600.0);
        list.scroll_to_index(500, false);
        assert!(list.scroll.offset > 0.0);
    }

    #[test]
    fn test_dirty_rect_merge() {
        let mut tracker = DirtyRectTracking::new(1920.0, 1080.0);
        tracker.begin_frame();
        tracker.add_dirty_rect(DirtyRect::new(10.0, 10.0, 50.0, 50.0, InvalidateReason::Paint));
        tracker.add_dirty_rect(DirtyRect::new(20.0, 20.0, 50.0, 50.0, InvalidateReason::Paint));
        tracker.merge_overlapping();
        // The two overlapping rects should merge into one.
        assert_eq!(tracker.current_rects.len(), 1);
    }

    #[test]
    fn test_element_batching() {
        let mut batcher = ElementBatching::new();
        batcher.begin();
        // Same state => should merge.
        batcher.submit_element(1, [0.0, 0.0, 100.0, 100.0], 4, 6, 0, false);
        batcher.submit_element(1, [0.0, 0.0, 100.0, 100.0], 4, 6, 0, false);
        batcher.submit_element(1, [0.0, 0.0, 100.0, 100.0], 4, 6, 0, false);
        batcher.finish();
        assert_eq!(batcher.stats.batch_count, 1);
        assert_eq!(batcher.stats.total_elements, 3);
    }

    #[test]
    fn test_element_batching_different_texture() {
        let mut batcher = ElementBatching::new();
        batcher.begin();
        batcher.submit_element(1, [0.0, 0.0, 100.0, 100.0], 4, 6, 0, false);
        batcher.submit_element(2, [0.0, 0.0, 100.0, 100.0], 4, 6, 0, false);
        batcher.finish();
        assert_eq!(batcher.stats.batch_count, 2);
    }

    #[test]
    fn test_widget_caching() {
        let mut cache = WidgetCaching::new();
        let constraints = Vec2::new(200.0, 100.0);
        assert!(cache.get(1, constraints).is_none());
        cache.put(1, Vec2::new(150.0, 80.0), constraints, 1);
        let result = cache.get(1, constraints);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), Vec2::new(150.0, 80.0));
    }

    #[test]
    fn test_widget_caching_invalidation() {
        let mut cache = WidgetCaching::new();
        let constraints = Vec2::new(200.0, 100.0);
        cache.put(1, Vec2::new(150.0, 80.0), constraints, 1);
        cache.invalidate(1);
        assert!(cache.get(1, constraints).is_none());
    }

    #[test]
    fn test_slate_sleep_lifecycle() {
        let mut sleep = SlateSleep::new();
        sleep.idle_threshold = 3;
        assert!(!sleep.sleeping);

        // Simulate idle frames.
        for _ in 0..10 {
            sleep.update();
        }
        assert!(sleep.sleeping);

        // Wake up.
        sleep.wake(WakeReason::MouseMove);
        assert!(!sleep.sleeping);
    }

    #[test]
    fn test_scroll_state() {
        let mut scroll = ScrollState::new();
        scroll.set_viewport(600.0, 10000.0);
        assert_eq!(scroll.max_offset, 9400.0);
        scroll.scroll_by(100.0);
        scroll.update(1.0);
        assert!(scroll.offset > 0.0);
    }

    #[test]
    fn test_retainer_box_phase() {
        let mut retainer = RetainerBox::new(UIId::INVALID, 4).with_phase(2);
        assert_eq!(retainer.phase, 2);
        // Frame 0 => counter=1, 1%4 != 2, but force_render=true.
        assert!(retainer.tick());
        retainer.end_render(1, 100);
        // Frame 1 => counter=2, 2%4 == 2.
        assert!(retainer.tick());
    }

    #[test]
    fn test_performance_stats() {
        let mut stats = PerformanceStats::new();
        for i in 0..10 {
            stats.record_frame(FrameTimings {
                total_us: 16000,
                layout_us: 2000,
                paint_us: 8000,
                event_dispatch_us: 1000,
                batch_us: 500,
                gpu_submit_us: 2000,
            });
        }
        assert!(stats.avg_layout_us() > 1900.0);
        assert!(stats.avg_paint_us() > 7900.0);
    }
}
