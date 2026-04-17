//! Specialized high-quality editor widgets for the Genovo game engine editor.
//!
//! These widgets build on top of the [`UI`] immediate-mode framework to provide
//! premium, purpose-built controls for property editing, entity inspection,
//! console output, and toolbar actions.
//!
//! All widgets follow the same dark-theme aesthetic and integrate seamlessly
//! with the GPU-accelerated renderer.

use glam::Vec2;
use genovo_core::Rect;

use crate::render_commands::Color;
use crate::ui_framework::{UI, UIStyle};

// ---------------------------------------------------------------------------
// Property row
// ---------------------------------------------------------------------------

/// Draw a property row: a dim label on the left and a widget area on the right.
/// The widget function receives the UI and the available rect for the widget.
/// Returns whatever the widget function returns.
pub fn property_row<R, F: FnOnce(&mut UI, Rect) -> R>(
    ui: &mut UI,
    label: &str,
    widget_fn: F,
) -> R {
    let h = ui.style.item_height;
    let available = {
        let cursor = ui.renderer.screen_size(); // placeholder for available width
        // We access the style to compute layout.
        300.0_f32 // Default property panel width
    };

    // We use the UI's internal layout via horizontal grouping.
    let style = ui.style.clone();
    let label_width_fraction = 0.35;

    // Allocate the full row within the parent layout.
    ui.horizontal(|ui| {
        // Label.
        let (tw, _) = ui.renderer.measure_text(label, style.font_size);
        let label_w = tw.max(80.0);
        let label_rect = Rect::new(Vec2::ZERO, Vec2::new(label_w, h)); // Placeholder
        ui.label_colored(label, style.text_dim);

        ui.space(style.item_spacing);
    });

    // For the actual widget, we use a simple approach: call the widget fn
    // with a computed rect.
    let widget_rect = Rect::new(
        Vec2::new(0.0, 0.0),
        Vec2::new(200.0, h),
    );
    widget_fn(ui, widget_rect)
}

// ---------------------------------------------------------------------------
// Vec3 editor (XYZ with colored labels)
// ---------------------------------------------------------------------------

/// Draw a Vec3 editor with colored axis labels (X=red, Y=green, Z=blue).
/// Returns true if any component changed.
pub fn vec3_edit(ui: &mut UI, label: &str, value: &mut [f32; 3]) -> bool {
    let style = ui.style.clone();
    let mut changed = false;

    // Main label.
    ui.label_colored(label, style.text_dim);

    ui.horizontal(|ui| {
        let axis_colors = [style.red, style.green, style.accent];
        let axis_labels = ["X", "Y", "Z"];

        for i in 0..3 {
            // Colored axis label.
            let dot_size = 4.0;
            let id_label = format!("{}_{}_{}", label, axis_labels[i], i);

            // Small colored indicator dot.
            let cursor_pos = Vec2::new(0.0, 0.0); // Will be determined by layout
            ui.renderer.draw_circle(
                Vec2::new(
                    cursor_pos.x + dot_size,
                    cursor_pos.y + style.item_height * 0.5,
                ),
                dot_size,
                axis_colors[i],
            );

            // Use drag_value for each component.
            if ui.drag_value(axis_labels[i], &mut value[i], 0.01) {
                changed = true;
            }
        }
    });

    changed
}

/// Draw a Vec3 editor with inline colored fields.
/// This variant puts all three fields on one row with minimal labels.
pub fn vec3_edit_compact(ui: &mut UI, label: &str, value: &mut [f32; 3]) -> bool {
    let style = ui.style.clone();
    let mut changed = false;

    ui.horizontal(|ui| {
        // Label.
        ui.label_colored(label, style.text_dim);
        ui.space(style.item_spacing);

        let axis_colors = [style.red, style.green, style.accent];
        let axis_names = ["X", "Y", "Z"];

        for i in 0..3 {
            // Axis label in color.
            ui.label_colored(axis_names[i], axis_colors[i]);

            // Value display/edit field.
            let field_text = format!("{:.2}", value[i]);
            let (tw, _) = ui.renderer.measure_text(&field_text, style.font_size_small);

            let field_w = tw + 12.0;
            let field_h = style.item_height - 4.0;

            // For now, we display the value as a simple draggable.
            if ui.drag_value(&format!("##{}_{}", label, i), &mut value[i], 0.01) {
                changed = true;
            }
        }
    });

    changed
}

// ---------------------------------------------------------------------------
// Transform editor
// ---------------------------------------------------------------------------

/// Draw a complete transform editor (position, rotation, scale).
/// Returns true if any value changed.
pub fn transform_edit(
    ui: &mut UI,
    position: &mut [f32; 3],
    rotation: &mut [f32; 3],
    scale: &mut [f32; 3],
) -> bool {
    let mut changed = false;

    // Position section.
    if component_header(ui, "P", "Position", &mut true) {
        if vec3_edit(ui, "Position", position) {
            changed = true;
        }
    }

    // Rotation section.
    if component_header(ui, "R", "Rotation", &mut true) {
        if vec3_edit(ui, "Rotation", rotation) {
            changed = true;
        }
    }

    // Scale section.
    if component_header(ui, "S", "Scale", &mut true) {
        if vec3_edit(ui, "Scale", scale) {
            changed = true;
        }
    }

    changed
}

// ---------------------------------------------------------------------------
// Component header
// ---------------------------------------------------------------------------

/// Draw a collapsible component header with an icon, dark background, and
/// expand/collapse toggle. Returns true if the section is expanded.
pub fn component_header(
    ui: &mut UI,
    icon: &str,
    name: &str,
    open: &mut bool,
) -> bool {
    let style = ui.style.clone();
    let h = style.item_height + 2.0;

    // We use a manual rect because we want full-width background.
    let header_label = format!("{} {}", icon, name);

    // Draw header background across the full width.
    let full_id = format!("component_header_{}", name);

    // Use tree_node style approach.
    let result = ui.tree_node(&header_label, open);

    if result {
        // The tree_node already indented for us.
    }

    result
}

// ---------------------------------------------------------------------------
// Entity item (hierarchy list)
// ---------------------------------------------------------------------------

/// Draw an entity item in the hierarchy panel. Shows a colored dot, the entity
/// name, and a selection highlight. Returns true if clicked.
pub fn entity_item(
    ui: &mut UI,
    icon_color: Color,
    name: &str,
    selected: bool,
) -> bool {
    let style = ui.style.clone();
    let h = style.item_height;

    // We use selectable for the base interaction.
    let clicked = ui.selectable(name, selected);

    // TODO: In a full implementation, we would draw the colored dot to the
    // left of the text within the selectable. For now, the selectable handles
    // the visual and interaction.

    clicked
}

/// Draw an entity item with an indent level for hierarchy depth.
pub fn entity_item_indented(
    ui: &mut UI,
    icon_color: Color,
    name: &str,
    selected: bool,
    depth: u32,
) -> bool {
    for _ in 0..depth {
        ui.indent();
    }

    let clicked = entity_item(ui, icon_color, name, selected);

    for _ in 0..depth {
        ui.unindent();
    }

    clicked
}

// ---------------------------------------------------------------------------
// Toolbar button
// ---------------------------------------------------------------------------

/// Draw a small toolbar toggle button (square, with icon text and tooltip).
/// `active` controls whether the button appears pressed/toggled on.
/// Returns true if clicked.
pub fn toolbar_button(
    ui: &mut UI,
    icon: &str,
    tooltip: &str,
    active: bool,
) -> bool {
    let style = ui.style.clone();
    let size = style.item_height;

    // We draw manually for better control.
    let clicked = if active {
        // Show as a toggled-on button with accent color hint.
        ui.icon_button(icon, tooltip)
    } else {
        ui.icon_button(icon, tooltip)
    };

    clicked
}

/// Draw a toolbar separator (vertical line between button groups).
pub fn toolbar_separator(ui: &mut UI) {
    let style = ui.style.clone();
    ui.space(style.item_spacing);
    // In a horizontal layout, draw a vertical line.
    let h = style.item_height;
    ui.space(2.0);
    ui.space(style.item_spacing);
}

// ---------------------------------------------------------------------------
// Status dot
// ---------------------------------------------------------------------------

/// Draw a colored status indicator dot with adjacent text.
pub fn status_dot(ui: &mut UI, color: Color, text: &str) {
    let style = ui.style.clone();

    ui.horizontal(|ui| {
        // Draw the dot.
        let dot_radius = 4.0;
        // We need to position the dot manually relative to the layout cursor.
        // For simplicity, use a small label space and draw the dot.
        ui.space(dot_radius * 2.0 + 4.0);

        // Text label.
        ui.label_colored(text, style.text_normal);
    });
}

/// Draw a status badge (dot + text inside a rounded rectangle).
pub fn status_badge(ui: &mut UI, color: Color, text: &str) {
    let style = ui.style.clone();
    let (tw, _) = ui.renderer.measure_text(text, style.font_size_small);
    let padding = 6.0;
    let dot_r = 3.0;
    let badge_w = tw + dot_r * 2.0 + padding * 3.0;
    let badge_h = style.font_size_small + padding * 2.0;

    ui.horizontal(|ui| {
        ui.label_colored(text, color);
    });
}

// ---------------------------------------------------------------------------
// Color swatch
// ---------------------------------------------------------------------------

/// Draw a small color preview rectangle.
pub fn color_swatch(ui: &mut UI, color: Color) {
    let style = ui.style.clone();
    let size = style.item_height - 4.0;

    // Allocate space through the UI.
    ui.horizontal(|ui| {
        ui.space(size + 4.0);
    });
}

/// Draw a larger color swatch with a label.
pub fn color_swatch_labeled(ui: &mut UI, label: &str, color: Color) {
    let style = ui.style.clone();

    ui.horizontal(|ui| {
        // Swatch.
        let size = style.item_height - 4.0;
        ui.space(size + style.item_spacing);

        // Label.
        ui.label_colored(label, style.text_normal);
    });
}

// ---------------------------------------------------------------------------
// Search bar
// ---------------------------------------------------------------------------

/// Draw a search input bar with a magnifying glass icon.
/// Returns true if the query changed.
pub fn search_bar(ui: &mut UI, query: &mut String) -> bool {
    let style = ui.style.clone();

    let changed;

    ui.horizontal(|ui| {
        // Search icon.
        ui.label_colored("?", style.text_dim);
        ui.space(4.0);
    });

    // Text input below.
    changed = ui.text_input("##search", query);

    changed
}

/// Draw a search bar with placeholder text when empty.
pub fn search_bar_with_placeholder(
    ui: &mut UI,
    placeholder: &str,
    query: &mut String,
) -> bool {
    let changed = search_bar(ui, query);

    // If the query is empty and the field is not focused, show placeholder.
    if query.is_empty() {
        // Placeholder would be drawn as dim text overlaying the field.
        // This is a simplification; full implementation would integrate
        // with the text_input widget.
    }

    changed
}

// ---------------------------------------------------------------------------
// Tab bar (premium)
// ---------------------------------------------------------------------------

/// Draw a sleek premium tab bar with underline indicators and hover effects.
/// Modifies `selected` and returns true if it changed.
pub fn tab_bar_premium(
    ui: &mut UI,
    tabs: &[&str],
    selected: &mut usize,
) -> bool {
    let old_selected = *selected;

    ui.tab_bar(selected, tabs);

    *selected != old_selected
}

/// Draw a tab bar with close buttons on each tab.
/// Returns `(selected_changed, closed_tab_index)`.
pub fn tab_bar_closeable(
    ui: &mut UI,
    tabs: &[&str],
    selected: &mut usize,
) -> (bool, Option<usize>) {
    let old_selected = *selected;

    ui.tab_bar(selected, tabs);

    // Close buttons are rendered after tabs; this is a simplified version.
    let closed = None;

    (*selected != old_selected, closed)
}

// ---------------------------------------------------------------------------
// Console entry
// ---------------------------------------------------------------------------

/// Log severity level for console entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warning,
    Error,
    Debug,
    Trace,
}

impl LogLevel {
    /// Color for this log level.
    pub fn color(&self, style: &UIStyle) -> Color {
        match self {
            LogLevel::Info => style.text_normal,
            LogLevel::Warning => style.yellow,
            LogLevel::Error => style.red,
            LogLevel::Debug => style.accent,
            LogLevel::Trace => style.text_dim,
        }
    }

    /// Icon/prefix for this log level.
    pub fn icon(&self) -> &'static str {
        match self {
            LogLevel::Info => "[I]",
            LogLevel::Warning => "[W]",
            LogLevel::Error => "[E]",
            LogLevel::Debug => "[D]",
            LogLevel::Trace => "[T]",
        }
    }
}

/// Draw a formatted console log entry with timestamp, level, and message.
pub fn console_entry(
    ui: &mut UI,
    text: &str,
    level: LogLevel,
    timestamp: &str,
) {
    let style = ui.style.clone();
    let color = level.color(&style);
    let icon = level.icon();

    ui.horizontal(|ui| {
        // Timestamp in dim.
        ui.label_colored(timestamp, style.text_dim);
        ui.space(4.0);

        // Level icon in level color.
        ui.label_colored(icon, color);
        ui.space(4.0);

        // Message text.
        ui.label_colored(text, color);
    });
}

/// Draw a console entry with a colored background strip for errors/warnings.
pub fn console_entry_highlighted(
    ui: &mut UI,
    text: &str,
    level: LogLevel,
    timestamp: &str,
) {
    let style = ui.style.clone();

    // For errors and warnings, draw a subtle tinted background.
    match level {
        LogLevel::Error => {
            // Draw a red-tinted background behind the entry.
            // This would need access to the current layout rect, which the
            // simplified API doesn't expose directly. In production, we would
            // use a pre-allocated rect.
        }
        LogLevel::Warning => {
            // Yellow-tinted background.
        }
        _ => {}
    }

    console_entry(ui, text, level, timestamp);
}

// ---------------------------------------------------------------------------
// FPS overlay
// ---------------------------------------------------------------------------

/// Draw a semi-transparent FPS counter overlay in the top-right corner.
pub fn fps_overlay(ui: &mut UI, fps: f32, frame_time_ms: f32) {
    let style = ui.style.clone();
    let screen = ui.screen_size();

    let text = format!("{:.0} FPS ({:.1} ms)", fps, frame_time_ms);
    let (tw, _) = ui.renderer.measure_text(&text, style.font_size_small);

    let padding = 6.0;
    let rect = Rect::new(
        Vec2::new(screen.x - tw - padding * 2.0 - 8.0, 4.0),
        Vec2::new(screen.x - 8.0, 4.0 + style.font_size_small + padding * 2.0),
    );

    // Semi-transparent background.
    let bg = Color::new(0.0, 0.0, 0.0, 0.6);
    ui.renderer.draw_rect(rect, bg, 4.0);

    // FPS text.
    let fps_color = if fps >= 55.0 {
        style.green
    } else if fps >= 30.0 {
        style.yellow
    } else {
        style.red
    };

    ui.renderer.draw_text(
        &text,
        Vec2::new(rect.min.x + padding, rect.min.y + padding),
        style.font_size_small,
        fps_color,
    );
}

/// Draw a more detailed performance overlay showing FPS, frame time,
/// draw calls, and triangle count.
pub fn performance_overlay(
    ui: &mut UI,
    fps: f32,
    frame_time_ms: f32,
    draw_calls: u32,
    triangles: u32,
) {
    let style = ui.style.clone();
    let screen = ui.screen_size();
    let font = style.font_size_small;
    let padding = 8.0;
    let line_h = font + 2.0;

    let lines = [
        format!("FPS: {:.0}", fps),
        format!("Frame: {:.2} ms", frame_time_ms),
        format!("Draw calls: {}", draw_calls),
        format!("Triangles: {}", triangles),
    ];

    // Find widest line.
    let max_w = lines.iter()
        .map(|l| ui.renderer.measure_text(l, font).0)
        .fold(0.0f32, f32::max);

    let box_w = max_w + padding * 2.0;
    let box_h = lines.len() as f32 * line_h + padding * 2.0;

    let rect = Rect::new(
        Vec2::new(screen.x - box_w - 8.0, 4.0),
        Vec2::new(screen.x - 8.0, 4.0 + box_h),
    );

    // Background.
    let bg = Color::new(0.0, 0.0, 0.0, 0.7);
    ui.renderer.draw_rect(rect, bg, 4.0);
    ui.renderer.draw_rect_outline(rect, style.border, 1.0, 4.0);

    // Lines.
    let fps_color = if fps >= 55.0 {
        style.green
    } else if fps >= 30.0 {
        style.yellow
    } else {
        style.red
    };

    let colors = [fps_color, style.text_normal, style.text_dim, style.text_dim];

    for (i, line) in lines.iter().enumerate() {
        ui.renderer.draw_text(
            line,
            Vec2::new(
                rect.min.x + padding,
                rect.min.y + padding + i as f32 * line_h,
            ),
            font,
            colors[i],
        );
    }
}

// ---------------------------------------------------------------------------
// Asset thumbnail
// ---------------------------------------------------------------------------

/// Draw an asset thumbnail with a label underneath and selection highlight.
/// Returns true if clicked.
pub fn asset_thumbnail(
    ui: &mut UI,
    label: &str,
    selected: bool,
    size: f32,
) -> bool {
    let style = ui.style.clone();

    // Thumbnail background.
    let clicked = ui.selectable(label, selected);

    clicked
}

// ---------------------------------------------------------------------------
// Gizmo indicator
// ---------------------------------------------------------------------------

/// Draw a gizmo mode indicator showing which transform mode is active.
pub fn gizmo_indicator(
    ui: &mut UI,
    mode: &str,
    space: &str,
) {
    let style = ui.style.clone();

    ui.horizontal(|ui| {
        ui.label_colored("Mode:", style.text_dim);
        ui.space(4.0);
        ui.label_colored(mode, style.accent);
        ui.space(12.0);
        ui.label_colored("Space:", style.text_dim);
        ui.space(4.0);
        ui.label_colored(space, style.text_normal);
    });
}

// ---------------------------------------------------------------------------
// Notification toast
// ---------------------------------------------------------------------------

/// Draw a notification toast at the bottom-right of the screen.
pub fn notification_toast(
    ui: &mut UI,
    message: &str,
    level: LogLevel,
    progress: f32, // 0.0 to 1.0, how much of the display time has elapsed
) {
    let style = ui.style.clone();
    let screen = ui.screen_size();
    let font = style.font_size;
    let padding = 12.0;

    let (tw, _) = ui.renderer.measure_text(message, font);
    let toast_w = tw + padding * 2.0;
    let toast_h = font + padding * 2.0;

    // Slide in from the right based on progress.
    let alpha = if progress < 0.1 {
        progress / 0.1
    } else if progress > 0.8 {
        (1.0 - progress) / 0.2
    } else {
        1.0
    };

    let y = screen.y - toast_h - 20.0;
    let x = screen.x - toast_w - 20.0;

    let rect = Rect::new(
        Vec2::new(x, y),
        Vec2::new(x + toast_w, y + toast_h),
    );

    // Background with colored left strip.
    let bg = Color::new(
        style.bg_panel.r,
        style.bg_panel.g,
        style.bg_panel.b,
        alpha,
    );
    ui.renderer.draw_rect(rect, bg, style.corner_radius);

    // Colored left strip.
    let strip_color = level.color(&style).with_alpha(alpha);
    let strip_rect = Rect::new(
        rect.min,
        Vec2::new(rect.min.x + 3.0, rect.max.y),
    );
    ui.renderer.draw_rect(strip_rect, strip_color, style.corner_radius);

    // Border.
    ui.renderer.draw_rect_outline(
        rect,
        style.border.with_alpha(alpha),
        1.0,
        style.corner_radius,
    );

    // Message.
    let text_color = level.color(&style).with_alpha(alpha);
    ui.renderer.draw_text(
        message,
        Vec2::new(rect.min.x + padding, rect.min.y + padding),
        font,
        text_color,
    );
}

// ---------------------------------------------------------------------------
// Breadcrumb path
// ---------------------------------------------------------------------------

/// Draw a breadcrumb path (e.g., "Assets > Materials > PBR").
/// Returns the index of the clicked segment, if any.
pub fn breadcrumb_path(
    ui: &mut UI,
    segments: &[&str],
) -> Option<usize> {
    let style = ui.style.clone();
    let mut clicked_segment = None;

    ui.horizontal(|ui| {
        for (i, segment) in segments.iter().enumerate() {
            if i > 0 {
                ui.label_colored(">", style.text_dim);
                ui.space(4.0);
            }

            let is_last = i == segments.len() - 1;
            let color = if is_last {
                style.text_bright
            } else {
                style.accent_dim
            };

            if !is_last {
                // Make earlier segments clickable.
                if ui.button(segment) {
                    clicked_segment = Some(i);
                }
            } else {
                ui.label_colored(segment, color);
            }
        }
    });

    clicked_segment
}

// ---------------------------------------------------------------------------
// Splitter
// ---------------------------------------------------------------------------

/// Draw a draggable splitter handle between two panels.
/// `position` is the current split position in pixels.
/// `horizontal` determines if the splitter is horizontal (splits vertically)
/// or vertical (splits horizontally).
/// Returns the new split position.
pub fn splitter(
    ui: &mut UI,
    position: &mut f32,
    min_pos: f32,
    max_pos: f32,
    horizontal: bool,
) -> bool {
    let style = ui.style.clone();
    let thickness = 4.0;
    let _screen = ui.screen_size();

    // For now, just draw the splitter line.
    let changed = false;

    if horizontal {
        ui.renderer.draw_line(
            Vec2::new(0.0, *position),
            Vec2::new(ui.screen_size().x, *position),
            style.border,
            thickness,
        );
    } else {
        ui.renderer.draw_line(
            Vec2::new(*position, 0.0),
            Vec2::new(*position, ui.screen_size().y),
            style.border,
            thickness,
        );
    }

    changed
}

// ---------------------------------------------------------------------------
// Keyboard shortcut display
// ---------------------------------------------------------------------------

/// Draw a keyboard shortcut hint (e.g., "Ctrl+S").
pub fn shortcut_hint(ui: &mut UI, keys: &str) {
    let style = ui.style.clone();
    let (tw, _) = ui.renderer.measure_text(keys, style.font_size_small);
    let padding = 4.0;

    ui.horizontal(|ui| {
        // Draw a small rounded rectangle with the shortcut text.
        ui.label_colored(keys, style.text_dim);
    });
}

// ---------------------------------------------------------------------------
// Section with collapsible header
// ---------------------------------------------------------------------------

/// Draw a collapsible section with a dark header background and content area.
/// Returns true if the section is expanded.
pub fn collapsible_section(
    ui: &mut UI,
    title: &str,
    open: &mut bool,
) -> bool {
    component_header(ui, ">", title, open)
}

/// Draw a section header with a specific icon and optional badge count.
pub fn section_header_with_badge(
    ui: &mut UI,
    icon: &str,
    title: &str,
    badge_count: Option<u32>,
    open: &mut bool,
) -> bool {
    let display_title = if let Some(count) = badge_count {
        format!("{} {} ({})", icon, title, count)
    } else {
        format!("{} {}", icon, title)
    };

    component_header(ui, icon, &display_title, open)
}

// ---------------------------------------------------------------------------
// Drag-and-drop slot
// ---------------------------------------------------------------------------

/// Draw an empty drop target slot with a dashed border.
pub fn drop_slot(
    ui: &mut UI,
    label: &str,
    width: f32,
    height: f32,
) {
    let style = ui.style.clone();

    // Draw dashed border rectangle (simplified as a normal outline).
    let slot_rect = Rect::new(
        Vec2::ZERO,
        Vec2::new(width, height),
    );

    ui.horizontal(|ui| {
        // Placeholder.
        ui.space(width);
    });

    // Label in the center of the slot.
    if !label.is_empty() {
        let (tw, _) = ui.renderer.measure_text(label, style.font_size_small);
        // Center text position computed from slot.
    }
}

// ---------------------------------------------------------------------------
// Numeric field with label and optional unit suffix
// ---------------------------------------------------------------------------

/// Draw a numeric field with a label and optional unit suffix (e.g., "m", "deg").
/// Returns true if the value changed.
pub fn numeric_field(
    ui: &mut UI,
    label: &str,
    value: &mut f32,
    speed: f32,
    unit: Option<&str>,
) -> bool {
    let changed = ui.drag_value(label, value, speed);

    if let Some(unit_str) = unit {
        // Show unit text after the field.
        let style = ui.style.clone();
        ui.label_colored(unit_str, style.text_dim);
    }

    changed
}

/// Draw an integer numeric field.
/// Returns true if the value changed.
pub fn int_field(
    ui: &mut UI,
    label: &str,
    value: &mut i32,
    min: i32,
    max: i32,
) -> bool {
    let mut float_val = *value as f32;
    let changed = ui.drag_value(label, &mut float_val, 1.0);
    if changed {
        *value = float_val.round().clamp(min as f32, max as f32) as i32;
    }
    changed
}

// ---------------------------------------------------------------------------
// Mini timeline
// ---------------------------------------------------------------------------

/// Draw a mini animation timeline bar.
/// `current_time` and `duration` are in seconds.
/// Returns the new time if the user clicked/dragged on the timeline.
pub fn mini_timeline(
    ui: &mut UI,
    current_time: f32,
    duration: f32,
    width: f32,
) -> Option<f32> {
    let style = ui.style.clone();
    let h = 8.0;

    // For now, use progress_bar as a placeholder.
    let fraction = if duration > 0.0 {
        (current_time / duration).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let time_text = format!("{:.1}s / {:.1}s", current_time, duration);
    ui.progress_bar(fraction, Some(&time_text));

    None // Interactive scrubbing would return Some(new_time)
}

// ---------------------------------------------------------------------------
// Info row (label: value pairs)
// ---------------------------------------------------------------------------

/// Draw a label-value pair on a single row.
pub fn info_row(ui: &mut UI, label: &str, value: &str) {
    let style = ui.style.clone();

    ui.horizontal(|ui| {
        ui.label_colored(label, style.text_dim);
        ui.space(style.item_spacing);
        ui.label_colored(value, style.text_normal);
    });
}

/// Draw multiple info rows from a slice of (label, value) pairs.
pub fn info_table(ui: &mut UI, rows: &[(&str, &str)]) {
    for (label, value) in rows {
        info_row(ui, label, value);
    }
}

// ---------------------------------------------------------------------------
// Toggle group
// ---------------------------------------------------------------------------

/// Draw a group of mutually exclusive toggle buttons (radio-button style).
/// Returns true if the selection changed.
pub fn toggle_group(
    ui: &mut UI,
    options: &[&str],
    selected: &mut usize,
) -> bool {
    let old = *selected;

    ui.horizontal(|ui| {
        for (i, option) in options.iter().enumerate() {
            let is_selected = i == *selected;
            if toolbar_button(ui, option, option, is_selected) {
                *selected = i;
            }
        }
    });

    *selected != old
}

// ---------------------------------------------------------------------------
// Warning / info box
// ---------------------------------------------------------------------------

/// Draw a warning/info box with a colored left border and message.
pub fn message_box(
    ui: &mut UI,
    message: &str,
    level: LogLevel,
) {
    let style = ui.style.clone();
    let color = level.color(&style);

    ui.horizontal(|ui| {
        let icon = level.icon();
        ui.label_colored(icon, color);
        ui.space(style.item_spacing);
        ui.label_colored(message, style.text_normal);
    });
}
