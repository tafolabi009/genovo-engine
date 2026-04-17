//! Slate Accessibility Support
//!
//! Provides comprehensive accessibility features for the Genovo UI framework,
//! enabling screen reader support, keyboard navigation, high contrast mode,
//! and reduced motion preferences.
//!
//! # Architecture
//!
//! ```text
//!  AccessibleWidget ──> AccessibilityManager ──> Screen Reader Bridge
//!       │                       │                       │
//!  AccessibleRole         FocusTracking           Announcements
//!  AccessibleState        KeyboardNavigation      LiveRegions
//!       │                       │
//!  HighContrastMode      ReducedMotion
//! ```
//!
//! # Platform Integration
//!
//! The accessibility system produces a tree of `AccessibleNode` structs that
//! can be bridged to platform-specific accessibility APIs (e.g., UI Automation
//! on Windows, NSAccessibility on macOS, AT-SPI on Linux).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use glam::Vec2;

use crate::core::UIId;
use crate::render_commands::Color;

// ---------------------------------------------------------------------------
// AccessibleRole
// ---------------------------------------------------------------------------

/// The semantic role of an accessible widget, corresponding to ARIA roles.
///
/// Screen readers use this to determine how to present the widget to the user
/// and which keyboard interactions to support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessibleRole {
    /// A clickable button.
    Button,
    /// A checkbox (checked/unchecked/indeterminate).
    CheckBox,
    /// A combo box (text input + dropdown list).
    ComboBox,
    /// A modal or non-modal dialog.
    Dialog,
    /// A data grid with rows and columns.
    Grid,
    /// A grid cell within a grid row.
    GridCell,
    /// An image or icon.
    Image,
    /// A hyperlink.
    Link,
    /// An unordered or ordered list.
    List,
    /// An item within a list.
    ListItem,
    /// A menu (context menu, dropdown menu).
    Menu,
    /// An item within a menu.
    MenuItem,
    /// A menu item with a checkbox.
    MenuItemCheckBox,
    /// A menu item with a radio button.
    MenuItemRadio,
    /// A progress bar (determinate or indeterminate).
    ProgressBar,
    /// A radio button (one of a group).
    RadioButton,
    /// A radio button group.
    RadioGroup,
    /// A scrollbar.
    ScrollBar,
    /// A search input field.
    Search,
    /// A separator / divider.
    Separator,
    /// A slider (range input).
    Slider,
    /// A spin button (numeric input with up/down).
    SpinButton,
    /// A status message area.
    Status,
    /// A tab control.
    Tab,
    /// A tab list container.
    TabList,
    /// A tab panel (content area for a tab).
    TabPanel,
    /// Static text.
    Text,
    /// A text input field.
    TextInput,
    /// A toolbar.
    Toolbar,
    /// A tooltip.
    Tooltip,
    /// A tree view.
    Tree,
    /// An item within a tree view.
    TreeItem,
    /// A top-level window.
    Window,
    /// A panel or group of related widgets.
    Group,
    /// A navigation landmark.
    Navigation,
    /// A content region.
    Region,
    /// An alert message.
    Alert,
    /// A heading.
    Heading,
    /// No specific role (generic container).
    None,
}

impl fmt::Display for AccessibleRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Button => "Button",
            Self::CheckBox => "CheckBox",
            Self::ComboBox => "ComboBox",
            Self::Dialog => "Dialog",
            Self::Grid => "Grid",
            Self::GridCell => "GridCell",
            Self::Image => "Image",
            Self::Link => "Link",
            Self::List => "List",
            Self::ListItem => "ListItem",
            Self::Menu => "Menu",
            Self::MenuItem => "MenuItem",
            Self::MenuItemCheckBox => "MenuItemCheckBox",
            Self::MenuItemRadio => "MenuItemRadio",
            Self::ProgressBar => "ProgressBar",
            Self::RadioButton => "RadioButton",
            Self::RadioGroup => "RadioGroup",
            Self::ScrollBar => "ScrollBar",
            Self::Search => "Search",
            Self::Separator => "Separator",
            Self::Slider => "Slider",
            Self::SpinButton => "SpinButton",
            Self::Status => "Status",
            Self::Tab => "Tab",
            Self::TabList => "TabList",
            Self::TabPanel => "TabPanel",
            Self::Text => "Text",
            Self::TextInput => "TextInput",
            Self::Toolbar => "Toolbar",
            Self::Tooltip => "Tooltip",
            Self::Tree => "Tree",
            Self::TreeItem => "TreeItem",
            Self::Window => "Window",
            Self::Group => "Group",
            Self::Navigation => "Navigation",
            Self::Region => "Region",
            Self::Alert => "Alert",
            Self::Heading => "Heading",
            Self::None => "None",
        };
        write!(f, "{}", name)
    }
}

impl AccessibleRole {
    /// Returns true if this role is typically focusable.
    pub fn is_focusable(&self) -> bool {
        matches!(
            self,
            Self::Button
                | Self::CheckBox
                | Self::ComboBox
                | Self::Link
                | Self::MenuItem
                | Self::MenuItemCheckBox
                | Self::MenuItemRadio
                | Self::RadioButton
                | Self::Slider
                | Self::SpinButton
                | Self::Tab
                | Self::TextInput
                | Self::TreeItem
                | Self::GridCell
        )
    }

    /// Returns true if this role supports a value.
    pub fn has_value(&self) -> bool {
        matches!(
            self,
            Self::Slider
                | Self::SpinButton
                | Self::ProgressBar
                | Self::ScrollBar
                | Self::TextInput
                | Self::ComboBox
        )
    }

    /// Returns true if this role supports expandable state.
    pub fn is_expandable(&self) -> bool {
        matches!(
            self,
            Self::TreeItem | Self::MenuItem | Self::ComboBox | Self::Group
        )
    }

    /// Returns true if this role supports checked state.
    pub fn is_checkable(&self) -> bool {
        matches!(
            self,
            Self::CheckBox
                | Self::MenuItemCheckBox
                | Self::MenuItemRadio
                | Self::RadioButton
        )
    }

    /// Returns the ARIA level for heading-like roles.
    pub fn default_heading_level(&self) -> Option<u32> {
        match self {
            Self::Heading => Some(2),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// AccessibleState
// ---------------------------------------------------------------------------

/// The current state of an accessible widget.
///
/// These flags correspond to ARIA states and properties that screen readers
/// use to convey the widget's current condition to the user.
#[derive(Debug, Clone, PartialEq)]
pub struct AccessibleState {
    /// Whether the widget is enabled (interactive).
    pub enabled: bool,
    /// Whether the widget currently has keyboard focus.
    pub focused: bool,
    /// Whether the widget is selected (in a list, grid, etc.).
    pub selected: bool,
    /// Whether the widget is expanded (tree item, combo box).
    pub expanded: Option<bool>,
    /// Whether the widget is checked (checkbox, radio button).
    pub checked: Option<CheckedState>,
    /// Whether the widget is required (form field).
    pub required: bool,
    /// Whether the widget is read-only.
    pub read_only: bool,
    /// Whether the widget is hidden from accessibility.
    pub hidden: bool,
    /// Whether the widget is currently being pressed.
    pub pressed: bool,
    /// Whether the widget is in an invalid state.
    pub invalid: bool,
    /// Whether the widget is busy (loading).
    pub busy: bool,
    /// The widget's position within a set (1-based).
    pub position_in_set: Option<u32>,
    /// The total size of the set this widget belongs to.
    pub set_size: Option<u32>,
    /// Heading level (1-6).
    pub level: Option<u32>,
    /// Numeric value (for sliders, progress bars).
    pub value_now: Option<f64>,
    /// Minimum value.
    pub value_min: Option<f64>,
    /// Maximum value.
    pub value_max: Option<f64>,
    /// Text representation of the value.
    pub value_text: Option<String>,
    /// Orientation: horizontal or vertical.
    pub orientation: Option<Orientation>,
    /// Whether this is a live region (announces changes).
    pub live: Option<LiveRegionMode>,
    /// Whether the widget has a popup.
    pub has_popup: bool,
    /// Whether the widget is modal.
    pub modal: bool,
    /// Whether the widget is multiselectable.
    pub multi_selectable: bool,
    /// Sort direction for columns.
    pub sort_direction: Option<SortDirection>,
}

impl Default for AccessibleState {
    fn default() -> Self {
        Self {
            enabled: true,
            focused: false,
            selected: false,
            expanded: None,
            checked: None,
            required: false,
            read_only: false,
            hidden: false,
            pressed: false,
            invalid: false,
            busy: false,
            position_in_set: None,
            set_size: None,
            level: None,
            value_now: None,
            value_min: None,
            value_max: None,
            value_text: None,
            orientation: None,
            live: None,
            has_popup: false,
            modal: false,
            multi_selectable: false,
            sort_direction: None,
        }
    }
}

impl AccessibleState {
    /// Creates a new default state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a state for a disabled widget.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Returns true if any "active" state is set (focused, selected, pressed, etc.).
    pub fn is_active(&self) -> bool {
        self.focused || self.selected || self.pressed
    }

    /// Returns a human-readable summary of the state.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.enabled {
            parts.push("disabled");
        }
        if self.focused {
            parts.push("focused");
        }
        if self.selected {
            parts.push("selected");
        }
        if let Some(expanded) = self.expanded {
            parts.push(if expanded { "expanded" } else { "collapsed" });
        }
        if let Some(checked) = &self.checked {
            parts.push(match checked {
                CheckedState::Checked => "checked",
                CheckedState::Unchecked => "unchecked",
                CheckedState::Indeterminate => "mixed",
            });
        }
        if self.pressed {
            parts.push("pressed");
        }
        if self.invalid {
            parts.push("invalid");
        }
        if self.busy {
            parts.push("busy");
        }
        if self.hidden {
            parts.push("hidden");
        }
        if parts.is_empty() {
            "normal".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Checked state for checkable widgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedState {
    /// Not checked.
    Unchecked,
    /// Checked.
    Checked,
    /// Indeterminate / mixed.
    Indeterminate,
}

/// Orientation for widgets like sliders and scrollbars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

/// Live region mode for announcing changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveRegionMode {
    /// Announce only when the region is focused.
    Off,
    /// Announce changes politely (after current speech).
    Polite,
    /// Announce changes immediately (interrupt current speech).
    Assertive,
}

/// Sort direction for grid columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
    Other,
    None,
}

// ---------------------------------------------------------------------------
// AccessibleWidget trait
// ---------------------------------------------------------------------------

/// Trait that widgets implement to provide accessibility information.
///
/// This is the primary interface between the widget system and the accessibility
/// layer. Each widget that should be exposed to assistive technologies implements
/// this trait to describe its role, name, value, and state.
pub trait AccessibleWidget {
    /// Returns the semantic role of this widget.
    fn accessible_role(&self) -> AccessibleRole;

    /// Returns the accessible name (label) for this widget.
    ///
    /// This is what screen readers announce as the widget's name. It should be
    /// a concise, human-readable label.
    fn accessible_name(&self) -> String;

    /// Returns an extended description of the widget.
    ///
    /// This provides additional context beyond the name, such as instructions
    /// for how to interact with the widget.
    fn accessible_description(&self) -> String {
        String::new()
    }

    /// Returns the current value of the widget, if applicable.
    ///
    /// For sliders, this might be "50%". For text inputs, the current text.
    /// For checkboxes, "checked" or "unchecked".
    fn accessible_value(&self) -> Option<String> {
        None
    }

    /// Returns the current state of the widget.
    fn accessible_state(&self) -> AccessibleState {
        AccessibleState::default()
    }

    /// Returns the widget's keyboard shortcut, if any.
    fn accessible_shortcut(&self) -> Option<String> {
        None
    }

    /// Returns the IDs of this widget's accessible children.
    fn accessible_children(&self) -> Vec<UIId> {
        Vec::new()
    }

    /// Returns the ID of the widget that labels this widget.
    fn accessible_labelled_by(&self) -> Option<UIId> {
        None
    }

    /// Returns the ID of the widget that describes this widget.
    fn accessible_described_by(&self) -> Option<UIId> {
        None
    }

    /// Returns the ID of the widget that this widget controls.
    fn accessible_controls(&self) -> Option<UIId> {
        None
    }

    /// Returns true if this widget should participate in tab navigation.
    fn is_tab_stop(&self) -> bool {
        self.accessible_role().is_focusable()
    }

    /// Returns the tab index override (negative = skip, 0 = natural order, positive = priority).
    fn tab_index(&self) -> i32 {
        0
    }
}

// ---------------------------------------------------------------------------
// AccessibleNode
// ---------------------------------------------------------------------------

/// A node in the accessibility tree, representing one accessible widget.
///
/// This is the data structure that gets passed to the platform accessibility
/// API bridge. It contains all information needed to present the widget to
/// assistive technologies.
#[derive(Debug, Clone)]
pub struct AccessibleNode {
    /// The widget ID this node represents.
    pub widget_id: UIId,
    /// The semantic role.
    pub role: AccessibleRole,
    /// The accessible name (label).
    pub name: String,
    /// The accessible description.
    pub description: String,
    /// The current value.
    pub value: Option<String>,
    /// The current state.
    pub state: AccessibleState,
    /// Keyboard shortcut.
    pub shortcut: Option<String>,
    /// Position on screen in logical pixels.
    pub bounds: [f32; 4],
    /// IDs of child accessible nodes.
    pub children: Vec<UIId>,
    /// ID of the parent node.
    pub parent: Option<UIId>,
    /// Tab index for keyboard navigation.
    pub tab_index: i32,
    /// Whether this node is a tab stop.
    pub is_tab_stop: bool,
    /// ID of the labelling widget.
    pub labelled_by: Option<UIId>,
    /// ID of the describing widget.
    pub described_by: Option<UIId>,
    /// ID of the controlled widget.
    pub controls: Option<UIId>,
    /// Index in the accessible tree (for stable ordering).
    pub tree_index: u32,
}

impl AccessibleNode {
    /// Creates a new accessible node with minimal required fields.
    pub fn new(widget_id: UIId, role: AccessibleRole, name: &str) -> Self {
        Self {
            widget_id,
            role,
            name: name.to_string(),
            description: String::new(),
            value: None,
            state: AccessibleState::default(),
            shortcut: None,
            bounds: [0.0; 4],
            children: Vec::new(),
            parent: None,
            tab_index: 0,
            is_tab_stop: role.is_focusable(),
            labelled_by: None,
            described_by: None,
            controls: None,
            tree_index: 0,
        }
    }

    /// Returns a full announcement string for screen readers.
    pub fn announce(&self) -> String {
        let mut parts = vec![self.name.clone()];
        if self.role != AccessibleRole::None {
            parts.push(self.role.to_string());
        }
        if let Some(value) = &self.value {
            parts.push(value.clone());
        }
        let state_summary = self.state.summary();
        if state_summary != "normal" {
            parts.push(state_summary);
        }
        parts.join(", ")
    }

    /// Returns true if this node should be visible to assistive technologies.
    pub fn is_accessible(&self) -> bool {
        !self.state.hidden && self.role != AccessibleRole::None
    }
}

// ---------------------------------------------------------------------------
// Announcement
// ---------------------------------------------------------------------------

/// A queued announcement for screen readers.
#[derive(Debug, Clone)]
pub struct Announcement {
    /// The text to announce.
    pub text: String,
    /// The priority level.
    pub priority: LiveRegionMode,
    /// When this announcement was created.
    pub timestamp: u64,
    /// Whether this announcement has been consumed.
    pub consumed: bool,
}

impl Announcement {
    /// Creates a new polite announcement.
    pub fn polite(text: &str, timestamp: u64) -> Self {
        Self {
            text: text.to_string(),
            priority: LiveRegionMode::Polite,
            timestamp,
            consumed: false,
        }
    }

    /// Creates a new assertive announcement.
    pub fn assertive(text: &str, timestamp: u64) -> Self {
        Self {
            text: text.to_string(),
            priority: LiveRegionMode::Assertive,
            timestamp,
            consumed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// AccessibilityManager
// ---------------------------------------------------------------------------

/// Manages the accessibility tree and coordinates with assistive technologies.
///
/// The manager walks the widget tree each frame, collects accessible data from
/// widgets that implement `AccessibleWidget`, builds an accessibility tree, and
/// provides focus tracking and change announcements.
///
/// # Focus tracking
///
/// The manager maintains which widget currently has accessibility focus (which
/// may differ from keyboard focus). When focus changes, it generates an
/// announcement for screen readers.
///
/// # Live regions
///
/// Widgets can register as live regions, causing their content changes to be
/// automatically announced to screen readers.
#[derive(Debug, Clone)]
pub struct AccessibilityManager {
    /// The accessibility tree, keyed by widget ID index.
    pub tree: HashMap<u32, AccessibleNode>,
    /// The ID of the currently focused accessible widget.
    pub focused_id: Option<UIId>,
    /// The ID of the previously focused widget (for tracking changes).
    pub previous_focused_id: Option<UIId>,
    /// Queue of pending announcements.
    pub announcements: VecDeque<Announcement>,
    /// Maximum announcements to queue.
    pub max_announcements: usize,
    /// Set of widget IDs registered as live regions.
    pub live_regions: HashSet<u32>,
    /// Previous values of live region widgets (for change detection).
    pub live_region_values: HashMap<u32, String>,
    /// Whether accessibility is enabled.
    pub enabled: bool,
    /// Current frame/timestamp counter.
    pub timestamp: u64,
    /// Ordered list of tab stops for keyboard navigation.
    pub tab_order: Vec<UIId>,
    /// Whether the tab order needs to be rebuilt.
    pub tab_order_dirty: bool,
    /// Screen reader connected flag.
    pub screen_reader_active: bool,
    /// Root node ID for the tree.
    pub root_id: Option<UIId>,
    /// Total nodes in the tree.
    pub node_count: u32,
}

impl AccessibilityManager {
    /// Creates a new accessibility manager.
    pub fn new() -> Self {
        Self {
            tree: HashMap::with_capacity(256),
            focused_id: None,
            previous_focused_id: None,
            announcements: VecDeque::with_capacity(16),
            max_announcements: 32,
            live_regions: HashSet::new(),
            live_region_values: HashMap::new(),
            enabled: true,
            timestamp: 0,
            tab_order: Vec::new(),
            tab_order_dirty: true,
            screen_reader_active: false,
            root_id: None,
            node_count: 0,
        }
    }

    /// Registers a widget in the accessibility tree.
    pub fn register_widget(&mut self, node: AccessibleNode) {
        let index = node.widget_id.index;
        if node.is_tab_stop {
            self.tab_order_dirty = true;
        }
        if let Some(live) = &node.state.live {
            if *live != LiveRegionMode::Off {
                self.live_regions.insert(index);
            }
        }
        self.tree.insert(index, node);
        self.node_count = self.tree.len() as u32;
    }

    /// Removes a widget from the accessibility tree.
    pub fn unregister_widget(&mut self, widget_id: UIId) {
        self.tree.remove(&widget_id.index);
        self.live_regions.remove(&widget_id.index);
        self.live_region_values.remove(&widget_id.index);
        self.tab_order_dirty = true;
        self.node_count = self.tree.len() as u32;

        if self.focused_id == Some(widget_id) {
            self.focused_id = None;
        }
    }

    /// Updates the accessibility tree for a new frame.
    pub fn update(&mut self) {
        self.timestamp += 1;
        self.check_live_regions();

        // Track focus changes.
        if self.focused_id != self.previous_focused_id {
            if let Some(id) = self.focused_id {
                if let Some(node) = self.tree.get(&id.index) {
                    let text = node.announce();
                    self.announce_assertive(&text);
                }
            }
            self.previous_focused_id = self.focused_id;
        }
    }

    /// Checks live regions for changes and generates announcements.
    fn check_live_regions(&mut self) {
        let regions: Vec<u32> = self.live_regions.iter().copied().collect();
        for index in regions {
            if let Some(node) = self.tree.get(&index) {
                let current_value = node.value.clone().unwrap_or_default();
                let prev_value = self
                    .live_region_values
                    .get(&index)
                    .cloned()
                    .unwrap_or_default();

                if current_value != prev_value {
                    let priority = node
                        .state
                        .live
                        .unwrap_or(LiveRegionMode::Polite);
                    let announcement = Announcement {
                        text: current_value.clone(),
                        priority,
                        timestamp: self.timestamp,
                        consumed: false,
                    };
                    self.announcements.push_back(announcement);
                    self.live_region_values
                        .insert(index, current_value);
                }
            }
        }

        // Trim old announcements.
        while self.announcements.len() > self.max_announcements {
            self.announcements.pop_front();
        }
    }

    /// Sets the focused widget.
    pub fn set_focus(&mut self, widget_id: UIId) {
        // Remove focus from previous widget.
        if let Some(prev) = self.focused_id {
            if let Some(node) = self.tree.get_mut(&prev.index) {
                node.state.focused = false;
            }
        }
        // Set focus on new widget.
        if let Some(node) = self.tree.get_mut(&widget_id.index) {
            node.state.focused = true;
        }
        self.focused_id = Some(widget_id);
    }

    /// Clears the current focus.
    pub fn clear_focus(&mut self) {
        if let Some(prev) = self.focused_id {
            if let Some(node) = self.tree.get_mut(&prev.index) {
                node.state.focused = false;
            }
        }
        self.focused_id = None;
    }

    /// Announces a message politely (after current speech finishes).
    pub fn announce_polite(&mut self, text: &str) {
        self.announcements
            .push_back(Announcement::polite(text, self.timestamp));
    }

    /// Announces a message assertively (interrupts current speech).
    pub fn announce_assertive(&mut self, text: &str) {
        self.announcements
            .push_back(Announcement::assertive(text, self.timestamp));
    }

    /// Consumes and returns the next pending announcement.
    pub fn next_announcement(&mut self) -> Option<Announcement> {
        while let Some(front) = self.announcements.front() {
            if front.consumed {
                self.announcements.pop_front();
            } else {
                break;
            }
        }
        if let Some(front) = self.announcements.front_mut() {
            front.consumed = true;
            Some(front.clone())
        } else {
            None
        }
    }

    /// Returns the accessible node for a widget, if it exists.
    pub fn get_node(&self, widget_id: UIId) -> Option<&AccessibleNode> {
        self.tree.get(&widget_id.index)
    }

    /// Returns the focused node, if any.
    pub fn get_focused_node(&self) -> Option<&AccessibleNode> {
        self.focused_id
            .and_then(|id| self.tree.get(&id.index))
    }

    /// Returns all accessible nodes in tree order.
    pub fn all_nodes(&self) -> Vec<&AccessibleNode> {
        let mut nodes: Vec<&AccessibleNode> = self.tree.values().collect();
        nodes.sort_by_key(|n| n.tree_index);
        nodes
    }

    /// Builds the tab order from the current tree.
    pub fn rebuild_tab_order(&mut self) {
        self.tab_order.clear();
        let mut tab_stops: Vec<(UIId, i32, u32)> = self
            .tree
            .values()
            .filter(|n| n.is_tab_stop && !n.state.hidden && n.state.enabled)
            .map(|n| (n.widget_id, n.tab_index, n.tree_index))
            .collect();

        // Sort: positive tab_index first (in order), then tab_index=0 in tree order.
        tab_stops.sort_by(|a, b| {
            match (a.1, b.1) {
                (0, 0) => a.2.cmp(&b.2),
                (0, _) => std::cmp::Ordering::Greater,
                (_, 0) => std::cmp::Ordering::Less,
                _ => a.1.cmp(&b.1).then(a.2.cmp(&b.2)),
            }
        });

        self.tab_order = tab_stops.into_iter().map(|(id, _, _)| id).collect();
        self.tab_order_dirty = false;
    }

    /// Returns the number of tab stops.
    pub fn tab_stop_count(&self) -> usize {
        if self.tab_order_dirty {
            // Can't mutate here, return estimate.
            self.tree.values().filter(|n| n.is_tab_stop).count()
        } else {
            self.tab_order.len()
        }
    }
}

impl Default for AccessibilityManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FocusScope
// ---------------------------------------------------------------------------

/// A group of widgets with internal arrow-key navigation.
///
/// Focus scopes define navigation boundaries. Within a scope, arrow keys move
/// focus between items. Tab moves between scopes. This is used for toolbars,
/// menus, tab lists, tree views, and grids.
#[derive(Debug, Clone)]
pub struct FocusScope {
    /// Unique identifier for this scope.
    pub id: u32,
    /// Name for debugging.
    pub name: String,
    /// Widget IDs in this scope, in navigation order.
    pub items: Vec<UIId>,
    /// Currently focused index within the scope.
    pub focused_index: Option<usize>,
    /// Navigation direction (horizontal for toolbars, vertical for lists).
    pub direction: FocusDirection,
    /// Whether navigation wraps around at the ends.
    pub wrap: bool,
    /// Whether this scope traps focus (used for dialogs).
    pub trap_focus: bool,
    /// Whether this scope is active (receives navigation events).
    pub active: bool,
    /// The parent scope ID, if any.
    pub parent_scope: Option<u32>,
    /// Child scope IDs.
    pub child_scopes: Vec<u32>,
}

/// Navigation direction within a focus scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusDirection {
    /// Left/right arrow keys navigate.
    Horizontal,
    /// Up/down arrow keys navigate.
    Vertical,
    /// Both directions navigate (grid).
    Both,
}

impl FocusScope {
    /// Creates a new focus scope.
    pub fn new(id: u32, name: &str, direction: FocusDirection) -> Self {
        Self {
            id,
            name: name.to_string(),
            items: Vec::new(),
            focused_index: None,
            direction,
            wrap: true,
            trap_focus: false,
            active: true,
            parent_scope: None,
            child_scopes: Vec::new(),
        }
    }

    /// Adds a widget to this scope.
    pub fn add_item(&mut self, widget_id: UIId) {
        if !self.items.contains(&widget_id) {
            self.items.push(widget_id);
        }
    }

    /// Removes a widget from this scope.
    pub fn remove_item(&mut self, widget_id: UIId) {
        self.items.retain(|id| *id != widget_id);
        if let Some(index) = self.focused_index {
            if index >= self.items.len() {
                self.focused_index = if self.items.is_empty() {
                    None
                } else {
                    Some(self.items.len() - 1)
                };
            }
        }
    }

    /// Moves focus to the next item.
    pub fn focus_next(&mut self) -> Option<UIId> {
        if self.items.is_empty() {
            return None;
        }
        let current = self.focused_index.unwrap_or(0);
        let next = if current + 1 >= self.items.len() {
            if self.wrap {
                0
            } else {
                return None;
            }
        } else {
            current + 1
        };
        self.focused_index = Some(next);
        Some(self.items[next])
    }

    /// Moves focus to the previous item.
    pub fn focus_previous(&mut self) -> Option<UIId> {
        if self.items.is_empty() {
            return None;
        }
        let current = self.focused_index.unwrap_or(0);
        let prev = if current == 0 {
            if self.wrap {
                self.items.len() - 1
            } else {
                return None;
            }
        } else {
            current - 1
        };
        self.focused_index = Some(prev);
        Some(self.items[prev])
    }

    /// Sets focus to a specific item by widget ID.
    pub fn focus_item(&mut self, widget_id: UIId) -> bool {
        if let Some(index) = self.items.iter().position(|id| *id == widget_id) {
            self.focused_index = Some(index);
            true
        } else {
            false
        }
    }

    /// Sets focus to the first item.
    pub fn focus_first(&mut self) -> Option<UIId> {
        if self.items.is_empty() {
            None
        } else {
            self.focused_index = Some(0);
            Some(self.items[0])
        }
    }

    /// Sets focus to the last item.
    pub fn focus_last(&mut self) -> Option<UIId> {
        if self.items.is_empty() {
            None
        } else {
            let last = self.items.len() - 1;
            self.focused_index = Some(last);
            Some(self.items[last])
        }
    }

    /// Returns the currently focused widget ID.
    pub fn current_focus(&self) -> Option<UIId> {
        self.focused_index.and_then(|i| self.items.get(i).copied())
    }

    /// Returns the number of items in this scope.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }
}

// ---------------------------------------------------------------------------
// KeyboardNavigation
// ---------------------------------------------------------------------------

/// Manages keyboard navigation across the entire UI.
///
/// Handles tab order (sequential focus navigation), arrow key navigation within
/// focus scopes, escape to close popups, enter to activate, and the visible
/// focus ring indicator.
#[derive(Debug, Clone)]
pub struct KeyboardNavigation {
    /// All focus scopes, keyed by scope ID.
    pub scopes: HashMap<u32, FocusScope>,
    /// The currently active scope ID.
    pub active_scope: Option<u32>,
    /// Global tab order of widget IDs.
    pub tab_order: Vec<UIId>,
    /// Current index in the global tab order.
    pub tab_index: Option<usize>,
    /// Whether the focus ring should be visible.
    pub show_focus_ring: bool,
    /// Whether keyboard navigation mode is active (vs mouse mode).
    pub keyboard_mode: bool,
    /// Focus ring color.
    pub focus_ring_color: Color,
    /// Focus ring thickness in pixels.
    pub focus_ring_thickness: f32,
    /// Focus ring corner radius.
    pub focus_ring_radius: f32,
    /// Focus ring padding (gap between widget and ring).
    pub focus_ring_padding: f32,
    /// Stack of popup scopes (for escape handling).
    pub popup_stack: Vec<u32>,
    /// ID counter for scope creation.
    next_scope_id: u32,
    /// The currently focused widget (across all scopes).
    pub focused_widget: Option<UIId>,
    /// Whether focus changed this frame.
    pub focus_changed: bool,
}

impl KeyboardNavigation {
    /// Creates a new keyboard navigation manager.
    pub fn new() -> Self {
        Self {
            scopes: HashMap::new(),
            active_scope: None,
            tab_order: Vec::new(),
            tab_index: None,
            show_focus_ring: false,
            keyboard_mode: false,
            focus_ring_color: Color::new(0.3, 0.5, 1.0, 0.8),
            focus_ring_thickness: 2.0,
            focus_ring_radius: 4.0,
            focus_ring_padding: 2.0,
            popup_stack: Vec::new(),
            next_scope_id: 1,
            focused_widget: None,
            focus_changed: false,
        }
    }

    /// Creates a new focus scope and returns its ID.
    pub fn create_scope(&mut self, name: &str, direction: FocusDirection) -> u32 {
        let id = self.next_scope_id;
        self.next_scope_id += 1;
        let scope = FocusScope::new(id, name, direction);
        self.scopes.insert(id, scope);
        id
    }

    /// Removes a focus scope.
    pub fn destroy_scope(&mut self, id: u32) {
        self.scopes.remove(&id);
        if self.active_scope == Some(id) {
            self.active_scope = None;
        }
        self.popup_stack.retain(|s| *s != id);
    }

    /// Adds a widget to a focus scope.
    pub fn add_to_scope(&mut self, scope_id: u32, widget_id: UIId) {
        if let Some(scope) = self.scopes.get_mut(&scope_id) {
            scope.add_item(widget_id);
        }
    }

    /// Pushes a popup scope onto the stack (for modal focus trapping).
    pub fn push_popup_scope(&mut self, scope_id: u32) {
        if let Some(scope) = self.scopes.get_mut(&scope_id) {
            scope.trap_focus = true;
        }
        self.popup_stack.push(scope_id);
        self.active_scope = Some(scope_id);
    }

    /// Pops the top popup scope from the stack.
    pub fn pop_popup_scope(&mut self) -> Option<u32> {
        let popped = self.popup_stack.pop();
        self.active_scope = self.popup_stack.last().copied();
        popped
    }

    /// Handles a Tab key press (forward navigation).
    pub fn handle_tab(&mut self, shift: bool) -> Option<UIId> {
        self.keyboard_mode = true;
        self.show_focus_ring = true;
        self.focus_changed = true;

        // If we're in a focus-trapping scope, navigate within it.
        if let Some(scope_id) = self.active_scope {
            if let Some(scope) = self.scopes.get(&scope_id) {
                if scope.trap_focus {
                    return self.navigate_within_scope(scope_id, shift);
                }
            }
        }

        // Global tab navigation.
        if self.tab_order.is_empty() {
            return None;
        }

        let current = self.tab_index.unwrap_or(if shift {
            self.tab_order.len()
        } else {
            0usize.wrapping_sub(1)
        });

        let next = if shift {
            if current == 0 {
                self.tab_order.len() - 1
            } else {
                current - 1
            }
        } else {
            (current + 1) % self.tab_order.len()
        };

        self.tab_index = Some(next);
        let widget_id = self.tab_order[next];
        self.focused_widget = Some(widget_id);
        Some(widget_id)
    }

    /// Handles arrow key navigation within the active scope.
    pub fn handle_arrow(&mut self, key: ArrowKey) -> Option<UIId> {
        self.keyboard_mode = true;
        self.show_focus_ring = true;
        self.focus_changed = true;

        let scope_id = self.active_scope?;
        let scope = self.scopes.get(&scope_id)?;

        let is_forward = match (scope.direction, key) {
            (FocusDirection::Horizontal, ArrowKey::Right) => true,
            (FocusDirection::Horizontal, ArrowKey::Left) => false,
            (FocusDirection::Vertical, ArrowKey::Down) => true,
            (FocusDirection::Vertical, ArrowKey::Up) => false,
            (FocusDirection::Both, ArrowKey::Right | ArrowKey::Down) => true,
            (FocusDirection::Both, ArrowKey::Left | ArrowKey::Up) => false,
            _ => return None,
        };

        self.navigate_within_scope(scope_id, !is_forward)
    }

    /// Navigates within a scope.
    fn navigate_within_scope(&mut self, scope_id: u32, reverse: bool) -> Option<UIId> {
        let scope = self.scopes.get_mut(&scope_id)?;
        let widget_id = if reverse {
            scope.focus_previous()
        } else {
            scope.focus_next()
        };
        if let Some(id) = widget_id {
            self.focused_widget = Some(id);
        }
        widget_id
    }

    /// Handles Escape key press (close popup / clear focus).
    pub fn handle_escape(&mut self) -> bool {
        self.focus_changed = true;
        if !self.popup_stack.is_empty() {
            self.pop_popup_scope();
            true
        } else {
            self.focused_widget = None;
            self.show_focus_ring = false;
            self.keyboard_mode = false;
            false
        }
    }

    /// Handles Enter key press (activate focused widget).
    pub fn handle_enter(&mut self) -> Option<UIId> {
        self.focused_widget
    }

    /// Called when the mouse is used, to hide the focus ring.
    pub fn on_mouse_input(&mut self) {
        self.keyboard_mode = false;
        self.show_focus_ring = false;
    }

    /// Sets focus to a specific widget.
    pub fn set_focus(&mut self, widget_id: UIId) {
        self.focused_widget = Some(widget_id);
        self.focus_changed = true;

        // Update tab index to match.
        if let Some(idx) = self.tab_order.iter().position(|id| *id == widget_id) {
            self.tab_index = Some(idx);
        }

        // Update scope focus.
        for scope in self.scopes.values_mut() {
            scope.focus_item(widget_id);
        }
    }

    /// Clears the current focus.
    pub fn clear_focus(&mut self) {
        self.focused_widget = None;
        self.tab_index = None;
        self.focus_changed = true;
    }

    /// Sets the global tab order.
    pub fn set_tab_order(&mut self, order: Vec<UIId>) {
        self.tab_order = order;
    }

    /// Returns the focus ring rectangle for the given widget bounds.
    pub fn focus_ring_rect(&self, widget_bounds: [f32; 4]) -> [f32; 4] {
        let pad = self.focus_ring_padding;
        [
            widget_bounds[0] - pad,
            widget_bounds[1] - pad,
            widget_bounds[2] + pad * 2.0,
            widget_bounds[3] + pad * 2.0,
        ]
    }

    /// Resets the focus_changed flag at end of frame.
    pub fn end_frame(&mut self) {
        self.focus_changed = false;
    }
}

impl Default for KeyboardNavigation {
    fn default() -> Self {
        Self::new()
    }
}

/// Arrow key direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowKey {
    Up,
    Down,
    Left,
    Right,
}

// ---------------------------------------------------------------------------
// HighContrastMode
// ---------------------------------------------------------------------------

/// High contrast mode support.
///
/// Detects the OS high contrast setting and provides override colors with
/// bright outlines, high contrast text, and no gradients. This makes the UI
/// usable for users with low vision.
#[derive(Debug, Clone)]
pub struct HighContrastMode {
    /// Whether high contrast mode is currently active.
    pub active: bool,
    /// Whether to auto-detect from OS settings.
    pub auto_detect: bool,
    /// Forced on/off (overrides auto-detect).
    pub forced: Option<bool>,
    /// Background color in high contrast mode.
    pub background: Color,
    /// Foreground (text) color.
    pub foreground: Color,
    /// Highlight color (focused/selected items).
    pub highlight: Color,
    /// Highlight text color.
    pub highlight_text: Color,
    /// Disabled text color.
    pub disabled_text: Color,
    /// Border/outline color.
    pub border: Color,
    /// Link/hyperlink color.
    pub link: Color,
    /// Button face color.
    pub button_face: Color,
    /// Button text color.
    pub button_text: Color,
    /// Outline thickness multiplier.
    pub outline_multiplier: f32,
    /// Whether gradients should be disabled.
    pub disable_gradients: bool,
    /// Whether shadows should be disabled.
    pub disable_shadows: bool,
    /// Whether background images should be hidden.
    pub hide_background_images: bool,
    /// Minimum contrast ratio to enforce.
    pub min_contrast_ratio: f32,
}

impl HighContrastMode {
    /// Creates a new high contrast mode manager.
    pub fn new() -> Self {
        Self {
            active: false,
            auto_detect: true,
            forced: None,
            background: Color::BLACK,
            foreground: Color::WHITE,
            highlight: Color::new(0.0, 0.5, 1.0, 1.0),
            highlight_text: Color::WHITE,
            disabled_text: Color::new(0.5, 0.5, 0.5, 1.0),
            border: Color::WHITE,
            link: Color::new(0.5, 0.8, 1.0, 1.0),
            button_face: Color::new(0.1, 0.1, 0.1, 1.0),
            button_text: Color::WHITE,
            outline_multiplier: 2.0,
            disable_gradients: true,
            disable_shadows: true,
            hide_background_images: true,
            min_contrast_ratio: 7.0,
        }
    }

    /// Returns true if high contrast mode is currently active.
    pub fn is_high_contrast(&self) -> bool {
        if let Some(forced) = self.forced {
            return forced;
        }
        if self.auto_detect {
            return self.detect_os_high_contrast();
        }
        self.active
    }

    /// Detects the OS high contrast setting.
    ///
    /// Platform-specific implementation. Returns false as a fallback on
    /// unsupported platforms.
    fn detect_os_high_contrast(&self) -> bool {
        // On Windows, we would check SystemParametersInfo(SPI_GETHIGHCONTRAST).
        // This is a compile-time stub; real implementation depends on platform.
        #[cfg(target_os = "windows")]
        {
            // Placeholder: would call Win32 API.
            false
        }
        #[cfg(not(target_os = "windows"))]
        {
            false
        }
    }

    /// Forces high contrast mode on or off.
    pub fn set_forced(&mut self, enabled: Option<bool>) {
        self.forced = enabled;
    }

    /// Returns the appropriate color for a given semantic purpose.
    pub fn get_color(&self, purpose: HighContrastColor) -> Color {
        match purpose {
            HighContrastColor::Background => self.background,
            HighContrastColor::Foreground => self.foreground,
            HighContrastColor::Highlight => self.highlight,
            HighContrastColor::HighlightText => self.highlight_text,
            HighContrastColor::DisabledText => self.disabled_text,
            HighContrastColor::Border => self.border,
            HighContrastColor::Link => self.link,
            HighContrastColor::ButtonFace => self.button_face,
            HighContrastColor::ButtonText => self.button_text,
        }
    }

    /// Calculates the contrast ratio between two colors (WCAG formula).
    pub fn contrast_ratio(color1: Color, color2: Color) -> f32 {
        let lum1 = Self::relative_luminance(color1);
        let lum2 = Self::relative_luminance(color2);
        let (lighter, darker) = if lum1 > lum2 {
            (lum1, lum2)
        } else {
            (lum2, lum1)
        };
        (lighter + 0.05) / (darker + 0.05)
    }

    /// Calculates relative luminance per WCAG 2.0.
    fn relative_luminance(color: Color) -> f32 {
        fn linearize(c: f32) -> f32 {
            if c <= 0.03928 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }
        let r = linearize(color.r);
        let g = linearize(color.g);
        let b = linearize(color.b);
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    /// Checks if two colors meet the minimum contrast requirement.
    pub fn meets_contrast(&self, fg: Color, bg: Color) -> bool {
        Self::contrast_ratio(fg, bg) >= self.min_contrast_ratio
    }
}

impl Default for HighContrastMode {
    fn default() -> Self {
        Self::new()
    }
}

/// Semantic color purpose for high contrast mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighContrastColor {
    Background,
    Foreground,
    Highlight,
    HighlightText,
    DisabledText,
    Border,
    Link,
    ButtonFace,
    ButtonText,
}

// ---------------------------------------------------------------------------
// ReducedMotion
// ---------------------------------------------------------------------------

/// Reduced motion preference support.
///
/// Detects the OS "reduce motion" / "reduce animations" preference and provides
/// a way for the UI system to respect it. When reduced motion is preferred,
/// animations are either disabled or shortened, and transitions use instant
/// or very fast timings.
#[derive(Debug, Clone)]
pub struct ReducedMotion {
    /// Whether reduced motion is currently active.
    pub active: bool,
    /// Whether to auto-detect from OS settings.
    pub auto_detect: bool,
    /// Forced on/off override.
    pub forced: Option<bool>,
    /// Maximum animation duration when reduced motion is active (seconds).
    pub max_duration: f32,
    /// Whether to completely disable animations (vs. shorten them).
    pub disable_completely: bool,
    /// Animation speed multiplier (1.0 = normal, higher = faster).
    pub speed_multiplier: f32,
    /// Whether parallax effects should be disabled.
    pub disable_parallax: bool,
    /// Whether auto-playing videos/animations should be paused.
    pub pause_autoplay: bool,
}

impl ReducedMotion {
    /// Creates a new reduced motion manager.
    pub fn new() -> Self {
        Self {
            active: false,
            auto_detect: true,
            forced: None,
            max_duration: 0.1,
            disable_completely: false,
            speed_multiplier: 5.0,
            disable_parallax: true,
            pause_autoplay: true,
        }
    }

    /// Returns true if reduced motion is preferred.
    pub fn prefers_reduced_motion(&self) -> bool {
        if let Some(forced) = self.forced {
            return forced;
        }
        if self.auto_detect {
            return self.detect_os_preference();
        }
        self.active
    }

    /// Detects the OS "reduce motion" preference.
    fn detect_os_preference(&self) -> bool {
        // On macOS: NSWorkspace.accessibilityDisplayShouldReduceMotion
        // On Windows: SPI_GETCLIENTAREAANIMATION
        // On Linux: org.gnome.desktop.interface.enable-animations
        // This is a compile-time stub.
        #[cfg(target_os = "windows")]
        {
            false
        }
        #[cfg(not(target_os = "windows"))]
        {
            false
        }
    }

    /// Forces reduced motion on or off.
    pub fn set_forced(&mut self, enabled: Option<bool>) {
        self.forced = enabled;
    }

    /// Returns the effective animation duration, capped by the preference.
    pub fn effective_duration(&self, desired_duration: f32) -> f32 {
        if self.prefers_reduced_motion() {
            if self.disable_completely {
                0.0
            } else {
                desired_duration.min(self.max_duration) / self.speed_multiplier
            }
        } else {
            desired_duration
        }
    }

    /// Returns the effective animation speed multiplier.
    pub fn effective_speed(&self) -> f32 {
        if self.prefers_reduced_motion() {
            self.speed_multiplier
        } else {
            1.0
        }
    }

    /// Returns true if a specific animation type should be played.
    pub fn should_animate(&self, animation_type: AnimationType) -> bool {
        if !self.prefers_reduced_motion() {
            return true;
        }
        match animation_type {
            AnimationType::Essential => true,
            AnimationType::Decorative => false,
            AnimationType::Interactive => !self.disable_completely,
            AnimationType::Parallax => !self.disable_parallax,
            AnimationType::Loading => true,
        }
    }
}

impl Default for ReducedMotion {
    fn default() -> Self {
        Self::new()
    }
}

/// Classification of animation types for reduced motion decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationType {
    /// Essential animations (e.g., loading spinner) - always play.
    Essential,
    /// Decorative animations (e.g., particle effects) - skip when reduced.
    Decorative,
    /// Interactive feedback animations (e.g., button press) - may be shortened.
    Interactive,
    /// Parallax/scrolling effects - skip when reduced.
    Parallax,
    /// Loading/progress indicators - always play.
    Loading,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accessible_role_focusable() {
        assert!(AccessibleRole::Button.is_focusable());
        assert!(AccessibleRole::TextInput.is_focusable());
        assert!(!AccessibleRole::Text.is_focusable());
        assert!(!AccessibleRole::Image.is_focusable());
    }

    #[test]
    fn test_accessible_state_summary() {
        let mut state = AccessibleState::new();
        assert_eq!(state.summary(), "normal");
        state.focused = true;
        state.selected = true;
        assert!(state.summary().contains("focused"));
        assert!(state.summary().contains("selected"));
    }

    #[test]
    fn test_focus_scope_navigation() {
        let id_a = UIId::new(1, 0);
        let id_b = UIId::new(2, 0);
        let id_c = UIId::new(3, 0);

        let mut scope = FocusScope::new(1, "test", FocusDirection::Horizontal);
        scope.add_item(id_a);
        scope.add_item(id_b);
        scope.add_item(id_c);

        assert_eq!(scope.focus_next(), Some(id_b)); // 0 -> 1
        assert_eq!(scope.focus_next(), Some(id_c)); // 1 -> 2
        assert_eq!(scope.focus_next(), Some(id_a)); // 2 -> 0 (wrap)
    }

    #[test]
    fn test_keyboard_navigation_tab() {
        let mut nav = KeyboardNavigation::new();
        let id_a = UIId::new(1, 0);
        let id_b = UIId::new(2, 0);
        let id_c = UIId::new(3, 0);
        nav.set_tab_order(vec![id_a, id_b, id_c]);

        let focused = nav.handle_tab(false);
        assert_eq!(focused, Some(id_a));
        let focused = nav.handle_tab(false);
        assert_eq!(focused, Some(id_b));
    }

    #[test]
    fn test_high_contrast_ratio() {
        let ratio = HighContrastMode::contrast_ratio(Color::WHITE, Color::BLACK);
        assert!(ratio > 20.0); // Should be 21:1.
    }

    #[test]
    fn test_reduced_motion_duration() {
        let mut rm = ReducedMotion::new();
        rm.forced = Some(true);
        let effective = rm.effective_duration(1.0);
        assert!(effective < 0.5);
    }

    #[test]
    fn test_accessibility_manager_announcements() {
        let mut mgr = AccessibilityManager::new();
        mgr.announce_polite("Hello");
        mgr.announce_assertive("Alert!");
        let first = mgr.next_announcement().unwrap();
        assert_eq!(first.text, "Hello");
        let second = mgr.next_announcement().unwrap();
        assert_eq!(second.text, "Alert!");
    }

    #[test]
    fn test_accessible_node_announce() {
        let node = AccessibleNode::new(UIId::new(1, 0), AccessibleRole::Button, "Save");
        let text = node.announce();
        assert!(text.contains("Save"));
        assert!(text.contains("Button"));
    }
}
