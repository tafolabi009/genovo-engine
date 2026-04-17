// engine/editor/src/editor_shortcuts.rs
//
// Keyboard shortcut system for the Genovo editor.
//
// Provides a flexible keyboard shortcut system:
//
// - **Shortcut registry** -- Register and look up shortcuts.
// - **Key combination matching** -- Match key + modifier combinations.
// - **Context-sensitive** -- Different shortcuts in different editor contexts.
// - **Shortcut display** -- Generate display strings for menus.
// - **Rebindable shortcuts** -- Users can customize key bindings.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Key codes
// ---------------------------------------------------------------------------

/// Virtual key code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Key0, Key1, Key2, Key3, Key4, Key5, Key6, Key7, Key8, Key9,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    Space, Enter, Escape, Tab, Backspace, Delete, Insert,
    Home, End, PageUp, PageDown,
    Left, Right, Up, Down,
    Plus, Minus, Equals,
    LeftBracket, RightBracket, Backslash, Semicolon, Quote, Comma, Period, Slash, Tilde,
}

impl fmt::Display for KeyCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A => write!(f, "A"), Self::B => write!(f, "B"), Self::C => write!(f, "C"),
            Self::D => write!(f, "D"), Self::E => write!(f, "E"), Self::F => write!(f, "F"),
            Self::G => write!(f, "G"), Self::H => write!(f, "H"), Self::I => write!(f, "I"),
            Self::J => write!(f, "J"), Self::K => write!(f, "K"), Self::L => write!(f, "L"),
            Self::M => write!(f, "M"), Self::N => write!(f, "N"), Self::O => write!(f, "O"),
            Self::P => write!(f, "P"), Self::Q => write!(f, "Q"), Self::R => write!(f, "R"),
            Self::S => write!(f, "S"), Self::T => write!(f, "T"), Self::U => write!(f, "U"),
            Self::V => write!(f, "V"), Self::W => write!(f, "W"), Self::X => write!(f, "X"),
            Self::Y => write!(f, "Y"), Self::Z => write!(f, "Z"),
            Self::Key0 => write!(f, "0"), Self::Key1 => write!(f, "1"), Self::Key2 => write!(f, "2"),
            Self::Key3 => write!(f, "3"), Self::Key4 => write!(f, "4"), Self::Key5 => write!(f, "5"),
            Self::Key6 => write!(f, "6"), Self::Key7 => write!(f, "7"), Self::Key8 => write!(f, "8"),
            Self::Key9 => write!(f, "9"),
            Self::F1 => write!(f, "F1"), Self::F2 => write!(f, "F2"), Self::F3 => write!(f, "F3"),
            Self::F4 => write!(f, "F4"), Self::F5 => write!(f, "F5"), Self::F6 => write!(f, "F6"),
            Self::F7 => write!(f, "F7"), Self::F8 => write!(f, "F8"), Self::F9 => write!(f, "F9"),
            Self::F10 => write!(f, "F10"), Self::F11 => write!(f, "F11"), Self::F12 => write!(f, "F12"),
            Self::Space => write!(f, "Space"), Self::Enter => write!(f, "Enter"),
            Self::Escape => write!(f, "Esc"), Self::Tab => write!(f, "Tab"),
            Self::Backspace => write!(f, "Backspace"), Self::Delete => write!(f, "Del"),
            Self::Insert => write!(f, "Ins"),
            Self::Home => write!(f, "Home"), Self::End => write!(f, "End"),
            Self::PageUp => write!(f, "PgUp"), Self::PageDown => write!(f, "PgDn"),
            Self::Left => write!(f, "Left"), Self::Right => write!(f, "Right"),
            Self::Up => write!(f, "Up"), Self::Down => write!(f, "Down"),
            Self::Plus => write!(f, "+"), Self::Minus => write!(f, "-"), Self::Equals => write!(f, "="),
            _ => write!(f, "?"),
        }
    }
}

/// Modifier keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Modifiers {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub super_key: bool,
}

impl Modifiers {
    pub const NONE: Self = Self { ctrl: false, shift: false, alt: false, super_key: false };
    pub const CTRL: Self = Self { ctrl: true, shift: false, alt: false, super_key: false };
    pub const SHIFT: Self = Self { ctrl: false, shift: true, alt: false, super_key: false };
    pub const ALT: Self = Self { ctrl: false, shift: false, alt: true, super_key: false };
    pub const CTRL_SHIFT: Self = Self { ctrl: true, shift: true, alt: false, super_key: false };
    pub const CTRL_ALT: Self = Self { ctrl: true, shift: false, alt: true, super_key: false };
}

impl fmt::Display for Modifiers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.ctrl { parts.push("Ctrl"); }
        if self.shift { parts.push("Shift"); }
        if self.alt { parts.push("Alt"); }
        if self.super_key { parts.push("Super"); }
        write!(f, "{}", parts.join("+"))
    }
}

// ---------------------------------------------------------------------------
// Key combination
// ---------------------------------------------------------------------------

/// A key combination (modifiers + key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyCombo {
    pub key: KeyCode,
    pub modifiers: Modifiers,
}

impl KeyCombo {
    pub const fn new(key: KeyCode, modifiers: Modifiers) -> Self {
        Self { key, modifiers }
    }

    /// Display string for menus (e.g., "Ctrl+S").
    pub fn display_string(&self) -> String {
        let mod_str = self.modifiers.to_string();
        if mod_str.is_empty() {
            self.key.to_string()
        } else {
            format!("{}+{}", mod_str, self.key)
        }
    }
}

impl fmt::Display for KeyCombo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_string())
    }
}

// ---------------------------------------------------------------------------
// Shortcut context
// ---------------------------------------------------------------------------

/// Editor context that determines which shortcuts are active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShortcutContext {
    Global,
    Viewport,
    Hierarchy,
    Inspector,
    AssetBrowser,
    NodeGraph,
    Timeline,
    Console,
    TextEditor,
    Custom(u32),
}

impl fmt::Display for ShortcutContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => write!(f, "Global"),
            Self::Viewport => write!(f, "Viewport"),
            Self::Hierarchy => write!(f, "Hierarchy"),
            Self::Inspector => write!(f, "Inspector"),
            Self::AssetBrowser => write!(f, "Asset Browser"),
            Self::NodeGraph => write!(f, "Node Graph"),
            Self::Timeline => write!(f, "Timeline"),
            Self::Console => write!(f, "Console"),
            Self::TextEditor => write!(f, "Text Editor"),
            Self::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

// ---------------------------------------------------------------------------
// Shortcut action
// ---------------------------------------------------------------------------

/// Unique identifier for a shortcut action.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ActionId(pub String);

impl ActionId {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

impl fmt::Display for ActionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A registered shortcut.
#[derive(Debug, Clone)]
pub struct Shortcut {
    pub action: ActionId,
    pub display_name: String,
    pub description: String,
    pub default_combo: KeyCombo,
    pub current_combo: KeyCombo,
    pub context: ShortcutContext,
    pub category: String,
    pub enabled: bool,
}

impl Shortcut {
    pub fn new(action: &str, name: &str, combo: KeyCombo, context: ShortcutContext) -> Self {
        Self {
            action: ActionId::new(action),
            display_name: name.to_string(),
            description: String::new(),
            default_combo: combo,
            current_combo: combo,
            context,
            category: "General".to_string(),
            enabled: true,
        }
    }

    pub fn is_default(&self) -> bool {
        self.current_combo == self.default_combo
    }

    pub fn reset_to_default(&mut self) {
        self.current_combo = self.default_combo;
    }
}

// ---------------------------------------------------------------------------
// Shortcut registry
// ---------------------------------------------------------------------------

/// Events from the shortcut system.
#[derive(Debug, Clone)]
pub enum ShortcutEvent {
    ActionTriggered(ActionId),
    ShortcutRebound { action: ActionId, old: KeyCombo, new: KeyCombo },
    ConflictDetected { action1: ActionId, action2: ActionId, combo: KeyCombo },
}

/// The keyboard shortcut registry and dispatcher.
pub struct ShortcutRegistry {
    shortcuts: Vec<Shortcut>,
    action_index: HashMap<ActionId, usize>,
    combo_index: HashMap<(ShortcutContext, KeyCombo), ActionId>,
    active_context: ShortcutContext,
    events: Vec<ShortcutEvent>,
}

impl ShortcutRegistry {
    pub fn new() -> Self {
        Self {
            shortcuts: Vec::new(),
            action_index: HashMap::new(),
            combo_index: HashMap::new(),
            active_context: ShortcutContext::Global,
            events: Vec::new(),
        }
    }

    /// Register a shortcut.
    pub fn register(&mut self, shortcut: Shortcut) {
        let idx = self.shortcuts.len();
        let action = shortcut.action.clone();
        let combo = shortcut.current_combo;
        let context = shortcut.context;
        self.combo_index.insert((context, combo), action.clone());
        self.action_index.insert(action, idx);
        self.shortcuts.push(shortcut);
    }

    /// Set the active context.
    pub fn set_context(&mut self, context: ShortcutContext) {
        self.active_context = context;
    }

    /// Get the active context.
    pub fn active_context(&self) -> ShortcutContext {
        self.active_context
    }

    /// Check if a key combo triggers an action in the current context.
    pub fn check(&mut self, combo: KeyCombo) -> Option<ActionId> {
        // Check context-specific first, then global.
        if let Some(action) = self.combo_index.get(&(self.active_context, combo)) {
            let action = action.clone();
            if self.is_enabled(&action) {
                self.events.push(ShortcutEvent::ActionTriggered(action.clone()));
                return Some(action);
            }
        }
        if self.active_context != ShortcutContext::Global {
            if let Some(action) = self.combo_index.get(&(ShortcutContext::Global, combo)) {
                let action = action.clone();
                if self.is_enabled(&action) {
                    self.events.push(ShortcutEvent::ActionTriggered(action.clone()));
                    return Some(action);
                }
            }
        }
        None
    }

    /// Rebind a shortcut.
    pub fn rebind(&mut self, action: &ActionId, new_combo: KeyCombo) -> Result<(), String> {
        let idx = match self.action_index.get(action) {
            Some(&i) => i,
            None => return Err("Action not found".into()),
        };

        // Check for conflicts.
        let context = self.shortcuts[idx].context;
        if let Some(existing) = self.combo_index.get(&(context, new_combo)) {
            if existing != action {
                self.events.push(ShortcutEvent::ConflictDetected {
                    action1: action.clone(),
                    action2: existing.clone(),
                    combo: new_combo,
                });
                return Err(format!("Combo {} already bound to {}", new_combo, existing));
            }
        }

        let old_combo = self.shortcuts[idx].current_combo;
        self.combo_index.remove(&(context, old_combo));
        self.shortcuts[idx].current_combo = new_combo;
        self.combo_index.insert((context, new_combo), action.clone());

        self.events.push(ShortcutEvent::ShortcutRebound {
            action: action.clone(),
            old: old_combo,
            new: new_combo,
        });

        Ok(())
    }

    /// Reset a shortcut to its default binding.
    pub fn reset_to_default(&mut self, action: &ActionId) {
        if let Some(&idx) = self.action_index.get(action) {
            let context = self.shortcuts[idx].context;
            let old = self.shortcuts[idx].current_combo;
            self.combo_index.remove(&(context, old));
            self.shortcuts[idx].reset_to_default();
            let new = self.shortcuts[idx].current_combo;
            self.combo_index.insert((context, new), action.clone());
        }
    }

    /// Reset all shortcuts to defaults.
    pub fn reset_all_to_defaults(&mut self) {
        self.combo_index.clear();
        for shortcut in &mut self.shortcuts {
            shortcut.reset_to_default();
            self.combo_index.insert(
                (shortcut.context, shortcut.current_combo),
                shortcut.action.clone(),
            );
        }
    }

    /// Get a shortcut by action ID.
    pub fn get(&self, action: &ActionId) -> Option<&Shortcut> {
        self.action_index.get(action).map(|&i| &self.shortcuts[i])
    }

    /// Check if an action is enabled.
    pub fn is_enabled(&self, action: &ActionId) -> bool {
        self.get(action).map(|s| s.enabled).unwrap_or(false)
    }

    /// Get the display string for an action's shortcut.
    pub fn display_string(&self, action: &ActionId) -> String {
        self.get(action).map(|s| s.current_combo.display_string()).unwrap_or_default()
    }

    /// Get all shortcuts in a category.
    pub fn shortcuts_in_category(&self, category: &str) -> Vec<&Shortcut> {
        self.shortcuts.iter().filter(|s| s.category == category).collect()
    }

    /// Get all categories.
    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self.shortcuts.iter().map(|s| s.category.clone()).collect();
        cats.sort();
        cats.dedup();
        cats
    }

    /// Get all shortcuts.
    pub fn all_shortcuts(&self) -> &[Shortcut] {
        &self.shortcuts
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<ShortcutEvent> {
        std::mem::take(&mut self.events)
    }

    /// Register default editor shortcuts.
    pub fn register_defaults(&mut self) {
        let defaults = vec![
            Shortcut::new("file.save", "Save", KeyCombo::new(KeyCode::S, Modifiers::CTRL), ShortcutContext::Global),
            Shortcut::new("file.open", "Open", KeyCombo::new(KeyCode::O, Modifiers::CTRL), ShortcutContext::Global),
            Shortcut::new("file.new", "New", KeyCombo::new(KeyCode::N, Modifiers::CTRL), ShortcutContext::Global),
            Shortcut::new("edit.undo", "Undo", KeyCombo::new(KeyCode::Z, Modifiers::CTRL), ShortcutContext::Global),
            Shortcut::new("edit.redo", "Redo", KeyCombo::new(KeyCode::Y, Modifiers::CTRL), ShortcutContext::Global),
            Shortcut::new("edit.copy", "Copy", KeyCombo::new(KeyCode::C, Modifiers::CTRL), ShortcutContext::Global),
            Shortcut::new("edit.paste", "Paste", KeyCombo::new(KeyCode::V, Modifiers::CTRL), ShortcutContext::Global),
            Shortcut::new("edit.cut", "Cut", KeyCombo::new(KeyCode::X, Modifiers::CTRL), ShortcutContext::Global),
            Shortcut::new("edit.delete", "Delete", KeyCombo::new(KeyCode::Delete, Modifiers::NONE), ShortcutContext::Global),
            Shortcut::new("edit.select_all", "Select All", KeyCombo::new(KeyCode::A, Modifiers::CTRL), ShortcutContext::Global),
            Shortcut::new("edit.duplicate", "Duplicate", KeyCombo::new(KeyCode::D, Modifiers::CTRL), ShortcutContext::Global),
            Shortcut::new("view.focus", "Focus Selected", KeyCombo::new(KeyCode::F, Modifiers::NONE), ShortcutContext::Viewport),
            Shortcut::new("transform.translate", "Translate", KeyCombo::new(KeyCode::W, Modifiers::NONE), ShortcutContext::Viewport),
            Shortcut::new("transform.rotate", "Rotate", KeyCombo::new(KeyCode::E, Modifiers::NONE), ShortcutContext::Viewport),
            Shortcut::new("transform.scale", "Scale", KeyCombo::new(KeyCode::R, Modifiers::NONE), ShortcutContext::Viewport),
            Shortcut::new("play.play", "Play", KeyCombo::new(KeyCode::F5, Modifiers::NONE), ShortcutContext::Global),
            Shortcut::new("play.pause", "Pause", KeyCombo::new(KeyCode::F6, Modifiers::NONE), ShortcutContext::Global),
            Shortcut::new("play.stop", "Stop", KeyCombo::new(KeyCode::F5, Modifiers::SHIFT), ShortcutContext::Global),
        ];
        for s in defaults {
            self.register(s);
        }
    }
}

impl Default for ShortcutRegistry {
    fn default() -> Self {
        let mut reg = Self::new();
        reg.register_defaults();
        reg
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortcut_trigger() {
        let mut reg = ShortcutRegistry::default();
        let result = reg.check(KeyCombo::new(KeyCode::S, Modifiers::CTRL));
        assert_eq!(result, Some(ActionId::new("file.save")));
    }

    #[test]
    fn test_context_specific() {
        let mut reg = ShortcutRegistry::default();
        reg.set_context(ShortcutContext::Viewport);
        let result = reg.check(KeyCombo::new(KeyCode::W, Modifiers::NONE));
        assert_eq!(result, Some(ActionId::new("transform.translate")));
    }

    #[test]
    fn test_rebind() {
        let mut reg = ShortcutRegistry::default();
        let action = ActionId::new("file.save");
        let new_combo = KeyCombo::new(KeyCode::S, Modifiers::CTRL_SHIFT);
        reg.rebind(&action, new_combo).unwrap();
        assert_eq!(reg.display_string(&action), "Ctrl+Shift+S");
    }

    #[test]
    fn test_display_string() {
        let combo = KeyCombo::new(KeyCode::S, Modifiers::CTRL);
        assert_eq!(combo.display_string(), "Ctrl+S");
    }
}
