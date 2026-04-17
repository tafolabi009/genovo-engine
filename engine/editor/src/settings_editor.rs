//! Settings/preferences editor: categorized settings, search, reset to defaults,
//! import/export settings.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum SettingValue {
    Bool(bool), Int(i32), Float(f32), String(String),
    Color([f32; 4]), Vec2([f32; 2]), Vec3([f32; 3]),
    Enum(usize, Vec<String>), Path(String), KeyBinding(String),
}

impl SettingValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool", Self::Int(_) => "int", Self::Float(_) => "float",
            Self::String(_) => "string", Self::Color(_) => "color",
            Self::Vec2(_) => "vec2", Self::Vec3(_) => "vec3",
            Self::Enum(_, _) => "enum", Self::Path(_) => "path",
            Self::KeyBinding(_) => "keybinding",
        }
    }
    pub fn as_bool(&self) -> Option<bool> { match self { Self::Bool(v) => Some(*v), _ => None } }
    pub fn as_int(&self) -> Option<i32> { match self { Self::Int(v) => Some(*v), _ => None } }
    pub fn as_float(&self) -> Option<f32> { match self { Self::Float(v) => Some(*v), _ => None } }
    pub fn as_string(&self) -> Option<&str> { match self { Self::String(v) => Some(v), Self::Path(v) => Some(v), _ => None } }
}

impl fmt::Display for SettingValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{}", v), Self::Int(v) => write!(f, "{}", v),
            Self::Float(v) => write!(f, "{:.3}", v), Self::String(v) => write!(f, "{}", v),
            Self::Color(c) => write!(f, "({:.2},{:.2},{:.2},{:.2})", c[0], c[1], c[2], c[3]),
            Self::Vec2(v) => write!(f, "({:.2},{:.2})", v[0], v[1]),
            Self::Vec3(v) => write!(f, "({:.2},{:.2},{:.2})", v[0], v[1], v[2]),
            Self::Enum(idx, opts) => write!(f, "{}", opts.get(*idx).map(|s| s.as_str()).unwrap_or("?")),
            Self::Path(v) => write!(f, "{}", v),
            Self::KeyBinding(v) => write!(f, "{}", v),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SettingDef {
    pub key: String,
    pub display_name: String,
    pub category: String,
    pub subcategory: Option<String>,
    pub value: SettingValue,
    pub default_value: SettingValue,
    pub description: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub requires_restart: bool,
    pub hidden: bool,
    pub modified: bool,
    pub tags: Vec<String>,
}

impl SettingDef {
    pub fn new(key: impl Into<String>, name: impl Into<String>, cat: impl Into<String>, value: SettingValue) -> Self {
        let v = value.clone();
        Self {
            key: key.into(), display_name: name.into(), category: cat.into(),
            subcategory: None, value, default_value: v, description: String::new(),
            min: None, max: None, step: None, requires_restart: false,
            hidden: false, modified: false, tags: Vec::new(),
        }
    }
    pub fn with_range(mut self, min: f64, max: f64) -> Self { self.min = Some(min); self.max = Some(max); self }
    pub fn with_description(mut self, desc: impl Into<String>) -> Self { self.description = desc.into(); self }
    pub fn with_restart(mut self) -> Self { self.requires_restart = true; self }
    pub fn is_default(&self) -> bool { self.value == self.default_value }
    pub fn reset(&mut self) { self.value = self.default_value.clone(); self.modified = false; }
}

#[derive(Debug, Clone)]
pub enum SettingsEvent {
    ValueChanged(String, SettingValue),
    CategorySelected(String),
    SearchChanged(String),
    ResetToDefaults(Option<String>),
    Imported(String),
    Exported(String),
    Saved,
}

pub struct SettingsEditorState {
    pub settings: Vec<SettingDef>,
    pub categories: Vec<String>,
    pub selected_category: Option<String>,
    pub search_text: String,
    pub events: Vec<SettingsEvent>,
    pub show_modified_only: bool,
    pub show_advanced: bool,
    pub dirty: bool,
    pub profiles: HashMap<String, Vec<(String, SettingValue)>>,
    pub active_profile: String,
}

impl SettingsEditorState {
    pub fn new() -> Self {
        Self {
            settings: Vec::new(), categories: Vec::new(),
            selected_category: None, search_text: String::new(),
            events: Vec::new(), show_modified_only: false,
            show_advanced: false, dirty: false,
            profiles: HashMap::new(), active_profile: "Default".to_string(),
        }
    }

    pub fn add_setting(&mut self, setting: SettingDef) {
        if !self.categories.contains(&setting.category) {
            self.categories.push(setting.category.clone());
            self.categories.sort();
        }
        self.settings.push(setting);
    }

    pub fn set_value(&mut self, key: &str, value: SettingValue) {
        if let Some(s) = self.settings.iter_mut().find(|s| s.key == key) {
            s.value = value.clone();
            s.modified = s.value != s.default_value;
            self.dirty = true;
            self.events.push(SettingsEvent::ValueChanged(key.to_string(), value));
        }
    }

    pub fn get_value(&self, key: &str) -> Option<&SettingValue> {
        self.settings.iter().find(|s| s.key == key).map(|s| &s.value)
    }

    pub fn reset_to_defaults(&mut self, category: Option<&str>) {
        for s in &mut self.settings {
            if category.map_or(true, |c| s.category == c) { s.reset(); }
        }
        self.dirty = true;
        self.events.push(SettingsEvent::ResetToDefaults(category.map(|s| s.to_string())));
    }

    pub fn select_category(&mut self, cat: impl Into<String>) {
        let cat = cat.into();
        self.selected_category = Some(cat.clone());
        self.events.push(SettingsEvent::CategorySelected(cat));
    }

    pub fn search(&mut self, text: impl Into<String>) {
        self.search_text = text.into();
        self.events.push(SettingsEvent::SearchChanged(self.search_text.clone()));
    }

    pub fn filtered_settings(&self) -> Vec<&SettingDef> {
        self.settings.iter().filter(|s| {
            if s.hidden && !self.show_advanced { return false; }
            if self.show_modified_only && !s.modified { return false; }
            if let Some(ref cat) = self.selected_category { if s.category != *cat { return false; } }
            if !self.search_text.is_empty() {
                let lower = self.search_text.to_lowercase();
                if !s.display_name.to_lowercase().contains(&lower)
                    && !s.key.to_lowercase().contains(&lower)
                    && !s.description.to_lowercase().contains(&lower) { return false; }
            }
            true
        }).collect()
    }

    pub fn modified_count(&self) -> usize { self.settings.iter().filter(|s| s.modified).count() }
    pub fn setting_count(&self) -> usize { self.settings.len() }
    pub fn category_count(&self) -> usize { self.categories.len() }

    pub fn export_to_string(&self) -> String {
        let mut out = String::new();
        for s in &self.settings {
            out.push_str(&format!("{}={}\n", s.key, s.value));
        }
        out
    }

    pub fn save_profile(&mut self, name: impl Into<String>) {
        let name = name.into();
        let values: Vec<(String, SettingValue)> = self.settings.iter()
            .map(|s| (s.key.clone(), s.value.clone())).collect();
        self.profiles.insert(name, values);
    }

    pub fn load_profile(&mut self, name: &str) -> bool {
        if let Some(values) = self.profiles.get(name).cloned() {
            for (key, val) in values { self.set_value(&key, val); }
            self.active_profile = name.to_string();
            true
        } else { false }
    }

    pub fn drain_events(&mut self) -> Vec<SettingsEvent> { std::mem::take(&mut self.events) }
}

impl Default for SettingsEditorState {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn settings_basic() {
        let mut editor = SettingsEditorState::new();
        editor.add_setting(SettingDef::new("render.vsync", "VSync", "Rendering", SettingValue::Bool(true)));
        editor.add_setting(SettingDef::new("render.fov", "FOV", "Rendering", SettingValue::Float(90.0)).with_range(60.0, 120.0));
        assert_eq!(editor.setting_count(), 2);
        assert_eq!(editor.category_count(), 1);
        editor.set_value("render.fov", SettingValue::Float(100.0));
        assert_eq!(editor.modified_count(), 1);
        editor.reset_to_defaults(None);
        assert_eq!(editor.modified_count(), 0);
    }
    #[test]
    fn search_filter() {
        let mut editor = SettingsEditorState::new();
        editor.add_setting(SettingDef::new("audio.volume", "Volume", "Audio", SettingValue::Float(1.0)));
        editor.add_setting(SettingDef::new("render.fov", "FOV", "Rendering", SettingValue::Float(90.0)));
        editor.search("vol");
        let filtered = editor.filtered_settings();
        assert_eq!(filtered.len(), 1);
    }
}
