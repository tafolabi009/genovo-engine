// engine/gameplay/src/hud_system.rs
//
// HUD framework: health bar, ammo display, crosshair, minimap integration,
// compass, notification toasts, damage indicators (screen edge), interaction
// prompts, score display, objective markers.

use std::collections::VecDeque;

// --- Color ---
#[derive(Debug, Clone, Copy)]
pub struct HudColor { pub r: f32, pub g: f32, pub b: f32, pub a: f32 }

impl HudColor {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const RED: Self = Self { r: 1.0, g: 0.2, b: 0.2, a: 1.0 };
    pub const GREEN: Self = Self { r: 0.2, g: 1.0, b: 0.2, a: 1.0 };
    pub const YELLOW: Self = Self { r: 1.0, g: 1.0, b: 0.2, a: 1.0 };
    pub const BLUE: Self = Self { r: 0.2, g: 0.4, b: 1.0, a: 1.0 };
    pub const TRANSPARENT: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self { r, g, b, a } }
    pub fn with_alpha(self, a: f32) -> Self { Self { a, ..self } }
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}

// --- Anchor ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HudAnchor { TopLeft, TopCenter, TopRight, CenterLeft, Center, CenterRight, BottomLeft, BottomCenter, BottomRight }

impl HudAnchor {
    pub fn offset(&self, screen_w: f32, screen_h: f32) -> (f32, f32) {
        match self {
            Self::TopLeft => (0.0, 0.0), Self::TopCenter => (screen_w * 0.5, 0.0), Self::TopRight => (screen_w, 0.0),
            Self::CenterLeft => (0.0, screen_h * 0.5), Self::Center => (screen_w * 0.5, screen_h * 0.5), Self::CenterRight => (screen_w, screen_h * 0.5),
            Self::BottomLeft => (0.0, screen_h), Self::BottomCenter => (screen_w * 0.5, screen_h), Self::BottomRight => (screen_w, screen_h),
        }
    }
}

// --- Health bar ---
#[derive(Debug, Clone)]
pub struct HealthBarWidget {
    pub current_health: f32,
    pub max_health: f32,
    pub armor: f32,
    pub max_armor: f32,
    pub position: (f32, f32),
    pub size: (f32, f32),
    pub anchor: HudAnchor,
    pub health_color: HudColor,
    pub armor_color: HudColor,
    pub background_color: HudColor,
    pub low_health_threshold: f32,
    pub pulse_when_low: bool,
    pub show_numeric: bool,
    pub visible: bool,
}

impl Default for HealthBarWidget {
    fn default() -> Self {
        Self {
            current_health: 100.0, max_health: 100.0, armor: 0.0, max_armor: 100.0,
            position: (20.0, -60.0), size: (250.0, 20.0), anchor: HudAnchor::BottomLeft,
            health_color: HudColor::GREEN, armor_color: HudColor::BLUE,
            background_color: HudColor::new(0.1, 0.1, 0.1, 0.8),
            low_health_threshold: 0.25, pulse_when_low: true, show_numeric: true, visible: true,
        }
    }
}

impl HealthBarWidget {
    pub fn health_ratio(&self) -> f32 { if self.max_health > 0.0 { (self.current_health / self.max_health).clamp(0.0, 1.0) } else { 0.0 } }
    pub fn armor_ratio(&self) -> f32 { if self.max_armor > 0.0 { (self.armor / self.max_armor).clamp(0.0, 1.0) } else { 0.0 } }
    pub fn is_low(&self) -> bool { self.health_ratio() < self.low_health_threshold }
    pub fn effective_color(&self) -> HudColor {
        let ratio = self.health_ratio();
        if ratio < 0.25 { HudColor::RED } else if ratio < 0.5 { HudColor::YELLOW } else { self.health_color }
    }
    pub fn set_health(&mut self, health: f32) { self.current_health = health.clamp(0.0, self.max_health); }
    pub fn set_armor(&mut self, armor: f32) { self.armor = armor.clamp(0.0, self.max_armor); }
}

// --- Ammo display ---
#[derive(Debug, Clone)]
pub struct AmmoDisplayWidget {
    pub current_ammo: u32,
    pub max_ammo: u32,
    pub reserve_ammo: u32,
    pub weapon_name: String,
    pub position: (f32, f32),
    pub anchor: HudAnchor,
    pub color: HudColor,
    pub low_ammo_color: HudColor,
    pub low_ammo_threshold: u32,
    pub show_weapon_name: bool,
    pub is_reloading: bool,
    pub reload_progress: f32,
    pub visible: bool,
}

impl Default for AmmoDisplayWidget {
    fn default() -> Self {
        Self {
            current_ammo: 30, max_ammo: 30, reserve_ammo: 120,
            weapon_name: "Rifle".into(),
            position: (-20.0, -60.0), anchor: HudAnchor::BottomRight,
            color: HudColor::WHITE, low_ammo_color: HudColor::RED,
            low_ammo_threshold: 5, show_weapon_name: true,
            is_reloading: false, reload_progress: 0.0, visible: true,
        }
    }
}

impl AmmoDisplayWidget {
    pub fn is_low(&self) -> bool { self.current_ammo <= self.low_ammo_threshold }
    pub fn is_empty(&self) -> bool { self.current_ammo == 0 }
    pub fn format_ammo(&self) -> String { format!("{} / {}", self.current_ammo, self.reserve_ammo) }
}

// --- Crosshair ---
#[derive(Debug, Clone)]
pub struct CrosshairWidget {
    pub style: CrosshairStyle,
    pub color: HudColor,
    pub outline_color: HudColor,
    pub size: f32,
    pub gap: f32,
    pub thickness: f32,
    pub dot: bool,
    pub dot_size: f32,
    pub spread: f32,
    pub dynamic_spread: bool,
    pub hit_marker_visible: bool,
    pub hit_marker_timer: f32,
    pub hit_marker_duration: f32,
    pub visible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrosshairStyle { Cross, Dot, Circle, Chevron, Custom }

impl Default for CrosshairWidget {
    fn default() -> Self {
        Self {
            style: CrosshairStyle::Cross, color: HudColor::WHITE, outline_color: HudColor::new(0.0, 0.0, 0.0, 0.8),
            size: 10.0, gap: 4.0, thickness: 2.0, dot: true, dot_size: 2.0,
            spread: 0.0, dynamic_spread: true,
            hit_marker_visible: false, hit_marker_timer: 0.0, hit_marker_duration: 0.2, visible: true,
        }
    }
}

impl CrosshairWidget {
    pub fn trigger_hit_marker(&mut self) { self.hit_marker_visible = true; self.hit_marker_timer = self.hit_marker_duration; }
    pub fn update(&mut self, dt: f32) {
        if self.hit_marker_timer > 0.0 { self.hit_marker_timer -= dt; if self.hit_marker_timer <= 0.0 { self.hit_marker_visible = false; } }
    }
}

// --- Damage indicator ---
#[derive(Debug, Clone)]
pub struct DamageIndicator {
    pub direction: f32,
    pub intensity: f32,
    pub timer: f32,
    pub duration: f32,
    pub color: HudColor,
}

impl DamageIndicator {
    pub fn new(direction_rad: f32, intensity: f32, duration: f32) -> Self {
        Self { direction: direction_rad, intensity: intensity.clamp(0.0, 1.0), timer: duration, duration, color: HudColor::RED }
    }
    pub fn alpha(&self) -> f32 { (self.timer / self.duration).clamp(0.0, 1.0) * self.intensity }
    pub fn is_expired(&self) -> bool { self.timer <= 0.0 }
}

// --- Toast notification ---
#[derive(Debug, Clone)]
pub struct ToastNotification {
    pub message: String,
    pub icon: Option<String>,
    pub color: HudColor,
    pub timer: f32,
    pub duration: f32,
    pub priority: i32,
}

impl ToastNotification {
    pub fn new(message: &str, duration: f32) -> Self {
        Self { message: message.to_string(), icon: None, color: HudColor::WHITE, timer: duration, duration, priority: 0 }
    }
    pub fn with_icon(mut self, icon: &str) -> Self { self.icon = Some(icon.to_string()); self }
    pub fn with_color(mut self, color: HudColor) -> Self { self.color = color; self }
    pub fn alpha(&self) -> f32 { if self.timer < 0.5 { (self.timer / 0.5).clamp(0.0, 1.0) } else { 1.0 } }
    pub fn is_expired(&self) -> bool { self.timer <= 0.0 }
}

// --- Interaction prompt ---
#[derive(Debug, Clone)]
pub struct InteractionPrompt {
    pub text: String,
    pub key_hint: String,
    pub world_position: [f32; 3],
    pub screen_position: Option<(f32, f32)>,
    pub visible: bool,
    pub progress: Option<f32>,
}

impl InteractionPrompt {
    pub fn new(text: &str, key: &str, world_pos: [f32; 3]) -> Self {
        Self { text: text.to_string(), key_hint: key.to_string(), world_position: world_pos, screen_position: None, visible: true, progress: None }
    }
}

// --- Objective marker ---
#[derive(Debug, Clone)]
pub struct ObjectiveMarker {
    pub id: u32,
    pub label: String,
    pub world_position: [f32; 3],
    pub screen_position: Option<(f32, f32)>,
    pub distance: f32,
    pub color: HudColor,
    pub icon: String,
    pub clamp_to_screen: bool,
    pub show_distance: bool,
    pub visible: bool,
    pub pulse: bool,
}

impl ObjectiveMarker {
    pub fn new(id: u32, label: &str, world_pos: [f32; 3], icon: &str) -> Self {
        Self {
            id, label: label.to_string(), world_position: world_pos, screen_position: None,
            distance: 0.0, color: HudColor::YELLOW, icon: icon.to_string(),
            clamp_to_screen: true, show_distance: true, visible: true, pulse: false,
        }
    }
    pub fn format_distance(&self) -> String {
        if self.distance > 1000.0 { format!("{:.1}km", self.distance / 1000.0) } else { format!("{:.0}m", self.distance) }
    }
}

// --- Compass ---
#[derive(Debug, Clone)]
pub struct CompassWidget {
    pub heading: f32,
    pub position: (f32, f32),
    pub width: f32,
    pub height: f32,
    pub anchor: HudAnchor,
    pub show_degrees: bool,
    pub show_cardinal: bool,
    pub markers: Vec<CompassMarker>,
    pub visible: bool,
}

#[derive(Debug, Clone)]
pub struct CompassMarker { pub label: String, pub bearing: f32, pub color: HudColor }

impl Default for CompassWidget {
    fn default() -> Self {
        Self {
            heading: 0.0, position: (0.0, 30.0), width: 400.0, height: 30.0,
            anchor: HudAnchor::TopCenter, show_degrees: true, show_cardinal: true,
            markers: Vec::new(), visible: true,
        }
    }
}

impl CompassWidget {
    pub fn set_heading(&mut self, degrees: f32) { self.heading = degrees % 360.0; }
    pub fn cardinal_direction(&self) -> &str {
        let h = ((self.heading % 360.0) + 360.0) % 360.0;
        match h as u32 {
            338..=360 | 0..=22 => "N", 23..=67 => "NE", 68..=112 => "E", 113..=157 => "SE",
            158..=202 => "S", 203..=247 => "SW", 248..=292 => "W", 293..=337 => "NW", _ => "N",
        }
    }
    pub fn add_marker(&mut self, label: &str, bearing: f32, color: HudColor) {
        self.markers.push(CompassMarker { label: label.to_string(), bearing, color });
    }
}

// --- Score display ---
#[derive(Debug, Clone)]
pub struct ScoreDisplayWidget {
    pub scores: Vec<ScoreEntry>,
    pub position: (f32, f32),
    pub anchor: HudAnchor,
    pub show_kills: bool,
    pub show_deaths: bool,
    pub show_ping: bool,
    pub visible: bool,
    pub expanded: bool,
}

#[derive(Debug, Clone)]
pub struct ScoreEntry {
    pub name: String,
    pub score: i32,
    pub kills: u32,
    pub deaths: u32,
    pub ping: u32,
    pub team_color: HudColor,
    pub is_local: bool,
}

impl Default for ScoreDisplayWidget {
    fn default() -> Self {
        Self {
            scores: Vec::new(), position: (0.0, 0.0), anchor: HudAnchor::Center,
            show_kills: true, show_deaths: true, show_ping: true, visible: false, expanded: false,
        }
    }
}

impl ScoreDisplayWidget {
    pub fn set_scores(&mut self, scores: Vec<ScoreEntry>) { self.scores = scores; self.scores.sort_by(|a, b| b.score.cmp(&a.score)); }
    pub fn toggle(&mut self) { self.visible = !self.visible; }
}

// --- HUD system ---
pub struct HudSystem {
    pub health_bar: HealthBarWidget,
    pub ammo_display: AmmoDisplayWidget,
    pub crosshair: CrosshairWidget,
    pub compass: CompassWidget,
    pub score_display: ScoreDisplayWidget,
    pub damage_indicators: Vec<DamageIndicator>,
    pub toasts: VecDeque<ToastNotification>,
    pub interaction_prompt: Option<InteractionPrompt>,
    pub objective_markers: Vec<ObjectiveMarker>,
    pub screen_width: f32,
    pub screen_height: f32,
    pub max_toasts: usize,
    pub max_damage_indicators: usize,
    pub global_opacity: f32,
    pub hud_scale: f32,
    pub visible: bool,
}

impl HudSystem {
    pub fn new(screen_width: f32, screen_height: f32) -> Self {
        Self {
            health_bar: HealthBarWidget::default(), ammo_display: AmmoDisplayWidget::default(),
            crosshair: CrosshairWidget::default(), compass: CompassWidget::default(),
            score_display: ScoreDisplayWidget::default(),
            damage_indicators: Vec::new(), toasts: VecDeque::new(),
            interaction_prompt: None, objective_markers: Vec::new(),
            screen_width, screen_height, max_toasts: 5, max_damage_indicators: 8,
            global_opacity: 1.0, hud_scale: 1.0, visible: true,
        }
    }

    pub fn update(&mut self, dt: f32) {
        if !self.visible { return; }
        self.crosshair.update(dt);
        self.damage_indicators.retain_mut(|d| { d.timer -= dt; !d.is_expired() });
        for toast in &mut self.toasts { toast.timer -= dt; }
        self.toasts.retain(|t| !t.is_expired());
    }

    pub fn add_damage_indicator(&mut self, direction: f32, intensity: f32) {
        if self.damage_indicators.len() >= self.max_damage_indicators { self.damage_indicators.remove(0); }
        self.damage_indicators.push(DamageIndicator::new(direction, intensity, 1.5));
    }

    pub fn show_toast(&mut self, notification: ToastNotification) {
        if self.toasts.len() >= self.max_toasts { self.toasts.pop_front(); }
        self.toasts.push_back(notification);
    }

    pub fn set_interaction_prompt(&mut self, prompt: Option<InteractionPrompt>) { self.interaction_prompt = prompt; }

    pub fn add_objective_marker(&mut self, marker: ObjectiveMarker) { self.objective_markers.push(marker); }
    pub fn remove_objective_marker(&mut self, id: u32) { self.objective_markers.retain(|m| m.id != id); }

    pub fn resize(&mut self, width: f32, height: f32) { self.screen_width = width; self.screen_height = height; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_bar() {
        let mut hb = HealthBarWidget::default();
        hb.set_health(25.0);
        assert!(hb.is_low());
        assert!((hb.health_ratio() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_toast_notification() {
        let mut hud = HudSystem::new(1920.0, 1080.0);
        hud.show_toast(ToastNotification::new("Test", 2.0));
        assert_eq!(hud.toasts.len(), 1);
        hud.update(3.0);
        assert_eq!(hud.toasts.len(), 0);
    }

    #[test]
    fn test_compass_direction() {
        let mut c = CompassWidget::default();
        c.set_heading(0.0);
        assert_eq!(c.cardinal_direction(), "N");
        c.set_heading(90.0);
        assert_eq!(c.cardinal_direction(), "E");
        c.set_heading(180.0);
        assert_eq!(c.cardinal_direction(), "S");
    }

    #[test]
    fn test_damage_indicator() {
        let mut hud = HudSystem::new(1920.0, 1080.0);
        hud.add_damage_indicator(1.5, 0.8);
        assert_eq!(hud.damage_indicators.len(), 1);
        hud.update(2.0);
        assert_eq!(hud.damage_indicators.len(), 0);
    }
}
