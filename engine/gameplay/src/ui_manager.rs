// engine/gameplay/src/ui_manager.rs
//
// HUD/UI management: UI stack, transitions, input routing, scaling, safe area margins.

use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x+self.y*r.y+self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self{x:self.y*r.z-self.z*r.y,y:self.z*r.x-self.x*r.z,z:self.x*r.y-self.y*r.x} }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f32 { self.dot(self) }
    pub fn normalize(self) -> Self { let l=self.length(); if l<1e-12{Self::ZERO}else{Self{x:self.x/l,y:self.y/l,z:self.z/l}} }
    pub fn scale(self, s: f32) -> Self { Self{x:self.x*s,y:self.y*s,z:self.z*s} }
    pub fn add(self, r: Self) -> Self { Self{x:self.x+r.x,y:self.y+r.y,z:self.z+r.z} }
    pub fn sub(self, r: Self) -> Self { Self{x:self.x-r.x,y:self.y-r.y,z:self.z-r.z} }
    pub fn neg(self) -> Self { Self{x:-self.x,y:-self.y,z:-self.z} }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScreenId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType { None, Fade, Slide, Scale, Custom }

#[derive(Debug, Clone)]
pub struct UITransition {
    pub transition_type: TransitionType,
    pub duration: f32,
    pub elapsed: f32,
    pub is_entering: bool,
}

impl UITransition {
    pub fn new(tt: TransitionType, duration: f32, entering: bool) -> Self {
        Self { transition_type: tt, duration, elapsed: 0.0, is_entering: entering }
    }
    pub fn progress(&self) -> f32 { (self.elapsed / self.duration.max(0.001)).clamp(0.0, 1.0) }
    pub fn is_complete(&self) -> bool { self.elapsed >= self.duration }
    pub fn update(&mut self, dt: f32) { self.elapsed += dt; }
}

#[derive(Debug, Clone)]
pub struct SafeAreaMargins { pub top: f32, pub bottom: f32, pub left: f32, pub right: f32 }
impl Default for SafeAreaMargins {
    fn default() -> Self { Self { top: 0.0, bottom: 0.0, left: 0.0, right: 0.0 } }
}

#[derive(Debug, Clone)]
pub struct UIScaling {
    pub reference_width: f32,
    pub reference_height: f32,
    pub current_width: f32,
    pub current_height: f32,
    pub scale_mode: ScaleMode,
    pub dpi_scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMode { FitWidth, FitHeight, FitBoth, Stretch }

impl UIScaling {
    pub fn new(ref_w: f32, ref_h: f32) -> Self {
        Self { reference_width: ref_w, reference_height: ref_h, current_width: ref_w, current_height: ref_h, scale_mode: ScaleMode::FitBoth, dpi_scale: 1.0 }
    }
    pub fn scale_factor(&self) -> f32 {
        let sx = self.current_width / self.reference_width;
        let sy = self.current_height / self.reference_height;
        match self.scale_mode {
            ScaleMode::FitWidth => sx * self.dpi_scale,
            ScaleMode::FitHeight => sy * self.dpi_scale,
            ScaleMode::FitBoth => sx.min(sy) * self.dpi_scale,
            ScaleMode::Stretch => 1.0 * self.dpi_scale,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UIInputResult { Consumed, Ignored }

/// A UI screen on the stack.
#[derive(Debug, Clone)]
pub struct UIScreen {
    pub id: ScreenId,
    pub name: String,
    pub is_modal: bool,
    pub blocks_input: bool,
    pub blocks_rendering: bool,
    pub transition: Option<UITransition>,
    pub is_visible: bool,
    pub opacity: f32,
    pub data: HashMap<String, String>,
}

/// UI management system.
pub struct UIManager {
    screen_stack: Vec<UIScreen>,
    next_id: u32,
    pub scaling: UIScaling,
    pub safe_area: SafeAreaMargins,
    pub input_enabled: bool,
    pub debug_draw: bool,
    pub transition_default_duration: f32,
    pub default_transition: TransitionType,
}

impl UIManager {
    pub fn new(ref_width: f32, ref_height: f32) -> Self {
        Self {
            screen_stack: Vec::new(),
            next_id: 1,
            scaling: UIScaling::new(ref_width, ref_height),
            safe_area: SafeAreaMargins::default(),
            input_enabled: true,
            debug_draw: false,
            transition_default_duration: 0.3,
            default_transition: TransitionType::Fade,
        }
    }

    pub fn push_screen(&mut self, name: &str, modal: bool) -> ScreenId {
        let id = ScreenId(self.next_id);
        self.next_id += 1;
        let screen = UIScreen {
            id, name: name.to_string(), is_modal: modal,
            blocks_input: modal, blocks_rendering: false,
            transition: Some(UITransition::new(self.default_transition, self.transition_default_duration, true)),
            is_visible: true, opacity: 0.0, data: HashMap::new(),
        };
        self.screen_stack.push(screen);
        id
    }

    pub fn pop_screen(&mut self) -> Option<ScreenId> {
        if let Some(screen) = self.screen_stack.last_mut() {
            screen.transition = Some(UITransition::new(self.default_transition, self.transition_default_duration, false));
            Some(screen.id)
        } else { None }
    }

    pub fn update(&mut self, dt: f32) {
        for screen in &mut self.screen_stack {
            if let Some(ref mut transition) = screen.transition {
                transition.update(dt);
                screen.opacity = if transition.is_entering { transition.progress() } else { 1.0 - transition.progress() };
            } else {
                screen.opacity = 1.0;
            }
        }
        // Remove completed exit transitions
        self.screen_stack.retain(|s| {
            if let Some(ref t) = s.transition {
                !(!t.is_entering && t.is_complete())
            } else { true }
        });
    }

    pub fn top_screen(&self) -> Option<&UIScreen> { self.screen_stack.last() }
    pub fn screen_count(&self) -> usize { self.screen_stack.len() }
    pub fn clear_all(&mut self) { self.screen_stack.clear(); }

    pub fn route_input(&self) -> Option<ScreenId> {
        if !self.input_enabled { return None; }
        for screen in self.screen_stack.iter().rev() {
            if screen.is_visible && screen.opacity > 0.5 {
                return Some(screen.id);
            }
            if screen.blocks_input { return None; }
        }
        None
    }

    pub fn set_resolution(&mut self, width: f32, height: f32) {
        self.scaling.current_width = width;
        self.scaling.current_height = height;
    }

    pub fn set_safe_area(&mut self, top: f32, bottom: f32, left: f32, right: f32) {
        self.safe_area = SafeAreaMargins { top, bottom, left, right };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ui_manager() {
        let mut mgr = UIManager::new(1920.0, 1080.0);
        let id = mgr.push_screen("MainMenu", false);
        assert_eq!(mgr.screen_count(), 1);
        assert!(mgr.top_screen().is_some());
        mgr.update(0.5);
        assert!(mgr.top_screen().unwrap().opacity > 0.0);
    }
    #[test]
    fn test_scaling() {
        let s = UIScaling::new(1920.0, 1080.0);
        assert!((s.scale_factor() - 1.0).abs() < 0.01);
    }
}

