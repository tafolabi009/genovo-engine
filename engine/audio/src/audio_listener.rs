// engine/audio/src/audio_listener.rs
//
// Audio listener system for the Genovo engine.
//
// Provides 3D audio listener management with position, orientation, velocity
// for Doppler effect, multiple listeners for split-screen, and listener priority.

pub const MAX_LISTENERS: usize = 4;
pub const SPEED_OF_SOUND: f32 = 343.0;
pub const DEFAULT_DOPPLER_FACTOR: f32 = 1.0;
pub const DEFAULT_DISTANCE_MODEL_MAX: f32 = 500.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceModel { Linear, InverseDistance, InverseDistanceClamped, Exponential }

impl DistanceModel {
    pub fn attenuate(&self, distance: f32, min_dist: f32, max_dist: f32, rolloff: f32) -> f32 {
        match self {
            Self::Linear => { let d = distance.clamp(min_dist, max_dist); 1.0 - rolloff * (d - min_dist) / (max_dist - min_dist) }
            Self::InverseDistance => { min_dist / (min_dist + rolloff * (distance - min_dist).max(0.0)) }
            Self::InverseDistanceClamped => { let d = distance.clamp(min_dist, max_dist); min_dist / (min_dist + rolloff * (d - min_dist)) }
            Self::Exponential => { (distance.max(min_dist) / min_dist).powf(-rolloff) }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ListenerTransform {
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],
    pub velocity: [f32; 3],
}

impl Default for ListenerTransform {
    fn default() -> Self { Self { position: [0.0; 3], forward: [0.0, 0.0, -1.0], up: [0.0, 1.0, 0.0], velocity: [0.0; 3] } }
}

impl ListenerTransform {
    pub fn right(&self) -> [f32; 3] { cross(self.forward, self.up) }
    pub fn direction_to(&self, target: [f32; 3]) -> [f32; 3] { normalize(sub(target, self.position)) }
    pub fn distance_to(&self, target: [f32; 3]) -> f32 { length(sub(target, self.position)) }

    pub fn compute_panning(&self, source_pos: [f32; 3]) -> StereoPan {
        let to_source = sub(source_pos, self.position);
        let dist = length(to_source);
        if dist < 0.001 { return StereoPan { left: 1.0, right: 1.0 }; }
        let dir = [to_source[0] / dist, to_source[1] / dist, to_source[2] / dist];
        let right = self.right();
        let pan = dot(dir, right);
        let angle = pan.asin();
        let left = ((std::f32::consts::FRAC_PI_4 - angle) * 0.5).cos();
        let right_gain = ((std::f32::consts::FRAC_PI_4 + angle) * 0.5).cos();
        StereoPan { left, right: right_gain }
    }

    pub fn compute_doppler(&self, source_pos: [f32; 3], source_vel: [f32; 3], doppler_factor: f32) -> f32 {
        let to_source = sub(source_pos, self.position);
        let dist = length(to_source);
        if dist < 0.001 { return 1.0; }
        let dir = [to_source[0] / dist, to_source[1] / dist, to_source[2] / dist];
        let listener_speed = dot(self.velocity, dir);
        let source_speed = dot(source_vel, dir);
        let c = SPEED_OF_SOUND;
        let denominator = c + doppler_factor * source_speed;
        let numerator = c + doppler_factor * listener_speed;
        if denominator.abs() < 0.01 { 1.0 } else { (numerator / denominator).clamp(0.5, 2.0) }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StereoPan { pub left: f32, pub right: f32 }

#[derive(Debug, Clone)]
pub struct AudioListenerConfig {
    pub distance_model: DistanceModel,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff_factor: f32,
    pub doppler_factor: f32,
    pub cone_inner_angle: f32,
    pub cone_outer_angle: f32,
    pub cone_outer_gain: f32,
}

impl Default for AudioListenerConfig {
    fn default() -> Self { Self { distance_model: DistanceModel::InverseDistanceClamped, min_distance: 1.0, max_distance: DEFAULT_DISTANCE_MODEL_MAX, rolloff_factor: 1.0, doppler_factor: DEFAULT_DOPPLER_FACTOR, cone_inner_angle: 360.0, cone_outer_angle: 360.0, cone_outer_gain: 0.0 } }
}

/// A single audio listener.
#[derive(Debug, Clone)]
pub struct Listener {
    pub id: u32,
    pub transform: ListenerTransform,
    pub config: AudioListenerConfig,
    pub priority: i32,
    pub active: bool,
    pub viewport_index: u32,
    pub gain: f32,
}

impl Listener {
    pub fn new(id: u32) -> Self {
        Self { id, transform: ListenerTransform::default(), config: AudioListenerConfig::default(), priority: 0, active: true, viewport_index: 0, gain: 1.0 }
    }

    pub fn evaluate_source(&self, source_pos: [f32; 3], source_vel: [f32; 3]) -> SourceEvaluation {
        let distance = self.transform.distance_to(source_pos);
        let attenuation = self.config.distance_model.attenuate(distance, self.config.min_distance, self.config.max_distance, self.config.rolloff_factor);
        let panning = self.transform.compute_panning(source_pos);
        let doppler = self.transform.compute_doppler(source_pos, source_vel, self.config.doppler_factor);
        SourceEvaluation { distance, attenuation: attenuation * self.gain, panning, doppler_pitch_shift: doppler, is_behind: dot(self.transform.direction_to(source_pos), self.transform.forward) < 0.0 }
    }
}

#[derive(Debug, Clone)]
pub struct SourceEvaluation { pub distance: f32, pub attenuation: f32, pub panning: StereoPan, pub doppler_pitch_shift: f32, pub is_behind: bool }

/// Manages multiple audio listeners for split-screen.
#[derive(Debug)]
pub struct AudioListenerManager {
    pub listeners: Vec<Listener>,
    pub max_listeners: usize,
    pub primary_listener: u32,
    next_id: u32,
}

impl AudioListenerManager {
    pub fn new() -> Self { Self { listeners: Vec::new(), max_listeners: MAX_LISTENERS, primary_listener: 0, next_id: 0 } }

    pub fn add_listener(&mut self) -> u32 {
        if self.listeners.len() >= self.max_listeners { return u32::MAX; }
        let id = self.next_id; self.next_id += 1;
        let mut l = Listener::new(id);
        l.viewport_index = self.listeners.len() as u32;
        self.listeners.push(l);
        if self.listeners.len() == 1 { self.primary_listener = id; }
        id
    }

    pub fn remove_listener(&mut self, id: u32) { self.listeners.retain(|l| l.id != id); }
    pub fn get(&self, id: u32) -> Option<&Listener> { self.listeners.iter().find(|l| l.id == id) }
    pub fn get_mut(&mut self, id: u32) -> Option<&mut Listener> { self.listeners.iter_mut().find(|l| l.id == id) }
    pub fn primary(&self) -> Option<&Listener> { self.get(self.primary_listener) }

    pub fn update_transform(&mut self, id: u32, transform: ListenerTransform) {
        if let Some(l) = self.get_mut(id) { l.transform = transform; }
    }

    pub fn evaluate_source_all(&self, source_pos: [f32; 3], source_vel: [f32; 3]) -> Vec<(u32, SourceEvaluation)> {
        self.listeners.iter().filter(|l| l.active).map(|l| (l.id, l.evaluate_source(source_pos, source_vel))).collect()
    }

    pub fn evaluate_source_primary(&self, source_pos: [f32; 3], source_vel: [f32; 3]) -> Option<SourceEvaluation> {
        self.primary().map(|l| l.evaluate_source(source_pos, source_vel))
    }

    pub fn active_count(&self) -> usize { self.listeners.iter().filter(|l| l.active).count() }
}

fn dot(a: [f32;3], b: [f32;3]) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
fn sub(a: [f32;3], b: [f32;3]) -> [f32;3] { [a[0]-b[0],a[1]-b[1],a[2]-b[2]] }
fn cross(a: [f32;3], b: [f32;3]) -> [f32;3] { [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]] }
fn length(v: [f32;3]) -> f32 { dot(v,v).sqrt() }
fn normalize(v: [f32;3]) -> [f32;3] { let l=length(v); if l<1e-8{[0.0;3]}else{[v[0]/l,v[1]/l,v[2]/l]} }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_distance_models() { assert!((DistanceModel::Linear.attenuate(5.0,1.0,10.0,1.0)-0.556).abs()<0.1); }
    #[test] fn test_panning() { let l = ListenerTransform::default(); let p = l.compute_panning([1.0,0.0,0.0]); assert!(p.right > p.left); }
    #[test] fn test_doppler() { let l = ListenerTransform::default(); let d = l.compute_doppler([0.0,0.0,-10.0],[0.0,0.0,50.0],1.0); assert!(d > 1.0); }
    #[test] fn test_manager() { let mut m = AudioListenerManager::new(); let id = m.add_listener(); assert_eq!(m.active_count(),1); m.remove_listener(id); assert_eq!(m.active_count(),0); }
}
