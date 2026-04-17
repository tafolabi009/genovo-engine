// engine/audio/src/audio_spatializer_v2.rs
// Enhanced spatial audio: HRTF, room simulation, distance attenuation, Doppler, occlusion.
use std::collections::HashMap;
use std::f32::consts::PI;
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

pub type SourceId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttenuationCurve { Linear, InverseDistance, InverseSquare, Logarithmic, Custom }

#[derive(Debug, Clone)]
pub struct SpatialSource {
    pub id: SourceId, pub position: Vec3, pub velocity: Vec3,
    pub min_distance: f32, pub max_distance: f32,
    pub attenuation: AttenuationCurve,
    pub rolloff_factor: f32, pub doppler_factor: f32,
    pub cone_inner_angle: f32, pub cone_outer_angle: f32, pub cone_outer_gain: f32,
    pub direction: Vec3, pub is_3d: bool,
    pub occlusion: f32, pub obstruction: f32,
    pub room_send: f32, pub direct_gain: f32,
}

impl Default for SpatialSource {
    fn default() -> Self {
        Self { id: 0, position: Vec3::ZERO, velocity: Vec3::ZERO, min_distance: 1.0, max_distance: 100.0, attenuation: AttenuationCurve::InverseDistance, rolloff_factor: 1.0, doppler_factor: 1.0, cone_inner_angle: 360.0, cone_outer_angle: 360.0, cone_outer_gain: 0.0, direction: Vec3::new(0.0,0.0,-1.0), is_3d: true, occlusion: 0.0, obstruction: 0.0, room_send: 0.3, direct_gain: 1.0 }
    }
}

#[derive(Debug, Clone)]
pub struct SpatialListener {
    pub position: Vec3, pub velocity: Vec3,
    pub forward: Vec3, pub up: Vec3, pub right: Vec3,
}

impl Default for SpatialListener {
    fn default() -> Self {
        Self { position: Vec3::ZERO, velocity: Vec3::ZERO, forward: Vec3::new(0.0,0.0,-1.0), up: Vec3::new(0.0,1.0,0.0), right: Vec3::new(1.0,0.0,0.0) }
    }
}

#[derive(Debug, Clone)]
pub struct RoomProperties {
    pub size: Vec3, pub rt60: f32,
    pub early_reflections: Vec<EarlyReflection>,
    pub late_reverb_gain: f32, pub late_reverb_delay_ms: f32,
    pub hf_damping: f32, pub diffusion: f32,
    pub wall_absorption: [f32; 6], // 6 walls
}

impl Default for RoomProperties {
    fn default() -> Self {
        Self { size: Vec3::new(10.0, 3.0, 10.0), rt60: 0.8, early_reflections: Vec::new(), late_reverb_gain: 0.5, late_reverb_delay_ms: 20.0, hf_damping: 0.5, diffusion: 0.7, wall_absorption: [0.3; 6] }
    }
}

#[derive(Debug, Clone)]
pub struct EarlyReflection { pub delay_ms: f32, pub gain: f32, pub direction: Vec3 }

#[derive(Debug, Clone)]
pub struct HrtfProfile {
    pub left_delays: Vec<f32>,
    pub right_delays: Vec<f32>,
    pub left_gains: Vec<f32>,
    pub right_gains: Vec<f32>,
    pub elevation_count: u32,
    pub azimuth_count: u32,
}

impl HrtfProfile {
    pub fn default_profile() -> Self {
        let count = 72; // 5-degree resolution
        let mut left_gains = Vec::with_capacity(count);
        let mut right_gains = Vec::with_capacity(count);
        let mut left_delays = Vec::with_capacity(count);
        let mut right_delays = Vec::with_capacity(count);
        for i in 0..count {
            let azimuth = (i as f32 / count as f32) * 2.0 * PI;
            // Simple HRTF model: ITD (interaural time difference) and ILD (level difference)
            let itd = 0.00065 * azimuth.sin(); // ~0.65ms max ITD
            let ild_db = 10.0 * azimuth.sin(); // ~10dB max ILD
            let ild_linear = 10.0_f32.powf(ild_db / 20.0);
            left_delays.push(if azimuth.sin() > 0.0 { 0.0 } else { itd.abs() });
            right_delays.push(if azimuth.sin() < 0.0 { 0.0 } else { itd.abs() });
            left_gains.push(if azimuth.sin() > 0.0 { 1.0 / ild_linear } else { 1.0 });
            right_gains.push(if azimuth.sin() < 0.0 { ild_linear } else { 1.0 });
        }
        Self { left_delays, right_delays, left_gains, right_gains, elevation_count: 1, azimuth_count: count as u32 }
    }

    pub fn lookup(&self, azimuth: f32, _elevation: f32) -> (f32, f32, f32, f32) {
        let idx = ((azimuth / (2.0 * PI)) * self.azimuth_count as f32) as usize % self.azimuth_count as usize;
        (self.left_gains[idx], self.right_gains[idx], self.left_delays[idx], self.right_delays[idx])
    }
}

#[derive(Debug, Clone)]
pub struct SpatializationResult { pub left_gain: f32, pub right_gain: f32, pub distance_gain: f32, pub doppler_pitch: f32, pub occlusion_filter: f32, pub room_contribution: f32 }

pub struct AudioSpatializerV2 {
    pub listener: SpatialListener,
    pub sources: HashMap<SourceId, SpatialSource>,
    pub room: RoomProperties,
    pub hrtf: HrtfProfile,
    pub enable_hrtf: bool,
    pub enable_doppler: bool,
    pub enable_room: bool,
    pub speed_of_sound: f32,
    next_id: SourceId,
}

impl AudioSpatializerV2 {
    pub fn new() -> Self {
        Self { listener: SpatialListener::default(), sources: HashMap::new(), room: RoomProperties::default(), hrtf: HrtfProfile::default_profile(), enable_hrtf: true, enable_doppler: true, enable_room: true, speed_of_sound: 343.0, next_id: 1 }
    }

    pub fn add_source(&mut self, source: SpatialSource) -> SourceId {
        let id = self.next_id; self.next_id += 1;
        self.sources.insert(id, SpatialSource { id, ..source });
        id
    }

    pub fn remove_source(&mut self, id: SourceId) { self.sources.remove(&id); }

    pub fn update_source(&mut self, id: SourceId, position: Vec3, velocity: Vec3) {
        if let Some(s) = self.sources.get_mut(&id) { s.position = position; s.velocity = velocity; }
    }

    pub fn update_listener(&mut self, position: Vec3, velocity: Vec3, forward: Vec3, up: Vec3) {
        self.listener.position = position;
        self.listener.velocity = velocity;
        self.listener.forward = forward.normalize();
        self.listener.up = up.normalize();
        self.listener.right = forward.cross(up).normalize();
    }

    pub fn spatialize(&self, source_id: SourceId) -> Option<SpatializationResult> {
        let source = self.sources.get(&source_id)?;
        if !source.is_3d { return Some(SpatializationResult { left_gain: 1.0, right_gain: 1.0, distance_gain: 1.0, doppler_pitch: 1.0, occlusion_filter: 1.0, room_contribution: 0.0 }); }

        let to_source = source.position.sub(self.listener.position);
        let distance = to_source.length();
        let dir = if distance > 1e-6 { to_source.scale(1.0 / distance) } else { Vec3::new(0.0, 0.0, -1.0) };

        // Distance attenuation
        let dist_gain = match source.attenuation {
            AttenuationCurve::Linear => 1.0 - ((distance - source.min_distance) / (source.max_distance - source.min_distance).max(0.001)).clamp(0.0, 1.0),
            AttenuationCurve::InverseDistance => source.min_distance / (source.min_distance + source.rolloff_factor * (distance - source.min_distance).max(0.0)),
            AttenuationCurve::InverseSquare => { let d = distance.max(source.min_distance); (source.min_distance * source.min_distance) / (d * d) }
            AttenuationCurve::Logarithmic => { let d = distance.max(source.min_distance); 1.0 - (d / source.min_distance).ln() / (source.max_distance / source.min_distance).ln() }
            AttenuationCurve::Custom => 1.0,
        }.clamp(0.0, 1.0);

        // Panning / HRTF
        let (left_gain, right_gain) = if self.enable_hrtf {
            let azimuth = dir.dot(self.listener.right).atan2(dir.dot(self.listener.forward));
            let (lg, rg, _, _) = self.hrtf.lookup(azimuth, 0.0);
            (lg, rg)
        } else {
            let pan = dir.dot(self.listener.right).clamp(-1.0, 1.0);
            let l = ((1.0 - pan) * 0.5 * PI * 0.5).cos();
            let r = ((1.0 + pan) * 0.5 * PI * 0.5).cos();
            (l, r)
        };

        // Doppler
        let doppler_pitch = if self.enable_doppler && source.doppler_factor > 0.0 {
            let vs_r = source.velocity.dot(dir);
            let vl_r = self.listener.velocity.dot(dir);
            ((self.speed_of_sound - vl_r) / (self.speed_of_sound - vs_r).max(0.1)).clamp(0.5, 2.0)
        } else { 1.0 };

        // Occlusion
        let occlusion_filter = 1.0 - source.occlusion * 0.8;

        // Cone attenuation
        let cone_gain = if source.cone_outer_angle < 360.0 {
            let source_to_listener = dir.neg();
            let cos_angle = source.direction.dot(source_to_listener);
            let angle = cos_angle.acos() * 180.0 / PI;
            let inner = source.cone_inner_angle * 0.5;
            let outer = source.cone_outer_angle * 0.5;
            if angle <= inner { 1.0 }
            else if angle >= outer { source.cone_outer_gain }
            else { 1.0 + (source.cone_outer_gain - 1.0) * (angle - inner) / (outer - inner).max(0.001) }
        } else { 1.0 };

        // Room contribution
        let room_contribution = if self.enable_room { source.room_send * dist_gain } else { 0.0 };

        Some(SpatializationResult {
            left_gain: left_gain * dist_gain * cone_gain * source.direct_gain,
            right_gain: right_gain * dist_gain * cone_gain * source.direct_gain,
            distance_gain: dist_gain,
            doppler_pitch,
            occlusion_filter,
            room_contribution,
        })
    }

    /// Generate early reflections for a source.
    pub fn compute_early_reflections(&self, source_pos: Vec3) -> Vec<EarlyReflection> {
        let mut reflections = Vec::new();
        let room_half = self.room.size.scale(0.5);
        // Reflect off each wall (image source method, simplified)
        let walls = [
            (Vec3::new(room_half.x, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 0),
            (Vec3::new(-room_half.x, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1),
            (Vec3::new(0.0, room_half.y, 0.0), Vec3::new(0.0, -1.0, 0.0), 2),
            (Vec3::new(0.0, -room_half.y, 0.0), Vec3::new(0.0, 1.0, 0.0), 3),
            (Vec3::new(0.0, 0.0, room_half.z), Vec3::new(0.0, 0.0, -1.0), 4),
            (Vec3::new(0.0, 0.0, -room_half.z), Vec3::new(0.0, 0.0, 1.0), 5),
        ];
        for (wall_pos, wall_normal, wall_idx) in &walls {
            let reflected = source_pos.sub(wall_normal.scale(2.0 * source_pos.sub(*wall_pos).dot(*wall_normal)));
            let path_length = reflected.distance(self.listener.position);
            let delay_ms = path_length / self.speed_of_sound * 1000.0;
            let absorption = self.room.wall_absorption[*wall_idx];
            let gain = (1.0 - absorption) / (path_length.max(1.0));
            let dir = reflected.sub(self.listener.position).normalize();
            reflections.push(EarlyReflection { delay_ms, gain: gain.min(0.5), direction: dir });
        }
        reflections
    }

    pub fn source_count(&self) -> usize { self.sources.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_spatialize() {
        let mut sp = AudioSpatializerV2::new();
        sp.update_listener(Vec3::ZERO, Vec3::ZERO, Vec3::new(0.0,0.0,-1.0), Vec3::new(0.0,1.0,0.0));
        let id = sp.add_source(SpatialSource { position: Vec3::new(5.0, 0.0, 0.0), ..Default::default() });
        let result = sp.spatialize(id);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.right_gain > r.left_gain); // source is to the right
    }
    #[test]
    fn test_distance_attenuation() {
        let mut sp = AudioSpatializerV2::new();
        let near = sp.add_source(SpatialSource { position: Vec3::new(2.0, 0.0, 0.0), ..Default::default() });
        let far = sp.add_source(SpatialSource { position: Vec3::new(50.0, 0.0, 0.0), ..Default::default() });
        let rn = sp.spatialize(near).unwrap();
        let rf = sp.spatialize(far).unwrap();
        assert!(rn.distance_gain > rf.distance_gain);
    }
    #[test]
    fn test_early_reflections() {
        let sp = AudioSpatializerV2::new();
        let reflections = sp.compute_early_reflections(Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(reflections.len(), 6);
    }
}
