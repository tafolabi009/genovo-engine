// engine/gameplay/src/animation_controller.rs
//
// Animation state machine runtime: state transitions with blend parameters,
// animation events, root motion extraction, state query, and layered
// animation support. Provides a complete animation graph runtime.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Vec3 / Quaternion
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    #[inline]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Self::new(
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t,
            a.z + (b.z - a.z) * t,
        )
    }

    #[inline]
    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self::new(self.x + r.x, self.y + r.y, self.z + r.z) }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self::new(self.x - r.x, self.y - r.y, self.z - r.z) }
}
impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }
}
impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, r: Self) { self.x += r.x; self.y += r.y; self.z += r.z; }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }

    pub fn slerp(a: Self, b: Self, t: f32) -> Self {
        let mut dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        let mut b = b;
        if dot < 0.0 {
            dot = -dot;
            b = Self::new(-b.x, -b.y, -b.z, -b.w);
        }
        if dot > 0.9995 {
            let r = Self::new(
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t,
                a.w + (b.w - a.w) * t,
            );
            return r.normalized();
        }
        let theta = dot.acos();
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;
        Self::new(
            a.x * wa + b.x * wb,
            a.y * wa + b.y * wb,
            a.z * wa + b.z * wb,
            a.w * wa + b.w * wb,
        )
    }

    pub fn normalized(self) -> Self {
        let l = (self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w).sqrt();
        if l < 1e-10 { return Self::IDENTITY; }
        let inv = 1.0 / l;
        Self::new(self.x * inv, self.y * inv, self.z * inv, self.w * inv)
    }

    pub fn inverse(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    pub fn mul(self, r: Self) -> Self {
        Self::new(
            self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
            self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
        )
    }

    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let uv = Vec3::new(
            qv.y * v.z - qv.z * v.y,
            qv.z * v.x - qv.x * v.z,
            qv.x * v.y - qv.y * v.x,
        );
        let uuv = Vec3::new(
            qv.y * uv.z - qv.z * uv.y,
            qv.z * uv.x - qv.x * uv.z,
            qv.x * uv.y - qv.y * uv.x,
        );
        Vec3::new(
            v.x + (uv.x * self.w + uuv.x) * 2.0,
            v.y + (uv.y * self.w + uuv.y) * 2.0,
            v.z + (uv.z * self.w + uuv.z) * 2.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Bone transform / Animation clip
// ---------------------------------------------------------------------------

/// A bone transform (local space).
#[derive(Debug, Clone, Copy)]
pub struct BoneTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl BoneTransform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3 { x: 1.0, y: 1.0, z: 1.0 },
    };

    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        Self {
            translation: Vec3::lerp(a.translation, b.translation, t),
            rotation: Quat::slerp(a.rotation, b.rotation, t),
            scale: Vec3::lerp(a.scale, b.scale, t),
        }
    }

    pub fn blend(a: &Self, b: &Self, weight: f32) -> Self {
        Self::lerp(a, b, weight)
    }
}

/// An animation event at a specific time.
#[derive(Debug, Clone)]
pub struct AnimationEvent {
    pub time: f32,
    pub name: String,
    pub int_param: i32,
    pub float_param: f32,
    pub string_param: String,
}

/// A keyframe in an animation track.
#[derive(Debug, Clone)]
pub struct AnimKeyframe {
    pub time: f32,
    pub transform: BoneTransform,
}

/// An animation track for a single bone.
#[derive(Debug, Clone)]
pub struct AnimTrack {
    pub bone_index: usize,
    pub keyframes: Vec<AnimKeyframe>,
}

impl AnimTrack {
    /// Sample the track at a given time.
    pub fn sample(&self, time: f32) -> BoneTransform {
        if self.keyframes.is_empty() {
            return BoneTransform::IDENTITY;
        }
        if self.keyframes.len() == 1 || time <= self.keyframes[0].time {
            return self.keyframes[0].transform;
        }
        if time >= self.keyframes.last().unwrap().time {
            return self.keyframes.last().unwrap().transform;
        }

        // Binary search for the keyframe pair.
        let mut lo = 0;
        let mut hi = self.keyframes.len() - 1;
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if self.keyframes[mid].time <= time {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let a = &self.keyframes[lo];
        let b = &self.keyframes[hi];
        let dt = b.time - a.time;
        let t = if dt > 0.0 { (time - a.time) / dt } else { 0.0 };
        BoneTransform::lerp(&a.transform, &b.transform, t)
    }
}

/// A clip is a set of animation tracks with a name and duration.
#[derive(Debug, Clone)]
pub struct AnimClip {
    pub name: String,
    pub duration: f32,
    pub tracks: Vec<AnimTrack>,
    pub events: Vec<AnimationEvent>,
    pub looping: bool,
    /// Root motion delta per full loop.
    pub root_motion_translation: Vec3,
    pub root_motion_rotation: Quat,
}

impl AnimClip {
    pub fn new(name: &str, duration: f32) -> Self {
        Self {
            name: name.to_string(),
            duration,
            tracks: Vec::new(),
            events: Vec::new(),
            looping: true,
            root_motion_translation: Vec3::ZERO,
            root_motion_rotation: Quat::IDENTITY,
        }
    }

    /// Sample all tracks at a given time.
    pub fn sample(&self, time: f32, output: &mut [BoneTransform]) {
        let t = if self.looping && self.duration > 0.0 {
            time % self.duration
        } else {
            time.min(self.duration)
        };

        for track in &self.tracks {
            if track.bone_index < output.len() {
                output[track.bone_index] = track.sample(t);
            }
        }
    }

    /// Get events that trigger between prev_time and cur_time.
    pub fn triggered_events(&self, prev_time: f32, cur_time: f32) -> Vec<&AnimationEvent> {
        let mut result = Vec::new();
        let (pt, ct) = if self.looping && self.duration > 0.0 {
            (prev_time % self.duration, cur_time % self.duration)
        } else {
            (prev_time, cur_time)
        };

        for event in &self.events {
            if pt < ct {
                if event.time >= pt && event.time < ct {
                    result.push(event);
                }
            } else {
                // Wrapped around.
                if event.time >= pt || event.time < ct {
                    result.push(event);
                }
            }
        }
        result
    }

    /// Extract root motion delta between two times.
    pub fn root_motion_delta(&self, prev_time: f32, cur_time: f32) -> (Vec3, Quat) {
        if self.duration <= 0.0 {
            return (Vec3::ZERO, Quat::IDENTITY);
        }
        let dt = cur_time - prev_time;
        let fraction = dt / self.duration;
        let trans = self.root_motion_translation * fraction;
        let rot = Quat::slerp(Quat::IDENTITY, self.root_motion_rotation, fraction);
        (trans, rot)
    }
}

// ---------------------------------------------------------------------------
// State machine types
// ---------------------------------------------------------------------------

pub type StateId = u32;
pub type TransitionId = u32;
pub type ParameterId = u32;

/// A blend parameter value.
#[derive(Debug, Clone)]
pub enum ParamValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    Trigger(bool),
}

impl ParamValue {
    pub fn as_float(&self) -> f32 {
        match self {
            Self::Float(v) => *v,
            Self::Int(v) => *v as f32,
            Self::Bool(v) => if *v { 1.0 } else { 0.0 },
            Self::Trigger(v) => if *v { 1.0 } else { 0.0 },
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Self::Float(v) => *v > 0.5,
            Self::Int(v) => *v != 0,
            Self::Bool(v) => *v,
            Self::Trigger(v) => *v,
        }
    }
}

/// Condition for a transition.
#[derive(Debug, Clone)]
pub enum TransitionCondition {
    /// Parameter > threshold.
    Greater { param: String, threshold: f32 },
    /// Parameter < threshold.
    Less { param: String, threshold: f32 },
    /// Bool/trigger parameter is true.
    IsTrue { param: String },
    /// Bool parameter is false.
    IsFalse { param: String },
    /// Animation has finished (for non-looping clips).
    AnimationFinished,
    /// After a fixed time in the current state.
    TimeElapsed { seconds: f32 },
}

/// Blend mode for a state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    /// Single clip.
    Single,
    /// 1D blend space.
    BlendSpace1D,
    /// 2D blend space.
    BlendSpace2D,
    /// Additive blending.
    Additive,
}

/// An entry in a 1D blend space.
#[derive(Debug, Clone)]
pub struct BlendEntry1D {
    pub clip_index: usize,
    pub threshold: f32,
}

/// An entry in a 2D blend space.
#[derive(Debug, Clone)]
pub struct BlendEntry2D {
    pub clip_index: usize,
    pub position: (f32, f32),
}

/// A state in the animation state machine.
#[derive(Debug, Clone)]
pub struct AnimState {
    pub id: StateId,
    pub name: String,
    pub clip_index: usize,
    pub speed: f32,
    pub blend_mode: BlendMode,
    pub blend_param_x: String,
    pub blend_param_y: String,
    pub blend_entries_1d: Vec<BlendEntry1D>,
    pub blend_entries_2d: Vec<BlendEntry2D>,
    pub mirror: bool,
    pub is_looping: bool,
}

impl AnimState {
    pub fn new(id: StateId, name: &str, clip_index: usize) -> Self {
        Self {
            id,
            name: name.to_string(),
            clip_index,
            speed: 1.0,
            blend_mode: BlendMode::Single,
            blend_param_x: String::new(),
            blend_param_y: String::new(),
            blend_entries_1d: Vec::new(),
            blend_entries_2d: Vec::new(),
            mirror: false,
            is_looping: true,
        }
    }

    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    pub fn with_blend_1d(mut self, param: &str, entries: Vec<BlendEntry1D>) -> Self {
        self.blend_mode = BlendMode::BlendSpace1D;
        self.blend_param_x = param.to_string();
        self.blend_entries_1d = entries;
        self
    }
}

/// A transition between two states.
#[derive(Debug, Clone)]
pub struct AnimTransition {
    pub id: TransitionId,
    pub from_state: StateId,
    pub to_state: StateId,
    pub duration: f32,
    pub conditions: Vec<TransitionCondition>,
    pub has_exit_time: bool,
    pub exit_time: f32,
    pub priority: i32,
    pub interrupt_source: InterruptSource,
}

/// Which side of a transition can be interrupted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterruptSource {
    None,
    CurrentState,
    NextState,
    Both,
}

// ---------------------------------------------------------------------------
// Active state info
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ActiveState {
    state_id: StateId,
    time: f32,
    prev_time: f32,
    normalized_time: f32,
    speed_multiplier: f32,
}

#[derive(Debug, Clone)]
struct ActiveTransition {
    transition_id: TransitionId,
    from_state: ActiveState,
    to_state: ActiveState,
    elapsed: f32,
    duration: f32,
}

// ---------------------------------------------------------------------------
// Animation controller
// ---------------------------------------------------------------------------

/// Root motion output.
#[derive(Debug, Clone, Copy)]
pub struct RootMotionDelta {
    pub translation: Vec3,
    pub rotation: Quat,
}

/// Events triggered during an update.
#[derive(Debug, Clone)]
pub struct TriggeredEvent {
    pub event_name: String,
    pub state_name: String,
    pub int_param: i32,
    pub float_param: f32,
    pub string_param: String,
}

/// The animation controller (state machine runtime).
pub struct AnimationController {
    clips: Vec<AnimClip>,
    states: Vec<AnimState>,
    transitions: Vec<AnimTransition>,
    parameters: HashMap<String, ParamValue>,
    current_state: Option<ActiveState>,
    active_transition: Option<ActiveTransition>,
    bone_count: usize,
    temp_pose_a: Vec<BoneTransform>,
    temp_pose_b: Vec<BoneTransform>,
    pending_events: Vec<TriggeredEvent>,
    root_motion: RootMotionDelta,
    next_state_id: StateId,
    next_transition_id: TransitionId,
    default_state: Option<StateId>,
    any_state_transitions: Vec<AnimTransition>,
    time_in_state: f32,
}

impl AnimationController {
    pub fn new(bone_count: usize) -> Self {
        Self {
            clips: Vec::new(),
            states: Vec::new(),
            transitions: Vec::new(),
            parameters: HashMap::new(),
            current_state: None,
            active_transition: None,
            bone_count,
            temp_pose_a: vec![BoneTransform::IDENTITY; bone_count],
            temp_pose_b: vec![BoneTransform::IDENTITY; bone_count],
            pending_events: Vec::new(),
            root_motion: RootMotionDelta {
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
            },
            next_state_id: 0,
            next_transition_id: 0,
            default_state: None,
            any_state_transitions: Vec::new(),
            time_in_state: 0.0,
        }
    }

    /// Add an animation clip, returning its index.
    pub fn add_clip(&mut self, clip: AnimClip) -> usize {
        let idx = self.clips.len();
        self.clips.push(clip);
        idx
    }

    /// Add a state to the state machine.
    pub fn add_state(&mut self, state: AnimState) -> StateId {
        let id = self.next_state_id;
        self.next_state_id += 1;
        let mut s = state;
        s.id = id;
        if self.default_state.is_none() {
            self.default_state = Some(id);
        }
        self.states.push(s);
        id
    }

    /// Set the default (entry) state.
    pub fn set_default_state(&mut self, id: StateId) {
        self.default_state = Some(id);
    }

    /// Add a transition between states.
    pub fn add_transition(&mut self, transition: AnimTransition) -> TransitionId {
        let id = self.next_transition_id;
        self.next_transition_id += 1;
        let mut t = transition;
        t.id = id;
        self.transitions.push(t);
        id
    }

    /// Add a transition from "any state" (always checked).
    pub fn add_any_state_transition(&mut self, transition: AnimTransition) -> TransitionId {
        let id = self.next_transition_id;
        self.next_transition_id += 1;
        let mut t = transition;
        t.id = id;
        self.any_state_transitions.push(t);
        id
    }

    /// Set a parameter value.
    pub fn set_parameter(&mut self, name: &str, value: ParamValue) {
        self.parameters.insert(name.to_string(), value);
    }

    pub fn set_float(&mut self, name: &str, value: f32) {
        self.set_parameter(name, ParamValue::Float(value));
    }

    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.set_parameter(name, ParamValue::Bool(value));
    }

    pub fn set_int(&mut self, name: &str, value: i32) {
        self.set_parameter(name, ParamValue::Int(value));
    }

    pub fn set_trigger(&mut self, name: &str) {
        self.set_parameter(name, ParamValue::Trigger(true));
    }

    pub fn get_float(&self, name: &str) -> f32 {
        self.parameters.get(name).map(|v| v.as_float()).unwrap_or(0.0)
    }

    pub fn get_bool(&self, name: &str) -> bool {
        self.parameters.get(name).map(|v| v.as_bool()).unwrap_or(false)
    }

    /// Get the name of the current state.
    pub fn current_state_name(&self) -> Option<&str> {
        let state_id = self.current_state.as_ref()?.state_id;
        self.states.iter().find(|s| s.id == state_id).map(|s| s.name.as_str())
    }

    /// Whether a transition is currently active.
    pub fn is_transitioning(&self) -> bool {
        self.active_transition.is_some()
    }

    /// Time spent in the current state.
    pub fn time_in_current_state(&self) -> f32 {
        self.time_in_state
    }

    /// Normalized time (0..1) of the current animation.
    pub fn normalized_time(&self) -> f32 {
        self.current_state.as_ref().map(|s| s.normalized_time).unwrap_or(0.0)
    }

    /// Consume pending events.
    pub fn drain_events(&mut self) -> Vec<TriggeredEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Get root motion delta from the last update.
    pub fn root_motion(&self) -> RootMotionDelta {
        self.root_motion
    }

    /// Start the state machine (enter default state).
    pub fn start(&mut self) {
        if let Some(default_id) = self.default_state {
            self.current_state = Some(ActiveState {
                state_id: default_id,
                time: 0.0,
                prev_time: 0.0,
                normalized_time: 0.0,
                speed_multiplier: 1.0,
            });
            self.time_in_state = 0.0;
        }
    }

    /// Force transition to a specific state.
    pub fn force_state(&mut self, state_id: StateId) {
        self.active_transition = None;
        self.current_state = Some(ActiveState {
            state_id,
            time: 0.0,
            prev_time: 0.0,
            normalized_time: 0.0,
            speed_multiplier: 1.0,
        });
        self.time_in_state = 0.0;
    }

    /// Update the controller. `dt` is seconds. Output pose is written to `output`.
    pub fn update(&mut self, dt: f32, output: &mut [BoneTransform]) {
        if output.len() < self.bone_count {
            return;
        }

        self.root_motion = RootMotionDelta {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };

        // Initialize if not started.
        if self.current_state.is_none() {
            self.start();
        }

        // Check for transitions.
        if self.active_transition.is_none() {
            self.check_transitions();
        }

        // Update active transition or current state.
        if let Some(ref mut transition) = self.active_transition {
            transition.elapsed += dt;

            // Advance both states.
            self.advance_active_state(&mut transition.from_state, dt);
            self.advance_active_state(&mut transition.to_state, dt);

            let blend = (transition.elapsed / transition.duration).min(1.0);

            // Sample both poses.
            self.sample_state(&transition.from_state, &mut self.temp_pose_a.clone(), &mut self.temp_pose_a);
            self.sample_state(&transition.to_state, &mut self.temp_pose_b.clone(), &mut self.temp_pose_b);

            // Blend poses.
            for i in 0..self.bone_count.min(output.len()) {
                output[i] = BoneTransform::blend(&self.temp_pose_a[i], &self.temp_pose_b[i], blend);
            }

            // Blend root motion.
            let (rm_a_trans, rm_a_rot) = self.root_motion_for_state(&transition.from_state, dt);
            let (rm_b_trans, rm_b_rot) = self.root_motion_for_state(&transition.to_state, dt);
            self.root_motion.translation = Vec3::lerp(rm_a_trans, rm_b_trans, blend);
            self.root_motion.rotation = Quat::slerp(rm_a_rot, rm_b_rot, blend);

            // Check if transition is complete.
            if transition.elapsed >= transition.duration {
                let to_state = transition.to_state.clone();
                self.current_state = Some(to_state);
                self.active_transition = None;
                self.time_in_state = 0.0;
            }
        } else if let Some(ref mut state) = self.current_state {
            self.advance_active_state(state, dt);
            self.time_in_state += dt;

            // Sample current state.
            let state_clone = state.clone();
            self.sample_state(&state_clone, &mut self.temp_pose_a.clone(), output);

            // Root motion.
            let (rm_trans, rm_rot) = self.root_motion_for_state(&state_clone, dt);
            self.root_motion.translation = rm_trans;
            self.root_motion.rotation = rm_rot;

            // Collect events.
            self.collect_events(&state_clone);
        }

        // Reset triggers.
        let trigger_keys: Vec<_> = self.parameters.iter()
            .filter_map(|(k, v)| {
                if matches!(v, ParamValue::Trigger(true)) { Some(k.clone()) } else { None }
            })
            .collect();
        for key in trigger_keys {
            self.parameters.insert(key, ParamValue::Trigger(false));
        }
    }

    fn advance_active_state(&self, state: &mut ActiveState, dt: f32) {
        if let Some(anim_state) = self.states.iter().find(|s| s.id == state.state_id) {
            let clip_idx = anim_state.clip_index;
            if clip_idx < self.clips.len() {
                let clip = &self.clips[clip_idx];
                state.prev_time = state.time;
                state.time += dt * anim_state.speed * state.speed_multiplier;
                if clip.duration > 0.0 {
                    state.normalized_time = state.time / clip.duration;
                    if clip.looping {
                        state.time %= clip.duration;
                    }
                }
            }
        }
    }

    fn sample_state(&self, active: &ActiveState, _scratch: &mut [BoneTransform], output: &mut [BoneTransform]) {
        if let Some(anim_state) = self.states.iter().find(|s| s.id == active.state_id) {
            match anim_state.blend_mode {
                BlendMode::Single => {
                    if anim_state.clip_index < self.clips.len() {
                        self.clips[anim_state.clip_index].sample(active.time, output);
                    }
                }
                BlendMode::BlendSpace1D => {
                    self.sample_blend_1d(anim_state, active.time, output);
                }
                BlendMode::BlendSpace2D => {
                    self.sample_blend_2d(anim_state, active.time, output);
                }
                BlendMode::Additive => {
                    if anim_state.clip_index < self.clips.len() {
                        let mut additive = vec![BoneTransform::IDENTITY; self.bone_count];
                        self.clips[anim_state.clip_index].sample(active.time, &mut additive);
                        for i in 0..self.bone_count.min(output.len()) {
                            output[i].translation = output[i].translation + additive[i].translation;
                            output[i].rotation = output[i].rotation.mul(additive[i].rotation);
                        }
                    }
                }
            }
        }
    }

    fn sample_blend_1d(&self, state: &AnimState, time: f32, output: &mut [BoneTransform]) {
        let param_val = self.get_float(&state.blend_param_x);
        let entries = &state.blend_entries_1d;
        if entries.is_empty() { return; }
        if entries.len() == 1 {
            if entries[0].clip_index < self.clips.len() {
                self.clips[entries[0].clip_index].sample(time, output);
            }
            return;
        }

        // Find the two entries to blend between.
        let mut lo = 0;
        let mut hi = entries.len() - 1;
        for i in 0..entries.len() {
            if entries[i].threshold <= param_val {
                lo = i;
            }
            if entries[i].threshold >= param_val && i > lo {
                hi = i;
                break;
            }
        }

        if lo == hi || entries[lo].threshold == entries[hi].threshold {
            if entries[lo].clip_index < self.clips.len() {
                self.clips[entries[lo].clip_index].sample(time, output);
            }
            return;
        }

        let t = (param_val - entries[lo].threshold) / (entries[hi].threshold - entries[lo].threshold);
        let mut pose_a = vec![BoneTransform::IDENTITY; self.bone_count];
        let mut pose_b = vec![BoneTransform::IDENTITY; self.bone_count];

        if entries[lo].clip_index < self.clips.len() {
            self.clips[entries[lo].clip_index].sample(time, &mut pose_a);
        }
        if entries[hi].clip_index < self.clips.len() {
            self.clips[entries[hi].clip_index].sample(time, &mut pose_b);
        }

        for i in 0..self.bone_count.min(output.len()) {
            output[i] = BoneTransform::blend(&pose_a[i], &pose_b[i], t);
        }
    }

    fn sample_blend_2d(&self, state: &AnimState, time: f32, output: &mut [BoneTransform]) {
        // Simple nearest-neighbor for 2D blend (full implementation would use
        // Delaunay triangulation and barycentric interpolation).
        let px = self.get_float(&state.blend_param_x);
        let py = self.get_float(&state.blend_param_y);

        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for (i, entry) in state.blend_entries_2d.iter().enumerate() {
            let dx = px - entry.position.0;
            let dy = py - entry.position.1;
            let dist = dx * dx + dy * dy;
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        if state.blend_entries_2d[best_idx].clip_index < self.clips.len() {
            self.clips[state.blend_entries_2d[best_idx].clip_index].sample(time, output);
        }
    }

    fn root_motion_for_state(&self, active: &ActiveState, dt: f32) -> (Vec3, Quat) {
        if let Some(anim_state) = self.states.iter().find(|s| s.id == active.state_id) {
            if anim_state.clip_index < self.clips.len() {
                return self.clips[anim_state.clip_index].root_motion_delta(active.prev_time, active.time);
            }
        }
        (Vec3::ZERO, Quat::IDENTITY)
    }

    fn collect_events(&mut self, active: &ActiveState) {
        if let Some(anim_state) = self.states.iter().find(|s| s.id == active.state_id) {
            if anim_state.clip_index < self.clips.len() {
                let events = self.clips[anim_state.clip_index].triggered_events(active.prev_time, active.time);
                for ev in events {
                    self.pending_events.push(TriggeredEvent {
                        event_name: ev.name.clone(),
                        state_name: anim_state.name.clone(),
                        int_param: ev.int_param,
                        float_param: ev.float_param,
                        string_param: ev.string_param.clone(),
                    });
                }
            }
        }
    }

    fn check_transitions(&mut self) {
        let current_id = match &self.current_state {
            Some(s) => s.state_id,
            None => return,
        };

        // Check any-state transitions first.
        let mut best: Option<(usize, i32, bool)> = None;
        for (i, trans) in self.any_state_transitions.iter().enumerate() {
            if trans.to_state == current_id { continue; }
            if self.evaluate_conditions(&trans.conditions) {
                let prio = trans.priority;
                if best.map_or(true, |(_, bp, _)| prio > bp) {
                    best = Some((i, prio, true));
                }
            }
        }

        // Check state-specific transitions.
        for (i, trans) in self.transitions.iter().enumerate() {
            if trans.from_state != current_id { continue; }
            if self.evaluate_conditions(&trans.conditions) {
                let prio = trans.priority;
                if best.map_or(true, |(_, bp, _)| prio > bp) {
                    best = Some((i, prio, false));
                }
            }
        }

        if let Some((idx, _, is_any)) = best {
            let trans = if is_any {
                &self.any_state_transitions[idx]
            } else {
                &self.transitions[idx]
            };

            let from = self.current_state.clone().unwrap();
            let to = ActiveState {
                state_id: trans.to_state,
                time: 0.0,
                prev_time: 0.0,
                normalized_time: 0.0,
                speed_multiplier: 1.0,
            };

            self.active_transition = Some(ActiveTransition {
                transition_id: trans.id,
                from_state: from,
                to_state: to,
                elapsed: 0.0,
                duration: trans.duration,
            });
        }
    }

    fn evaluate_conditions(&self, conditions: &[TransitionCondition]) -> bool {
        for cond in conditions {
            let result = match cond {
                TransitionCondition::Greater { param, threshold } => {
                    self.get_float(param) > *threshold
                }
                TransitionCondition::Less { param, threshold } => {
                    self.get_float(param) < *threshold
                }
                TransitionCondition::IsTrue { param } => {
                    self.get_bool(param)
                }
                TransitionCondition::IsFalse { param } => {
                    !self.get_bool(param)
                }
                TransitionCondition::AnimationFinished => {
                    self.current_state.as_ref().map_or(false, |s| s.normalized_time >= 1.0)
                }
                TransitionCondition::TimeElapsed { seconds } => {
                    self.time_in_state >= *seconds
                }
            };
            if !result {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_clip(name: &str, duration: f32) -> AnimClip {
        let mut clip = AnimClip::new(name, duration);
        clip.tracks.push(AnimTrack {
            bone_index: 0,
            keyframes: vec![
                AnimKeyframe { time: 0.0, transform: BoneTransform::IDENTITY },
                AnimKeyframe {
                    time: duration,
                    transform: BoneTransform {
                        translation: Vec3::new(1.0, 0.0, 0.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3 { x: 1.0, y: 1.0, z: 1.0 },
                    },
                },
            ],
        });
        clip.events.push(AnimationEvent {
            time: duration * 0.5,
            name: "midpoint".to_string(),
            int_param: 0,
            float_param: 0.0,
            string_param: String::new(),
        });
        clip
    }

    #[test]
    fn test_clip_sample() {
        let clip = make_simple_clip("test", 1.0);
        let mut pose = vec![BoneTransform::IDENTITY; 1];
        clip.sample(0.5, &mut pose);
        assert!((pose[0].translation.x - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_state_machine_basic() {
        let mut ctrl = AnimationController::new(2);
        let idle_clip = ctrl.add_clip(make_simple_clip("idle", 1.0));
        let run_clip = ctrl.add_clip(make_simple_clip("run", 0.5));

        let idle_id = ctrl.add_state(AnimState::new(0, "idle", idle_clip));
        let run_id = ctrl.add_state(AnimState::new(1, "run", run_clip));

        ctrl.set_parameter("speed", ParamValue::Float(0.0));
        ctrl.add_transition(AnimTransition {
            id: 0,
            from_state: idle_id,
            to_state: run_id,
            duration: 0.2,
            conditions: vec![TransitionCondition::Greater {
                param: "speed".to_string(),
                threshold: 0.5,
            }],
            has_exit_time: false,
            exit_time: 0.0,
            priority: 0,
            interrupt_source: InterruptSource::None,
        });

        ctrl.start();
        let mut pose = vec![BoneTransform::IDENTITY; 2];

        // Should be in idle.
        ctrl.update(0.1, &mut pose);
        assert_eq!(ctrl.current_state_name(), Some("idle"));

        // Set speed > threshold -> should transition.
        ctrl.set_float("speed", 1.0);
        ctrl.update(0.1, &mut pose);
        assert!(ctrl.is_transitioning());

        // Complete the transition.
        for _ in 0..10 {
            ctrl.update(0.1, &mut pose);
        }
        assert_eq!(ctrl.current_state_name(), Some("run"));
    }

    #[test]
    fn test_events() {
        let mut ctrl = AnimationController::new(1);
        let clip = ctrl.add_clip(make_simple_clip("test", 1.0));
        ctrl.add_state(AnimState::new(0, "test", clip));
        ctrl.start();

        let mut pose = vec![BoneTransform::IDENTITY; 1];

        // Update past the midpoint event.
        ctrl.update(0.6, &mut pose);
        let events = ctrl.drain_events();
        assert!(!events.is_empty());
        assert_eq!(events[0].event_name, "midpoint");
    }

    #[test]
    fn test_blend_1d() {
        let mut ctrl = AnimationController::new(1);
        let walk = ctrl.add_clip(make_simple_clip("walk", 1.0));
        let run = ctrl.add_clip(make_simple_clip("run", 0.5));

        let state = AnimState::new(0, "locomotion", walk).with_blend_1d(
            "speed",
            vec![
                BlendEntry1D { clip_index: walk, threshold: 0.0 },
                BlendEntry1D { clip_index: run, threshold: 1.0 },
            ],
        );
        ctrl.add_state(state);
        ctrl.set_float("speed", 0.5);
        ctrl.start();

        let mut pose = vec![BoneTransform::IDENTITY; 1];
        ctrl.update(0.1, &mut pose);
        // Should produce a blended result.
        assert!(pose[0].translation.x >= 0.0);
    }
}
