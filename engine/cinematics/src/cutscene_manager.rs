//! Cutscene management: cutscene loading, state machine, skip handling,
//! cutscene events, and cutscene blending with gameplay.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CutsceneId(pub u64);
impl CutsceneId { pub fn from_name(n: &str) -> Self { use std::hash::{Hash,Hasher}; let mut h = std::collections::hash_map::DefaultHasher::new(); n.hash(&mut h); Self(h.finish()) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutsceneState { Idle, Loading, Playing, Paused, Skipping, Blending, Finished, Error }

#[derive(Debug, Clone)]
pub struct CutsceneEventDef { pub time: f32, pub event_type: String, pub data: HashMap<String, String> }
impl CutsceneEventDef { pub fn new(time: f32, et: impl Into<String>) -> Self { Self { time, event_type: et.into(), data: HashMap::new() } } }

#[derive(Debug, Clone)]
pub struct CutsceneDef {
    pub id: CutsceneId, pub name: String, pub asset_path: String, pub duration: f32,
    pub skippable: bool, pub skip_fade_duration: f32, pub blend_in_duration: f32,
    pub blend_out_duration: f32, pub events: Vec<CutsceneEventDef>,
    pub camera_sequence: Option<String>, pub audio_track: Option<String>,
    pub subtitle_track: Option<String>, pub priority: i32,
}
impl CutsceneDef {
    pub fn new(id: CutsceneId, name: impl Into<String>, duration: f32) -> Self {
        Self { id, name: name.into(), asset_path: String::new(), duration, skippable: true, skip_fade_duration: 0.5, blend_in_duration: 0.3, blend_out_duration: 0.3, events: Vec::new(), camera_sequence: None, audio_track: None, subtitle_track: None, priority: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct CutsceneInstance {
    pub definition: CutsceneDef, pub state: CutsceneState, pub current_time: f32,
    pub playback_speed: f32, pub pending_events: Vec<CutsceneEventDef>,
    pub triggered_events: Vec<CutsceneEventDef>, pub blend_alpha: f32,
    pub skip_requested: bool, pub skip_timer: f32,
}
impl CutsceneInstance {
    pub fn new(def: CutsceneDef) -> Self {
        Self { definition: def, state: CutsceneState::Idle, current_time: 0.0, playback_speed: 1.0, pending_events: Vec::new(), triggered_events: Vec::new(), blend_alpha: 0.0, skip_requested: false, skip_timer: 0.0 }
    }
    pub fn progress(&self) -> f32 { if self.definition.duration > 0.0 { self.current_time / self.definition.duration } else { 0.0 } }
    pub fn is_finished(&self) -> bool { self.state == CutsceneState::Finished }
}

#[derive(Debug, Clone)]
pub enum CutsceneManagerEvent {
    CutsceneStarted(CutsceneId), CutsceneFinished(CutsceneId), CutsceneSkipped(CutsceneId),
    CutscenePaused(CutsceneId), CutsceneResumed(CutsceneId),
    EventTriggered(CutsceneId, String), BlendInComplete(CutsceneId), BlendOutComplete(CutsceneId),
}

pub struct CutsceneManager {
    pub definitions: HashMap<CutsceneId, CutsceneDef>,
    pub active: Option<CutsceneInstance>,
    pub queue: Vec<CutsceneId>,
    pub events: Vec<CutsceneManagerEvent>,
    pub gameplay_camera_stored: bool,
    pub stored_camera: Option<([f32; 3], [f32; 3])>,
    pub input_blocked: bool,
    pub auto_play_queue: bool,
}

impl CutsceneManager {
    pub fn new() -> Self { Self { definitions: HashMap::new(), active: None, queue: Vec::new(), events: Vec::new(), gameplay_camera_stored: false, stored_camera: None, input_blocked: false, auto_play_queue: true } }

    pub fn register(&mut self, def: CutsceneDef) { self.definitions.insert(def.id, def); }

    pub fn play(&mut self, id: CutsceneId) -> bool {
        if let Some(def) = self.definitions.get(&id) {
            let mut inst = CutsceneInstance::new(def.clone());
            inst.state = CutsceneState::Blending;
            inst.pending_events = def.events.clone();
            self.active = Some(inst);
            self.input_blocked = true;
            self.events.push(CutsceneManagerEvent::CutsceneStarted(id));
            true
        } else { false }
    }

    pub fn queue_cutscene(&mut self, id: CutsceneId) { self.queue.push(id); }

    pub fn skip(&mut self) {
        if let Some(ref mut inst) = self.active {
            if inst.definition.skippable && !inst.skip_requested {
                inst.skip_requested = true;
                inst.state = CutsceneState::Skipping;
                self.events.push(CutsceneManagerEvent::CutsceneSkipped(inst.definition.id));
            }
        }
    }

    pub fn pause(&mut self) {
        if let Some(ref mut inst) = self.active {
            if inst.state == CutsceneState::Playing { inst.state = CutsceneState::Paused; self.events.push(CutsceneManagerEvent::CutscenePaused(inst.definition.id)); }
        }
    }

    pub fn resume(&mut self) {
        if let Some(ref mut inst) = self.active {
            if inst.state == CutsceneState::Paused { inst.state = CutsceneState::Playing; self.events.push(CutsceneManagerEvent::CutsceneResumed(inst.definition.id)); }
        }
    }

    pub fn update(&mut self, dt: f32) {
        let mut finished = false;
        if let Some(ref mut inst) = self.active {
            match inst.state {
                CutsceneState::Blending => {
                    inst.blend_alpha += dt / inst.definition.blend_in_duration.max(0.01);
                    if inst.blend_alpha >= 1.0 { inst.blend_alpha = 1.0; inst.state = CutsceneState::Playing; self.events.push(CutsceneManagerEvent::BlendInComplete(inst.definition.id)); }
                }
                CutsceneState::Playing => {
                    inst.current_time += dt * inst.playback_speed;
                    let mut triggered = Vec::new();
                    inst.pending_events.retain(|e| { if e.time <= inst.current_time { triggered.push(e.clone()); false } else { true } });
                    for e in &triggered { self.events.push(CutsceneManagerEvent::EventTriggered(inst.definition.id, e.event_type.clone())); }
                    inst.triggered_events.extend(triggered);
                    if inst.current_time >= inst.definition.duration { inst.state = CutsceneState::Blending; inst.blend_alpha = 1.0; }
                }
                CutsceneState::Skipping => {
                    inst.skip_timer += dt;
                    inst.blend_alpha -= dt / inst.definition.skip_fade_duration.max(0.01);
                    if inst.blend_alpha <= 0.0 { inst.state = CutsceneState::Finished; finished = true; }
                }
                CutsceneState::Finished => { finished = true; }
                _ => {}
            }
            if inst.current_time >= inst.definition.duration && inst.state == CutsceneState::Blending && inst.blend_alpha <= 0.0 {
                inst.state = CutsceneState::Finished; finished = true;
            }
        }
        if finished {
            if let Some(inst) = self.active.take() {
                self.events.push(CutsceneManagerEvent::CutsceneFinished(inst.definition.id));
                self.input_blocked = false;
            }
            if self.auto_play_queue && !self.queue.is_empty() {
                let next = self.queue.remove(0);
                self.play(next);
            }
        }
    }

    pub fn is_playing(&self) -> bool { self.active.is_some() }
    pub fn current_progress(&self) -> f32 { self.active.as_ref().map(|i| i.progress()).unwrap_or(0.0) }
    pub fn drain_events(&mut self) -> Vec<CutsceneManagerEvent> { std::mem::take(&mut self.events) }
}

impl Default for CutsceneManager { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cutscene_playback() {
        let mut mgr = CutsceneManager::new();
        let id = CutsceneId::from_name("intro");
        mgr.register(CutsceneDef::new(id, "Intro", 5.0));
        assert!(mgr.play(id));
        assert!(mgr.is_playing());
        for _ in 0..100 { mgr.update(0.1); }
    }
    #[test]
    fn cutscene_skip() {
        let mut mgr = CutsceneManager::new();
        let id = CutsceneId::from_name("skip_test");
        let mut def = CutsceneDef::new(id, "Skip Test", 10.0);
        def.blend_in_duration = 0.01;
        mgr.register(def);
        mgr.play(id);
        mgr.update(0.1);
        mgr.skip();
        for _ in 0..50 { mgr.update(0.1); }
    }
}
