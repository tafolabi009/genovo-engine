// engine/gameplay/src/game_state.rs
//
// Game state machine: MainMenu, Loading, Playing, Paused, Cutscene,
// GameOver, Victory states with transitions and data passing.

use std::collections::HashMap;

/// All possible game states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameStateId {
    Startup,
    MainMenu,
    Loading,
    Playing,
    Paused,
    Cutscene,
    GameOver,
    Victory,
    Settings,
    Credits,
    LevelSelect,
}

/// Data that can be passed between states.
#[derive(Debug, Clone)]
pub enum StateData {
    None,
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    LevelId(u32),
    Map(HashMap<String, StateData>),
}

impl Default for StateData {
    fn default() -> Self { Self::None }
}

/// A transition between two states.
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from: GameStateId,
    pub to: GameStateId,
    pub data: StateData,
    pub duration: f32,
    pub elapsed: f32,
    pub fade_out_duration: f32,
    pub fade_in_duration: f32,
}

impl StateTransition {
    pub fn new(from: GameStateId, to: GameStateId) -> Self {
        Self {
            from, to, data: StateData::None,
            duration: 0.5, elapsed: 0.0,
            fade_out_duration: 0.25, fade_in_duration: 0.25,
        }
    }

    pub fn with_data(mut self, data: StateData) -> Self { self.data = data; self }
    pub fn with_duration(mut self, d: f32) -> Self { self.duration = d; self }

    pub fn progress(&self) -> f32 { (self.elapsed / self.duration.max(0.001)).clamp(0.0, 1.0) }
    pub fn is_complete(&self) -> bool { self.elapsed >= self.duration }

    /// Current fade value: 1.0 = fully visible, 0.0 = black.
    pub fn fade_value(&self) -> f32 {
        let p = self.progress();
        let fade_out_end = self.fade_out_duration / self.duration.max(0.001);
        let fade_in_start = 1.0 - self.fade_in_duration / self.duration.max(0.001);
        if p < fade_out_end {
            1.0 - p / fade_out_end.max(0.001)
        } else if p > fade_in_start {
            (p - fade_in_start) / (1.0 - fade_in_start).max(0.001)
        } else {
            0.0
        }
    }
}

/// Events emitted by the state machine.
#[derive(Debug, Clone)]
pub enum GameStateEvent {
    StateEntered(GameStateId),
    StateExited(GameStateId),
    TransitionStarted { from: GameStateId, to: GameStateId },
    TransitionCompleted { from: GameStateId, to: GameStateId },
    PauseRequested,
    ResumeRequested,
    QuitRequested,
}

/// Allowed transition rules.
#[derive(Debug, Clone)]
pub struct TransitionRule {
    pub from: GameStateId,
    pub to: GameStateId,
    pub condition: Option<String>,
    pub auto_transition: bool,
    pub auto_delay: f32,
}

/// The game state machine.
pub struct GameStateMachine {
    current_state: GameStateId,
    previous_state: Option<GameStateId>,
    transition: Option<StateTransition>,
    state_data: HashMap<GameStateId, StateData>,
    state_timers: HashMap<GameStateId, f32>,
    rules: Vec<TransitionRule>,
    events: Vec<GameStateEvent>,
    pause_stack: Vec<GameStateId>,
    is_paused: bool,
    total_play_time: f32,
    state_enter_count: HashMap<GameStateId, u32>,
}

impl GameStateMachine {
    pub fn new() -> Self {
        Self {
            current_state: GameStateId::Startup,
            previous_state: None,
            transition: None,
            state_data: HashMap::new(),
            state_timers: HashMap::new(),
            rules: Vec::new(),
            events: Vec::new(),
            pause_stack: Vec::new(),
            is_paused: false,
            total_play_time: 0.0,
            state_enter_count: HashMap::new(),
        }
    }

    /// Add a transition rule.
    pub fn add_rule(&mut self, from: GameStateId, to: GameStateId) {
        self.rules.push(TransitionRule {
            from, to, condition: None, auto_transition: false, auto_delay: 0.0,
        });
    }

    /// Check if a transition is allowed.
    pub fn can_transition(&self, from: GameStateId, to: GameStateId) -> bool {
        self.rules.iter().any(|r| r.from == from && r.to == to)
    }

    /// Request a state transition.
    pub fn transition_to(&mut self, to: GameStateId) -> bool {
        self.transition_to_with_data(to, StateData::None)
    }

    pub fn transition_to_with_data(&mut self, to: GameStateId, data: StateData) -> bool {
        if self.transition.is_some() { return false; }
        if !self.can_transition(self.current_state, to) && !self.rules.is_empty() { return false; }

        let mut t = StateTransition::new(self.current_state, to);
        t.data = data;
        self.events.push(GameStateEvent::TransitionStarted { from: self.current_state, to });
        self.transition = Some(t);
        true
    }

    /// Update the state machine.
    pub fn update(&mut self, dt: f32) {
        // Track play time
        if self.current_state == GameStateId::Playing {
            self.total_play_time += dt;
        }

        // Update state timer
        *self.state_timers.entry(self.current_state).or_insert(0.0) += dt;

        // Process transition
        if let Some(ref mut transition) = self.transition {
            transition.elapsed += dt;
            if transition.is_complete() {
                let from = transition.from;
                let to = transition.to;
                let data = transition.data.clone();

                self.events.push(GameStateEvent::StateExited(from));
                self.previous_state = Some(from);
                self.current_state = to;
                self.state_data.insert(to, data);
                self.state_timers.insert(to, 0.0);
                *self.state_enter_count.entry(to).or_insert(0) += 1;
                self.events.push(GameStateEvent::StateEntered(to));
                self.events.push(GameStateEvent::TransitionCompleted { from, to });
                self.transition = None;
            }
        }

        // Check auto-transitions
        let current = self.current_state;
        let timer = self.state_timers.get(&current).copied().unwrap_or(0.0);
        for rule in &self.rules {
            if rule.from == current && rule.auto_transition && timer >= rule.auto_delay {
                if self.transition.is_none() {
                    self.transition_to(rule.to);
                    break;
                }
            }
        }
    }

    /// Pause the game (push current state, go to Paused).
    pub fn pause(&mut self) {
        if self.current_state == GameStateId::Playing {
            self.pause_stack.push(self.current_state);
            self.is_paused = true;
            self.events.push(GameStateEvent::PauseRequested);
            let _ = self.transition_to(GameStateId::Paused);
        }
    }

    /// Resume from pause.
    pub fn resume(&mut self) {
        if self.current_state == GameStateId::Paused {
            if let Some(prev) = self.pause_stack.pop() {
                self.is_paused = false;
                self.events.push(GameStateEvent::ResumeRequested);
                let _ = self.transition_to(prev);
            }
        }
    }

    /// Get current state.
    pub fn current(&self) -> GameStateId { self.current_state }
    pub fn previous(&self) -> Option<GameStateId> { self.previous_state }
    pub fn is_transitioning(&self) -> bool { self.transition.is_some() }
    pub fn is_paused(&self) -> bool { self.is_paused }
    pub fn play_time(&self) -> f32 { self.total_play_time }
    pub fn time_in_state(&self) -> f32 { self.state_timers.get(&self.current_state).copied().unwrap_or(0.0) }

    pub fn fade_value(&self) -> f32 {
        self.transition.as_ref().map(|t| t.fade_value()).unwrap_or(1.0)
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<GameStateEvent> {
        self.events.drain(..).collect()
    }

    /// Get state data.
    pub fn get_state_data(&self, state: GameStateId) -> Option<&StateData> {
        self.state_data.get(&state)
    }

    /// Build a default game state machine with standard transitions.
    pub fn with_default_rules() -> Self {
        let mut sm = Self::new();
        sm.add_rule(GameStateId::Startup, GameStateId::MainMenu);
        sm.add_rule(GameStateId::MainMenu, GameStateId::Loading);
        sm.add_rule(GameStateId::MainMenu, GameStateId::Settings);
        sm.add_rule(GameStateId::MainMenu, GameStateId::Credits);
        sm.add_rule(GameStateId::MainMenu, GameStateId::LevelSelect);
        sm.add_rule(GameStateId::Settings, GameStateId::MainMenu);
        sm.add_rule(GameStateId::Credits, GameStateId::MainMenu);
        sm.add_rule(GameStateId::LevelSelect, GameStateId::Loading);
        sm.add_rule(GameStateId::LevelSelect, GameStateId::MainMenu);
        sm.add_rule(GameStateId::Loading, GameStateId::Playing);
        sm.add_rule(GameStateId::Loading, GameStateId::Cutscene);
        sm.add_rule(GameStateId::Playing, GameStateId::Paused);
        sm.add_rule(GameStateId::Playing, GameStateId::Cutscene);
        sm.add_rule(GameStateId::Playing, GameStateId::GameOver);
        sm.add_rule(GameStateId::Playing, GameStateId::Victory);
        sm.add_rule(GameStateId::Paused, GameStateId::Playing);
        sm.add_rule(GameStateId::Paused, GameStateId::MainMenu);
        sm.add_rule(GameStateId::Paused, GameStateId::Settings);
        sm.add_rule(GameStateId::Cutscene, GameStateId::Playing);
        sm.add_rule(GameStateId::GameOver, GameStateId::MainMenu);
        sm.add_rule(GameStateId::GameOver, GameStateId::Loading);
        sm.add_rule(GameStateId::Victory, GameStateId::MainMenu);
        sm.add_rule(GameStateId::Victory, GameStateId::Loading);
        sm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let sm = GameStateMachine::new();
        assert_eq!(sm.current(), GameStateId::Startup);
    }

    #[test]
    fn test_transition() {
        let mut sm = GameStateMachine::with_default_rules();
        assert!(sm.transition_to(GameStateId::MainMenu));
        assert!(sm.is_transitioning());
        for _ in 0..100 { sm.update(0.01); }
        assert_eq!(sm.current(), GameStateId::MainMenu);
    }

    #[test]
    fn test_pause_resume() {
        let mut sm = GameStateMachine::with_default_rules();
        sm.current_state = GameStateId::Playing; // force for test
        sm.pause();
        for _ in 0..100 { sm.update(0.01); }
        assert_eq!(sm.current(), GameStateId::Paused);
        sm.resume();
        for _ in 0..100 { sm.update(0.01); }
        assert_eq!(sm.current(), GameStateId::Playing);
    }

    #[test]
    fn test_state_data() {
        let mut sm = GameStateMachine::with_default_rules();
        sm.transition_to_with_data(GameStateId::MainMenu, StateData::String("test".into()));
        for _ in 0..100 { sm.update(0.01); }
        assert!(sm.get_state_data(GameStateId::MainMenu).is_some());
    }

    #[test]
    fn test_fade_value() {
        let mut sm = GameStateMachine::with_default_rules();
        sm.transition_to(GameStateId::MainMenu);
        let fade = sm.fade_value();
        assert!(fade >= 0.0 && fade <= 1.0);
    }
}
