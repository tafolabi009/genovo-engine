// engine/ai/src/dialogue_manager.rs
//
// Dialogue management for the Genovo engine.
//
// Manages the lifecycle of in-game dialogues:
//
// - **Dialogue queue** -- Queue dialogues for sequential playback.
// - **Simultaneous conversations** -- Multiple dialogues at once.
// - **Interruption priority** -- Higher-priority dialogues interrupt lower ones.
// - **Dialogue cooldown** -- Prevent the same dialogue from replaying too soon.
// - **Subtitled/voiced/bark modes** -- Different presentation styles.
// - **Dialogue events** -- Trigger gameplay actions from dialogue nodes.

use std::collections::{HashMap, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DialogueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConversationId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeakerId(pub u32);

impl fmt::Display for DialogueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dialogue({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Dialogue mode
// ---------------------------------------------------------------------------

/// How the dialogue is presented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DialogueMode {
    /// Full dialogue with subtitle UI and camera focus.
    Cinematic,
    /// Subtitles only (no camera change).
    Subtitled,
    /// Voiced with subtitles.
    VoicedSubtitled,
    /// Short bark (spoken text above NPC head).
    Bark,
    /// Radio/comm channel dialogue.
    Radio,
    /// Internal thought (italicized, no speaker visible).
    Thought,
}

impl Default for DialogueMode {
    fn default() -> Self {
        Self::Subtitled
    }
}

// ---------------------------------------------------------------------------
// Dialogue line
// ---------------------------------------------------------------------------

/// A single line of dialogue.
#[derive(Debug, Clone)]
pub struct DialogueLine {
    /// Speaker.
    pub speaker: SpeakerId,
    /// Speaker display name.
    pub speaker_name: String,
    /// Text content.
    pub text: String,
    /// Audio clip identifier (empty = no audio).
    pub audio_clip: String,
    /// Duration to display (seconds, 0 = auto based on text length).
    pub duration: f32,
    /// Animation to play on the speaker.
    pub animation: String,
    /// Emotion/mood of the speaker.
    pub emotion: String,
    /// Camera angle suggestion.
    pub camera: DialogueCamera,
    /// Events to fire when this line starts.
    pub events: Vec<DialogueGameEvent>,
}

impl DialogueLine {
    pub fn new(speaker: SpeakerId, name: &str, text: &str) -> Self {
        Self {
            speaker,
            speaker_name: name.to_string(),
            text: text.to_string(),
            audio_clip: String::new(),
            duration: 0.0,
            animation: String::new(),
            emotion: String::new(),
            camera: DialogueCamera::Default,
            events: Vec::new(),
        }
    }

    /// Compute display duration based on text length.
    pub fn auto_duration(&self) -> f32 {
        if self.duration > 0.0 {
            self.duration
        } else {
            let word_count = self.text.split_whitespace().count();
            (word_count as f32 * 0.4).max(2.0).min(10.0)
        }
    }
}

/// Camera behavior during a dialogue line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogueCamera {
    Default,
    CloseUp,
    OverShoulder,
    Wide,
    Reaction,
    Custom(u32),
}

/// A gameplay event triggered by dialogue.
#[derive(Debug, Clone)]
pub struct DialogueGameEvent {
    pub event_name: String,
    pub parameters: HashMap<String, String>,
    pub delay: f32,
}

// ---------------------------------------------------------------------------
// Dialogue definition
// ---------------------------------------------------------------------------

/// A complete dialogue sequence.
#[derive(Debug, Clone)]
pub struct DialogueDefinition {
    pub id: DialogueId,
    pub name: String,
    pub mode: DialogueMode,
    pub priority: i32,
    pub lines: Vec<DialogueLine>,
    pub cooldown: f32,
    pub can_be_interrupted: bool,
    pub skippable: bool,
    pub conditions: Vec<String>,
    pub on_complete_events: Vec<DialogueGameEvent>,
}

impl DialogueDefinition {
    pub fn new(id: DialogueId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            mode: DialogueMode::Subtitled,
            priority: 0,
            lines: Vec::new(),
            cooldown: 0.0,
            can_be_interrupted: true,
            skippable: true,
            conditions: Vec::new(),
            on_complete_events: Vec::new(),
        }
    }

    pub fn add_line(&mut self, line: DialogueLine) {
        self.lines.push(line);
    }

    pub fn total_duration(&self) -> f32 {
        self.lines.iter().map(|l| l.auto_duration()).sum()
    }
}

// ---------------------------------------------------------------------------
// Active conversation
// ---------------------------------------------------------------------------

/// State of an active conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationState {
    Playing,
    Paused,
    WaitingForInput,
    Finished,
    Interrupted,
}

/// An active conversation instance.
#[derive(Debug, Clone)]
pub struct ActiveConversation {
    pub id: ConversationId,
    pub dialogue_id: DialogueId,
    pub state: ConversationState,
    pub current_line: usize,
    pub line_timer: f32,
    pub mode: DialogueMode,
    pub priority: i32,
    pub participants: Vec<SpeakerId>,
    pub elapsed: f32,
}

impl ActiveConversation {
    pub fn new(id: ConversationId, dialogue: &DialogueDefinition) -> Self {
        let participants: Vec<SpeakerId> = dialogue
            .lines
            .iter()
            .map(|l| l.speaker)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        Self {
            id,
            dialogue_id: dialogue.id,
            state: ConversationState::Playing,
            current_line: 0,
            line_timer: 0.0,
            mode: dialogue.mode,
            priority: dialogue.priority,
            participants,
            elapsed: 0.0,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self.state, ConversationState::Playing | ConversationState::Paused | ConversationState::WaitingForInput)
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.state, ConversationState::Finished | ConversationState::Interrupted)
    }
}

// ---------------------------------------------------------------------------
// Dialogue events
// ---------------------------------------------------------------------------

/// Events emitted by the dialogue manager.
#[derive(Debug, Clone)]
pub enum DialogueMgrEvent {
    ConversationStarted { conversation: ConversationId, dialogue: DialogueId },
    LineStarted { conversation: ConversationId, line_index: usize, speaker: SpeakerId, text: String },
    LineEnded { conversation: ConversationId, line_index: usize },
    ConversationEnded { conversation: ConversationId, dialogue: DialogueId },
    ConversationInterrupted { conversation: ConversationId, by: ConversationId },
    GameEvent { conversation: ConversationId, event: DialogueGameEvent },
    CooldownStarted { dialogue: DialogueId, duration: f32 },
    BarkDisplayed { speaker: SpeakerId, text: String },
}

// ---------------------------------------------------------------------------
// Dialogue manager
// ---------------------------------------------------------------------------

/// Manages dialogue playback, queueing, and lifecycle.
pub struct DialogueManager {
    /// Dialogue definitions.
    definitions: HashMap<DialogueId, DialogueDefinition>,
    /// Active conversations.
    active: Vec<ActiveConversation>,
    /// Queued dialogues waiting to play.
    queue: VecDeque<DialogueId>,
    /// Cooldowns (dialogue_id -> remaining cooldown).
    cooldowns: HashMap<DialogueId, f32>,
    /// Next conversation ID.
    next_conv_id: u32,
    /// Maximum simultaneous conversations.
    max_simultaneous: usize,
    /// Events this frame.
    events: Vec<DialogueMgrEvent>,
    /// Bark display list.
    active_barks: Vec<ActiveBark>,
    /// Game time.
    time: f64,
}

/// An active bark (short text above an NPC's head).
#[derive(Debug, Clone)]
pub struct ActiveBark {
    pub speaker: SpeakerId,
    pub text: String,
    pub remaining: f32,
    pub position: [f32; 3],
}

impl DialogueManager {
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            active: Vec::new(),
            queue: VecDeque::new(),
            cooldowns: HashMap::new(),
            next_conv_id: 0,
            max_simultaneous: 3,
            events: Vec::new(),
            active_barks: Vec::new(),
            time: 0.0,
        }
    }

    /// Register a dialogue definition.
    pub fn register(&mut self, definition: DialogueDefinition) {
        self.definitions.insert(definition.id, definition);
    }

    /// Get a definition.
    pub fn definition(&self, id: DialogueId) -> Option<&DialogueDefinition> {
        self.definitions.get(&id)
    }

    /// Start a dialogue immediately.
    pub fn start(&mut self, dialogue_id: DialogueId) -> Option<ConversationId> {
        let def = self.definitions.get(&dialogue_id)?.clone();

        // Check cooldown.
        if let Some(&cd) = self.cooldowns.get(&dialogue_id) {
            if cd > 0.0 {
                return None;
            }
        }

        // Check if we need to interrupt lower-priority conversations.
        if self.active.len() >= self.max_simultaneous {
            let lowest = self.active.iter().min_by_key(|c| c.priority);
            if let Some(low) = lowest {
                if low.priority < def.priority {
                    let low_id = low.id;
                    self.interrupt(low_id);
                } else {
                    // Queue instead.
                    self.queue.push_back(dialogue_id);
                    return None;
                }
            }
        }

        let conv_id = ConversationId(self.next_conv_id);
        self.next_conv_id += 1;

        let conv = ActiveConversation::new(conv_id, &def);
        self.active.push(conv);

        self.events.push(DialogueMgrEvent::ConversationStarted {
            conversation: conv_id,
            dialogue: dialogue_id,
        });

        // Fire first line events.
        if !def.lines.is_empty() {
            let line = &def.lines[0];
            self.events.push(DialogueMgrEvent::LineStarted {
                conversation: conv_id,
                line_index: 0,
                speaker: line.speaker,
                text: line.text.clone(),
            });
            for event in &line.events {
                self.events.push(DialogueMgrEvent::GameEvent {
                    conversation: conv_id,
                    event: event.clone(),
                });
            }
        }

        Some(conv_id)
    }

    /// Queue a dialogue for later playback.
    pub fn enqueue(&mut self, dialogue_id: DialogueId) {
        self.queue.push_back(dialogue_id);
    }

    /// Interrupt a conversation.
    pub fn interrupt(&mut self, conv_id: ConversationId) {
        if let Some(conv) = self.active.iter_mut().find(|c| c.id == conv_id) {
            conv.state = ConversationState::Interrupted;
            self.events.push(DialogueMgrEvent::ConversationEnded {
                conversation: conv_id,
                dialogue: conv.dialogue_id,
            });
        }
    }

    /// Skip the current line of a conversation.
    pub fn skip_line(&mut self, conv_id: ConversationId) {
        if let Some(conv) = self.active.iter_mut().find(|c| c.id == conv_id) {
            if let Some(def) = self.definitions.get(&conv.dialogue_id) {
                if !def.skippable {
                    return;
                }
            }
            conv.line_timer = 0.0;
            // Will advance on next update.
        }
    }

    /// Display a bark (short text over NPC head).
    pub fn bark(&mut self, speaker: SpeakerId, text: &str, position: [f32; 3], duration: f32) {
        self.active_barks.push(ActiveBark {
            speaker,
            text: text.to_string(),
            remaining: duration,
            position,
        });
        self.events.push(DialogueMgrEvent::BarkDisplayed {
            speaker,
            text: text.to_string(),
        });
    }

    /// Update all conversations.
    pub fn update(&mut self, dt: f32) {
        self.time += dt as f64;

        // Update cooldowns.
        for cd in self.cooldowns.values_mut() {
            *cd = (*cd - dt).max(0.0);
        }

        // Update barks.
        self.active_barks.retain_mut(|b| {
            b.remaining -= dt;
            b.remaining > 0.0
        });

        // Update active conversations.
        let mut finished = Vec::new();

        for conv in &mut self.active {
            if !conv.is_active() {
                continue;
            }

            if conv.state == ConversationState::Paused {
                continue;
            }

            conv.elapsed += dt;

            let def = match self.definitions.get(&conv.dialogue_id) {
                Some(d) => d,
                None => {
                    conv.state = ConversationState::Finished;
                    continue;
                }
            };

            if conv.current_line >= def.lines.len() {
                conv.state = ConversationState::Finished;
                finished.push((conv.id, conv.dialogue_id));
                continue;
            }

            let line = &def.lines[conv.current_line];
            let line_duration = line.auto_duration();

            conv.line_timer += dt;
            if conv.line_timer >= line_duration {
                // Line ended.
                self.events.push(DialogueMgrEvent::LineEnded {
                    conversation: conv.id,
                    line_index: conv.current_line,
                });

                conv.current_line += 1;
                conv.line_timer = 0.0;

                // Start next line or finish.
                if conv.current_line < def.lines.len() {
                    let next_line = &def.lines[conv.current_line];
                    self.events.push(DialogueMgrEvent::LineStarted {
                        conversation: conv.id,
                        line_index: conv.current_line,
                        speaker: next_line.speaker,
                        text: next_line.text.clone(),
                    });
                    for event in &next_line.events {
                        self.events.push(DialogueMgrEvent::GameEvent {
                            conversation: conv.id,
                            event: event.clone(),
                        });
                    }
                } else {
                    conv.state = ConversationState::Finished;
                    finished.push((conv.id, conv.dialogue_id));
                }
            }
        }

        // Handle finished conversations.
        for (conv_id, dialogue_id) in &finished {
            self.events.push(DialogueMgrEvent::ConversationEnded {
                conversation: *conv_id,
                dialogue: *dialogue_id,
            });

            // Set cooldown.
            if let Some(def) = self.definitions.get(dialogue_id) {
                if def.cooldown > 0.0 {
                    self.cooldowns.insert(*dialogue_id, def.cooldown);
                    self.events.push(DialogueMgrEvent::CooldownStarted {
                        dialogue: *dialogue_id,
                        duration: def.cooldown,
                    });
                }
                // Fire completion events.
                for event in &def.on_complete_events {
                    self.events.push(DialogueMgrEvent::GameEvent {
                        conversation: *conv_id,
                        event: event.clone(),
                    });
                }
            }
        }

        // Remove finished conversations.
        self.active.retain(|c| c.is_active());

        // Process queue.
        while self.active.len() < self.max_simultaneous {
            if let Some(next) = self.queue.pop_front() {
                self.start(next);
            } else {
                break;
            }
        }
    }

    /// Get active conversations.
    pub fn active_conversations(&self) -> &[ActiveConversation] {
        &self.active
    }

    /// Get active barks.
    pub fn active_barks(&self) -> &[ActiveBark] {
        &self.active_barks
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<DialogueMgrEvent> {
        std::mem::take(&mut self.events)
    }

    /// Check if a dialogue is on cooldown.
    pub fn is_on_cooldown(&self, id: DialogueId) -> bool {
        self.cooldowns.get(&id).map(|&cd| cd > 0.0).unwrap_or(false)
    }

    /// Set the maximum simultaneous conversations.
    pub fn set_max_simultaneous(&mut self, max: usize) {
        self.max_simultaneous = max.max(1);
    }

    /// Get the number of active conversations.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Get the queue length.
    pub fn queue_length(&self) -> usize {
        self.queue.len()
    }
}

impl Default for DialogueManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dialogue(id: u32) -> DialogueDefinition {
        let mut def = DialogueDefinition::new(DialogueId(id), &format!("Dialogue {}", id));
        def.lines.push(DialogueLine::new(SpeakerId(0), "NPC", "Hello there!"));
        def.lines.push(DialogueLine::new(SpeakerId(0), "NPC", "How are you?"));
        def
    }

    #[test]
    fn test_start_dialogue() {
        let mut mgr = DialogueManager::new();
        mgr.register(make_dialogue(0));
        let conv = mgr.start(DialogueId(0));
        assert!(conv.is_some());
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_dialogue_progression() {
        let mut mgr = DialogueManager::new();
        let mut def = make_dialogue(0);
        for line in &mut def.lines {
            line.duration = 1.0;
        }
        mgr.register(def);
        mgr.start(DialogueId(0));

        mgr.update(1.5); // Should advance past first line.
        let events = mgr.drain_events();
        assert!(events.iter().any(|e| matches!(e, DialogueMgrEvent::LineEnded { .. })));
    }

    #[test]
    fn test_cooldown() {
        let mut mgr = DialogueManager::new();
        let mut def = make_dialogue(0);
        def.cooldown = 5.0;
        for line in &mut def.lines {
            line.duration = 0.1;
        }
        mgr.register(def);

        mgr.start(DialogueId(0));
        mgr.update(1.0); // Should finish.

        // Try to start again during cooldown.
        assert!(mgr.start(DialogueId(0)).is_none());
        assert!(mgr.is_on_cooldown(DialogueId(0)));
    }

    #[test]
    fn test_bark() {
        let mut mgr = DialogueManager::new();
        mgr.bark(SpeakerId(1), "Watch out!", [0.0; 3], 3.0);
        assert_eq!(mgr.active_barks().len(), 1);
        mgr.update(4.0);
        assert_eq!(mgr.active_barks().len(), 0);
    }

    #[test]
    fn test_queue() {
        let mut mgr = DialogueManager::new();
        mgr.set_max_simultaneous(1);
        mgr.register(make_dialogue(0));
        mgr.register(make_dialogue(1));

        mgr.start(DialogueId(0));
        mgr.enqueue(DialogueId(1));

        assert_eq!(mgr.active_count(), 1);
        assert_eq!(mgr.queue_length(), 1);
    }
}
