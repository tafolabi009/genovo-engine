// engine/ai/src/conversation.rs
//
// AI conversation system for multi-participant dialogue with turn-taking,
// interruption, topic switching, memory of previous conversations,
// relationship changes, and skill checks (barter/persuade/intimidate).

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParticipantId(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConversationId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TopicId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationState { NotStarted, Active, Paused, Ended, Interrupted }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState { Speaking, Listening, Thinking, Interrupted, WaitingForResponse }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillCheckType { Persuasion, Intimidation, Barter, Deception, Charm, Knowledge }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillCheckResult { CriticalSuccess, Success, Failure, CriticalFailure }

#[derive(Debug, Clone)]
pub struct SkillCheck {
    pub skill_type: SkillCheckType,
    pub difficulty: u32,
    pub player_skill: u32,
    pub modifiers: Vec<(String, i32)>,
    pub result: Option<SkillCheckResult>,
}

impl SkillCheck {
    pub fn new(skill_type: SkillCheckType, difficulty: u32) -> Self {
        Self { skill_type, difficulty, player_skill: 0, modifiers: Vec::new(), result: None }
    }
    pub fn add_modifier(&mut self, name: &str, value: i32) { self.modifiers.push((name.to_string(), value)); }
    pub fn total_skill(&self) -> i32 {
        self.player_skill as i32 + self.modifiers.iter().map(|(_, v)| v).sum::<i32>()
    }
    pub fn resolve(&mut self, roll: u32) -> SkillCheckResult {
        let total = self.total_skill() + roll as i32;
        let diff = self.difficulty as i32;
        let result = if total >= diff + 10 { SkillCheckResult::CriticalSuccess }
        else if total >= diff { SkillCheckResult::Success }
        else if total >= diff - 10 { SkillCheckResult::Failure }
        else { SkillCheckResult::CriticalFailure };
        self.result = Some(result);
        result
    }
}

#[derive(Debug, Clone)]
pub struct ConversationTopic {
    pub id: TopicId,
    pub name: String,
    pub description: String,
    pub exhausted: bool,
    pub requires_knowledge: bool,
    pub knowledge_key: String,
    pub disposition_requirement: f32,
    pub available: bool,
    pub responses: Vec<DialogueResponse>,
}

impl ConversationTopic {
    pub fn new(id: TopicId, name: &str) -> Self {
        Self { id, name: name.to_string(), description: String::new(), exhausted: false, requires_knowledge: false, knowledge_key: String::new(), disposition_requirement: 0.0, available: true, responses: Vec::new() }
    }
    pub fn add_response(&mut self, response: DialogueResponse) { self.responses.push(response); }
}

#[derive(Debug, Clone)]
pub struct DialogueResponse {
    pub text: String,
    pub speaker: ParticipantId,
    pub skill_check: Option<SkillCheck>,
    pub disposition_change: f32,
    pub next_topic: Option<TopicId>,
    pub end_conversation: bool,
    pub conditions: Vec<(String, bool)>,
}

impl DialogueResponse {
    pub fn new(text: &str, speaker: ParticipantId) -> Self {
        Self { text: text.to_string(), speaker, skill_check: None, disposition_change: 0.0, next_topic: None, end_conversation: false, conditions: Vec::new() }
    }
    pub fn with_disposition_change(mut self, change: f32) -> Self { self.disposition_change = change; self }
    pub fn with_next_topic(mut self, topic: TopicId) -> Self { self.next_topic = Some(topic); self }
    pub fn ending(mut self) -> Self { self.end_conversation = true; self }
    pub fn with_skill_check(mut self, check: SkillCheck) -> Self { self.skill_check = Some(check); self }
}

#[derive(Debug, Clone)]
pub struct ConversationParticipant {
    pub id: ParticipantId,
    pub name: String,
    pub turn_state: TurnState,
    pub disposition: f32,
    pub patience: f32,
    pub max_patience: f32,
    pub speaking_speed: f32,
    pub interruption_threshold: f32,
    pub skills: HashMap<SkillCheckType, u32>,
}

impl ConversationParticipant {
    pub fn new(id: ParticipantId, name: &str) -> Self {
        Self { id, name: name.to_string(), turn_state: TurnState::Listening, disposition: 50.0, patience: 100.0, max_patience: 100.0, speaking_speed: 1.0, interruption_threshold: 20.0, skills: HashMap::new() }
    }
    pub fn set_skill(&mut self, skill: SkillCheckType, level: u32) { self.skills.insert(skill, level); }
    pub fn get_skill(&self, skill: SkillCheckType) -> u32 { self.skills.get(&skill).copied().unwrap_or(0) }
    pub fn modify_disposition(&mut self, amount: f32) { self.disposition = (self.disposition + amount).clamp(0.0, 100.0); }
    pub fn lose_patience(&mut self, amount: f32) { self.patience = (self.patience - amount).max(0.0); }
    pub fn is_patient(&self) -> bool { self.patience > 0.0 }
}

#[derive(Debug, Clone)]
pub struct ConversationMemory {
    pub conversation_id: ConversationId,
    pub participants: Vec<ParticipantId>,
    pub topics_discussed: Vec<TopicId>,
    pub disposition_changes: HashMap<ParticipantId, f32>,
    pub skill_check_results: Vec<(SkillCheckType, SkillCheckResult)>,
    pub timestamp: f64,
    pub duration: f32,
    pub was_interrupted: bool,
    pub outcome: String,
}

impl ConversationMemory {
    pub fn new(id: ConversationId, participants: Vec<ParticipantId>, timestamp: f64) -> Self {
        Self { conversation_id: id, participants, topics_discussed: Vec::new(), disposition_changes: HashMap::new(), skill_check_results: Vec::new(), timestamp, duration: 0.0, was_interrupted: false, outcome: String::new() }
    }
}

#[derive(Debug)]
pub struct Conversation {
    pub id: ConversationId,
    pub state: ConversationState,
    pub participants: Vec<ConversationParticipant>,
    pub topics: HashMap<TopicId, ConversationTopic>,
    pub current_topic: Option<TopicId>,
    pub current_speaker: Option<ParticipantId>,
    pub turn_timer: f32,
    pub turn_duration: f32,
    pub elapsed: f32,
    pub history: Vec<(ParticipantId, String)>,
    pub pending_interruption: Option<ParticipantId>,
    pub allow_interruptions: bool,
    pub max_duration: f32,
}

impl Conversation {
    pub fn new(id: ConversationId) -> Self {
        Self { id, state: ConversationState::NotStarted, participants: Vec::new(), topics: HashMap::new(), current_topic: None, current_speaker: None, turn_timer: 0.0, turn_duration: 5.0, elapsed: 0.0, history: Vec::new(), pending_interruption: None, allow_interruptions: true, max_duration: 300.0 }
    }
    pub fn add_participant(&mut self, p: ConversationParticipant) { self.participants.push(p); }
    pub fn add_topic(&mut self, topic: ConversationTopic) { self.topics.insert(topic.id, topic); }
    pub fn start(&mut self) {
        self.state = ConversationState::Active;
        if let Some(first) = self.participants.first() { self.current_speaker = Some(first.id); }
    }
    pub fn end(&mut self) { self.state = ConversationState::Ended; }
    pub fn switch_topic(&mut self, topic_id: TopicId) {
        if self.topics.contains_key(&topic_id) { self.current_topic = Some(topic_id); }
    }
    pub fn interrupt(&mut self, by: ParticipantId) {
        if self.allow_interruptions { self.pending_interruption = Some(by); }
    }
    pub fn say(&mut self, speaker: ParticipantId, text: &str) {
        self.history.push((speaker, text.to_string()));
        self.current_speaker = Some(speaker);
    }
    pub fn update(&mut self, dt: f32) {
        if self.state != ConversationState::Active { return; }
        self.elapsed += dt;
        self.turn_timer += dt;
        if self.elapsed >= self.max_duration { self.end(); return; }
        if let Some(interrupter) = self.pending_interruption.take() {
            self.current_speaker = Some(interrupter);
            self.turn_timer = 0.0;
            self.state = ConversationState::Active;
        }
        if self.turn_timer >= self.turn_duration { self.advance_turn(); }
        for p in &mut self.participants { if self.current_speaker != Some(p.id) { p.lose_patience(dt * 0.5); } }
    }
    fn advance_turn(&mut self) {
        self.turn_timer = 0.0;
        if let Some(current) = self.current_speaker {
            let idx = self.participants.iter().position(|p| p.id == current).unwrap_or(0);
            let next_idx = (idx + 1) % self.participants.len();
            self.current_speaker = Some(self.participants[next_idx].id);
        }
    }
    pub fn create_memory(&self) -> ConversationMemory {
        let participant_ids = self.participants.iter().map(|p| p.id).collect();
        let topics = self.topics.values().filter(|t| t.exhausted).map(|t| t.id).collect();
        let mut mem = ConversationMemory::new(self.id, participant_ids, 0.0);
        mem.topics_discussed = topics;
        mem.duration = self.elapsed;
        mem.was_interrupted = self.state == ConversationState::Interrupted;
        mem
    }
    pub fn participant_count(&self) -> usize { self.participants.len() }
    pub fn topic_count(&self) -> usize { self.topics.len() }
}

#[derive(Debug)]
pub struct ConversationSystem {
    pub active_conversations: HashMap<ConversationId, Conversation>,
    pub memories: Vec<ConversationMemory>,
    next_id: u32,
}

impl ConversationSystem {
    pub fn new() -> Self { Self { active_conversations: HashMap::new(), memories: Vec::new(), next_id: 0 } }
    pub fn create_conversation(&mut self) -> ConversationId {
        let id = ConversationId(self.next_id); self.next_id += 1;
        self.active_conversations.insert(id, Conversation::new(id)); id
    }
    pub fn update(&mut self, dt: f32) {
        let ended: Vec<_> = self.active_conversations.iter().filter(|(_, c)| c.state == ConversationState::Ended).map(|(id, _)| *id).collect();
        for id in ended {
            if let Some(conv) = self.active_conversations.remove(&id) { self.memories.push(conv.create_memory()); }
        }
        for conv in self.active_conversations.values_mut() { conv.update(dt); }
    }
    pub fn get(&self, id: ConversationId) -> Option<&Conversation> { self.active_conversations.get(&id) }
    pub fn get_mut(&mut self, id: ConversationId) -> Option<&mut Conversation> { self.active_conversations.get_mut(&id) }
    pub fn active_count(&self) -> usize { self.active_conversations.len() }
    pub fn memory_count(&self) -> usize { self.memories.len() }
}

impl Default for ConversationSystem { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_conversation_basic() {
        let mut conv = Conversation::new(ConversationId(0));
        conv.add_participant(ConversationParticipant::new(ParticipantId(1), "Player"));
        conv.add_participant(ConversationParticipant::new(ParticipantId(2), "NPC"));
        conv.start();
        conv.say(ParticipantId(1), "Hello!");
        assert_eq!(conv.history.len(), 1);
    }
    #[test]
    fn test_skill_check() {
        let mut check = SkillCheck::new(SkillCheckType::Persuasion, 10);
        check.player_skill = 8;
        let result = check.resolve(5);
        assert_eq!(result, SkillCheckResult::Success);
    }
    #[test]
    fn test_conversation_system() {
        let mut sys = ConversationSystem::new();
        let id = sys.create_conversation();
        assert_eq!(sys.active_count(), 1);
        sys.get_mut(id).unwrap().end();
        sys.update(0.1);
        assert_eq!(sys.active_count(), 0);
        assert_eq!(sys.memory_count(), 1);
    }
}
