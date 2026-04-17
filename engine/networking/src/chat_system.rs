// engine/networking/src/chat_system.rs
//
// In-game chat: text channels (all/team/whisper), message history, chat
// commands (/team, /all, /whisper), message filtering (profanity filter stub),
// chat UI data, chat events.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

pub type ChatPlayerId = u64;

// --- Chat channel ---
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatChannel { All, Team, Whisper, System, Admin, Party, Guild, Custom(u32) }

impl ChatChannel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::All => "All", Self::Team => "Team", Self::Whisper => "Whisper",
            Self::System => "System", Self::Admin => "Admin", Self::Party => "Party",
            Self::Guild => "Guild", Self::Custom(_) => "Custom",
        }
    }
    pub fn color(&self) -> [f32; 4] {
        match self {
            Self::All => [1.0, 1.0, 1.0, 1.0], Self::Team => [0.4, 0.8, 1.0, 1.0],
            Self::Whisper => [0.9, 0.5, 0.9, 1.0], Self::System => [1.0, 1.0, 0.3, 1.0],
            Self::Admin => [1.0, 0.3, 0.3, 1.0], Self::Party => [0.3, 1.0, 0.6, 1.0],
            Self::Guild => [0.3, 1.0, 0.3, 1.0], Self::Custom(_) => [0.7, 0.7, 0.7, 1.0],
        }
    }
}

// --- Chat message ---
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub id: u64,
    pub sender_id: ChatPlayerId,
    pub sender_name: String,
    pub channel: ChatChannel,
    pub text: String,
    pub timestamp: f64,
    pub filtered: bool,
    pub original_text: Option<String>,
    pub recipient_id: Option<ChatPlayerId>,
    pub team_id: Option<u8>,
}

impl ChatMessage {
    pub fn new(id: u64, sender_id: ChatPlayerId, sender_name: &str, channel: ChatChannel, text: &str) -> Self {
        Self {
            id, sender_id, sender_name: sender_name.to_string(), channel,
            text: text.to_string(), timestamp: 0.0, filtered: false,
            original_text: None, recipient_id: None, team_id: None,
        }
    }
    pub fn format_display(&self) -> String {
        match self.channel {
            ChatChannel::Whisper => format!("[Whisper] {}: {}", self.sender_name, self.text),
            ChatChannel::Team => format!("[Team] {}: {}", self.sender_name, self.text),
            ChatChannel::System => format!("[System] {}", self.text),
            ChatChannel::Admin => format!("[Admin] {}: {}", self.sender_name, self.text),
            _ => format!("{}: {}", self.sender_name, self.text),
        }
    }
}

// --- Chat command ---
#[derive(Debug, Clone)]
pub struct ParsedChatInput {
    pub is_command: bool,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub channel: ChatChannel,
    pub target_player: Option<String>,
    pub message_text: String,
}

pub fn parse_chat_input(input: &str, default_channel: ChatChannel) -> ParsedChatInput {
    let trimmed = input.trim();
    if trimmed.starts_with('/') {
        let parts: Vec<&str> = trimmed[1..].splitn(2, ' ').collect();
        let cmd = parts[0].to_lowercase();
        let rest = if parts.len() > 1 { parts[1] } else { "" };

        match cmd.as_str() {
            "all" | "say" => ParsedChatInput {
                is_command: false, command: None, args: Vec::new(),
                channel: ChatChannel::All, target_player: None, message_text: rest.to_string(),
            },
            "team" | "t" => ParsedChatInput {
                is_command: false, command: None, args: Vec::new(),
                channel: ChatChannel::Team, target_player: None, message_text: rest.to_string(),
            },
            "whisper" | "w" | "tell" | "msg" => {
                let whisper_parts: Vec<&str> = rest.splitn(2, ' ').collect();
                let target = whisper_parts.first().unwrap_or(&"").to_string();
                let msg = if whisper_parts.len() > 1 { whisper_parts[1] } else { "" };
                ParsedChatInput {
                    is_command: false, command: None, args: Vec::new(),
                    channel: ChatChannel::Whisper, target_player: Some(target), message_text: msg.to_string(),
                }
            }
            "party" | "p" => ParsedChatInput {
                is_command: false, command: None, args: Vec::new(),
                channel: ChatChannel::Party, target_player: None, message_text: rest.to_string(),
            },
            "guild" | "g" => ParsedChatInput {
                is_command: false, command: None, args: Vec::new(),
                channel: ChatChannel::Guild, target_player: None, message_text: rest.to_string(),
            },
            _ => {
                let args: Vec<String> = rest.split_whitespace().map(|s| s.to_string()).collect();
                ParsedChatInput {
                    is_command: true, command: Some(cmd), args,
                    channel: default_channel, target_player: None, message_text: String::new(),
                }
            }
        }
    } else {
        ParsedChatInput {
            is_command: false, command: None, args: Vec::new(),
            channel: default_channel, target_player: None, message_text: trimmed.to_string(),
        }
    }
}

// --- Profanity filter ---
pub struct ProfanityFilter {
    blocked_words: Vec<String>,
    replacement_char: char,
    enabled: bool,
}

impl ProfanityFilter {
    pub fn new() -> Self { Self { blocked_words: Vec::new(), replacement_char: '*', enabled: true } }
    pub fn add_word(&mut self, word: &str) { self.blocked_words.push(word.to_lowercase()); }
    pub fn add_words(&mut self, words: &[&str]) { for w in words { self.add_word(w); } }
    pub fn set_enabled(&mut self, enabled: bool) { self.enabled = enabled; }

    pub fn filter(&self, text: &str) -> (String, bool) {
        if !self.enabled { return (text.to_string(), false); }
        let mut result = text.to_string();
        let mut was_filtered = false;
        let lower = text.to_lowercase();
        for word in &self.blocked_words {
            if lower.contains(word) {
                let replacement = self.replacement_char.to_string().repeat(word.len());
                // Case-insensitive replacement
                let mut idx = 0;
                let mut new_result = String::new();
                let lower_result = result.to_lowercase();
                while let Some(pos) = lower_result[idx..].find(word) {
                    new_result.push_str(&result[idx..idx + pos]);
                    new_result.push_str(&replacement);
                    idx += pos + word.len();
                    was_filtered = true;
                }
                new_result.push_str(&result[idx..]);
                result = new_result;
            }
        }
        (result, was_filtered)
    }
}

// --- Chat event ---
#[derive(Debug, Clone)]
pub enum ChatEvent {
    MessageSent { message: ChatMessage },
    MessageReceived { message: ChatMessage },
    MessageFiltered { message_id: u64 },
    ChannelChanged { channel: ChatChannel },
    PlayerMuted { player_id: ChatPlayerId },
    PlayerUnmuted { player_id: ChatPlayerId },
    CommandExecuted { command: String, args: Vec<String> },
}

// --- Chat UI data ---
#[derive(Debug, Clone)]
pub struct ChatUIState {
    pub visible: bool,
    pub input_active: bool,
    pub input_text: String,
    pub active_channel: ChatChannel,
    pub scroll_offset: usize,
    pub max_visible_messages: usize,
    pub fade_time: f32,
    pub fade_duration: f32,
    pub auto_hide: bool,
}

impl Default for ChatUIState {
    fn default() -> Self {
        Self {
            visible: true, input_active: false, input_text: String::new(),
            active_channel: ChatChannel::All, scroll_offset: 0,
            max_visible_messages: 10, fade_time: 0.0, fade_duration: 8.0, auto_hide: true,
        }
    }
}

impl ChatUIState {
    pub fn open_input(&mut self) { self.input_active = true; self.visible = true; self.fade_time = 0.0; }
    pub fn close_input(&mut self) { self.input_active = false; self.input_text.clear(); }
    pub fn submit_input(&mut self) -> Option<String> {
        if self.input_text.is_empty() { return None; }
        let text = self.input_text.clone();
        self.input_text.clear();
        self.input_active = false;
        Some(text)
    }
    pub fn update(&mut self, dt: f32) {
        if self.auto_hide && !self.input_active {
            self.fade_time += dt;
            if self.fade_time > self.fade_duration { self.visible = false; }
        }
    }
    pub fn opacity(&self) -> f32 {
        if self.input_active { return 1.0; }
        if self.fade_time < self.fade_duration - 1.0 { 1.0 }
        else { ((self.fade_duration - self.fade_time) / 1.0).clamp(0.0, 1.0) }
    }
    pub fn show_messages(&mut self) { self.visible = true; self.fade_time = 0.0; }
}

// --- Chat system ---
pub struct ChatSystem {
    messages: VecDeque<ChatMessage>,
    max_messages: usize,
    next_message_id: u64,
    muted_players: Vec<ChatPlayerId>,
    profanity_filter: ProfanityFilter,
    pub ui_state: ChatUIState,
    events: Vec<ChatEvent>,
    player_names: HashMap<ChatPlayerId, String>,
    local_player_id: ChatPlayerId,
    local_team_id: Option<u8>,
    rate_limit: HashMap<ChatPlayerId, (u32, f64)>,
    max_messages_per_second: u32,
    command_handlers: HashMap<String, String>,
}

impl ChatSystem {
    pub fn new(local_player_id: ChatPlayerId) -> Self {
        Self {
            messages: VecDeque::new(), max_messages: 200, next_message_id: 1,
            muted_players: Vec::new(), profanity_filter: ProfanityFilter::new(),
            ui_state: ChatUIState::default(), events: Vec::new(),
            player_names: HashMap::new(), local_player_id, local_team_id: None,
            rate_limit: HashMap::new(), max_messages_per_second: 3,
            command_handlers: HashMap::new(),
        }
    }

    pub fn register_player(&mut self, id: ChatPlayerId, name: &str) { self.player_names.insert(id, name.to_string()); }
    pub fn unregister_player(&mut self, id: ChatPlayerId) { self.player_names.remove(&id); }
    pub fn set_local_team(&mut self, team: Option<u8>) { self.local_team_id = team; }

    pub fn send_message(&mut self, text: &str, timestamp: f64) -> Option<ChatMessage> {
        let parsed = parse_chat_input(text, self.ui_state.active_channel);

        if parsed.is_command {
            if let Some(cmd) = &parsed.command {
                self.events.push(ChatEvent::CommandExecuted { command: cmd.clone(), args: parsed.args.clone() });
            }
            return None;
        }

        if parsed.message_text.is_empty() { return None; }

        let sender_name = self.player_names.get(&self.local_player_id).cloned().unwrap_or("Unknown".into());
        let (filtered_text, was_filtered) = self.profanity_filter.filter(&parsed.message_text);

        let mut msg = ChatMessage::new(self.next_message_id, self.local_player_id, &sender_name, parsed.channel, &filtered_text);
        self.next_message_id += 1;
        msg.timestamp = timestamp;
        msg.team_id = self.local_team_id;
        if was_filtered { msg.filtered = true; msg.original_text = Some(parsed.message_text.clone()); }

        self.add_message(msg.clone());
        self.events.push(ChatEvent::MessageSent { message: msg.clone() });
        self.ui_state.show_messages();
        Some(msg)
    }

    pub fn receive_message(&mut self, msg: ChatMessage) {
        if self.muted_players.contains(&msg.sender_id) { return; }
        self.add_message(msg.clone());
        self.events.push(ChatEvent::MessageReceived { message: msg });
        self.ui_state.show_messages();
    }

    pub fn mute_player(&mut self, id: ChatPlayerId) {
        if !self.muted_players.contains(&id) { self.muted_players.push(id); self.events.push(ChatEvent::PlayerMuted { player_id: id }); }
    }

    pub fn unmute_player(&mut self, id: ChatPlayerId) {
        self.muted_players.retain(|&p| p != id);
        self.events.push(ChatEvent::PlayerUnmuted { player_id: id });
    }

    pub fn is_muted(&self, id: ChatPlayerId) -> bool { self.muted_players.contains(&id) }

    pub fn get_messages(&self, channel: Option<ChatChannel>, count: usize) -> Vec<&ChatMessage> {
        let iter = self.messages.iter().rev();
        if let Some(ch) = channel {
            iter.filter(|m| m.channel == ch).take(count).collect::<Vec<_>>().into_iter().rev().collect()
        } else {
            iter.take(count).collect::<Vec<_>>().into_iter().rev().collect()
        }
    }

    pub fn visible_messages(&self) -> Vec<&ChatMessage> {
        self.get_messages(None, self.ui_state.max_visible_messages)
    }

    pub fn add_system_message(&mut self, text: &str, timestamp: f64) {
        let mut msg = ChatMessage::new(self.next_message_id, 0, "System", ChatChannel::System, text);
        self.next_message_id += 1;
        msg.timestamp = timestamp;
        self.add_message(msg);
        self.ui_state.show_messages();
    }

    pub fn drain_events(&mut self) -> Vec<ChatEvent> { std::mem::take(&mut self.events) }
    pub fn profanity_filter_mut(&mut self) -> &mut ProfanityFilter { &mut self.profanity_filter }

    pub fn update(&mut self, dt: f32) { self.ui_state.update(dt); }

    fn add_message(&mut self, msg: ChatMessage) {
        self.messages.push_back(msg);
        while self.messages.len() > self.max_messages { self.messages.pop_front(); }
    }

    pub fn message_count(&self) -> usize { self.messages.len() }
    pub fn clear_messages(&mut self) { self.messages.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_all_chat() {
        let parsed = parse_chat_input("hello world", ChatChannel::All);
        assert!(!parsed.is_command);
        assert_eq!(parsed.channel, ChatChannel::All);
        assert_eq!(parsed.message_text, "hello world");
    }

    #[test]
    fn test_parse_team_command() {
        let parsed = parse_chat_input("/team push B!", ChatChannel::All);
        assert!(!parsed.is_command);
        assert_eq!(parsed.channel, ChatChannel::Team);
        assert_eq!(parsed.message_text, "push B!");
    }

    #[test]
    fn test_parse_whisper() {
        let parsed = parse_chat_input("/whisper PlayerA hello there", ChatChannel::All);
        assert_eq!(parsed.channel, ChatChannel::Whisper);
        assert_eq!(parsed.target_player, Some("PlayerA".into()));
        assert_eq!(parsed.message_text, "hello there");
    }

    #[test]
    fn test_profanity_filter() {
        let mut filter = ProfanityFilter::new();
        filter.add_word("bad");
        let (result, filtered) = filter.filter("this is bad stuff");
        assert!(filtered);
        assert!(result.contains("***"));
        assert!(!result.contains("bad"));
    }

    #[test]
    fn test_chat_system_send_receive() {
        let mut chat = ChatSystem::new(1);
        chat.register_player(1, "Player1");
        chat.register_player(2, "Player2");

        let msg = chat.send_message("Hello!", 0.0);
        assert!(msg.is_some());
        assert_eq!(chat.message_count(), 1);

        let received = ChatMessage::new(100, 2, "Player2", ChatChannel::All, "Hi back!");
        chat.receive_message(received);
        assert_eq!(chat.message_count(), 2);
    }

    #[test]
    fn test_mute_player() {
        let mut chat = ChatSystem::new(1);
        chat.register_player(2, "Troll");
        chat.mute_player(2);
        let msg = ChatMessage::new(1, 2, "Troll", ChatChannel::All, "spam");
        chat.receive_message(msg);
        assert_eq!(chat.message_count(), 0); // Muted, not added
    }
}
