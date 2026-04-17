// engine/gameplay/src/input_combo.rs
//
// Fighting game combo input system for the Genovo gameplay framework.
//
// Provides input sequence detection, direction notation (236P = quarter
// circle forward + punch), cancel windows, combo tree, frame-perfect
// inputs, input buffer, and combo counter.

use std::collections::{HashMap, VecDeque};
use std::fmt;

pub type ComboId = u32;
pub type MoveId = u32;

pub const INPUT_BUFFER_SIZE: usize = 60;
pub const DEFAULT_INPUT_WINDOW: u32 = 10;
pub const DEFAULT_CANCEL_WINDOW: u32 = 5;
pub const MAX_COMBO_LENGTH: usize = 32;
pub const DEFAULT_LINK_WINDOW: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction { Neutral, Up, Down, Left, Right, UpLeft, UpRight, DownLeft, DownRight }

impl Direction {
    pub fn numpad(self) -> u8 {
        match self { Self::DownLeft => 1, Self::Down => 2, Self::DownRight => 3, Self::Left => 4, Self::Neutral => 5, Self::Right => 6, Self::UpLeft => 7, Self::Up => 8, Self::UpRight => 9 }
    }
    pub fn from_numpad(n: u8) -> Self {
        match n { 1 => Self::DownLeft, 2 => Self::Down, 3 => Self::DownRight, 4 => Self::Left, 6 => Self::Right, 7 => Self::UpLeft, 8 => Self::Up, 9 => Self::UpRight, _ => Self::Neutral }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.numpad()) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Button { Punch, Kick, Slash, HeavySlash, Dust, Special, Grab, Dash, Any }

impl fmt::Display for Button {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Punch => write!(f, "P"), Self::Kick => write!(f, "K"), Self::Slash => write!(f, "S"), Self::HeavySlash => write!(f, "HS"), Self::Dust => write!(f, "D"), Self::Special => write!(f, "SP"), Self::Grab => write!(f, "G"), Self::Dash => write!(f, "DA"), Self::Any => write!(f, "*") }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InputEvent {
    pub direction: Direction,
    pub button: Option<Button>,
    pub frame: u64,
    pub pressed: bool,
    pub released: bool,
}

impl InputEvent {
    pub fn direction(dir: Direction, frame: u64) -> Self { Self { direction: dir, button: None, frame, pressed: true, released: false } }
    pub fn button(btn: Button, frame: u64) -> Self { Self { direction: Direction::Neutral, button: Some(btn), frame, pressed: true, released: false } }
    pub fn dir_button(dir: Direction, btn: Button, frame: u64) -> Self { Self { direction: dir, button: Some(btn), frame, pressed: true, released: false } }
}

#[derive(Debug, Clone)]
pub enum MotionInput {
    Direction(Direction),
    Button(Button),
    DirectionButton(Direction, Button),
    QuarterCircleForward(Button),
    QuarterCircleBack(Button),
    DragonPunch(Button),
    HalfCircleForward(Button),
    HalfCircleBack(Button),
    FullCircle(Button),
    DoubleDirection(Direction),
    ChargeDirection { hold: Direction, release: Direction, button: Button, charge_frames: u32 },
    Sequence(Vec<MotionInput>),
}

impl MotionInput {
    pub fn notation(&self) -> String {
        match self {
            Self::Direction(d) => format!("{}", d),
            Self::Button(b) => format!("{}", b),
            Self::DirectionButton(d, b) => format!("{}{}", d, b),
            Self::QuarterCircleForward(b) => format!("236{}", b),
            Self::QuarterCircleBack(b) => format!("214{}", b),
            Self::DragonPunch(b) => format!("623{}", b),
            Self::HalfCircleForward(b) => format!("41236{}", b),
            Self::HalfCircleBack(b) => format!("63214{}", b),
            Self::FullCircle(b) => format!("632147896{}", b),
            Self::DoubleDirection(d) => format!("{}{}", d, d),
            Self::ChargeDirection { hold, release, button, .. } => format!("[{}]{}+{}", hold, release, button),
            Self::Sequence(inputs) => inputs.iter().map(|i| i.notation()).collect::<Vec<_>>().join(" "),
        }
    }

    pub fn required_directions(&self) -> Vec<Direction> {
        match self {
            Self::QuarterCircleForward(_) => vec![Direction::Down, Direction::DownRight, Direction::Right],
            Self::QuarterCircleBack(_) => vec![Direction::Down, Direction::DownLeft, Direction::Left],
            Self::DragonPunch(_) => vec![Direction::Right, Direction::Down, Direction::DownRight],
            Self::HalfCircleForward(_) => vec![Direction::Left, Direction::DownLeft, Direction::Down, Direction::DownRight, Direction::Right],
            Self::HalfCircleBack(_) => vec![Direction::Right, Direction::DownRight, Direction::Down, Direction::DownLeft, Direction::Left],
            Self::Direction(d) => vec![*d],
            _ => Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComboMove {
    pub id: MoveId,
    pub name: String,
    pub input: MotionInput,
    pub damage: f32,
    pub hitstun: u32,
    pub blockstun: u32,
    pub startup_frames: u32,
    pub active_frames: u32,
    pub recovery_frames: u32,
    pub cancel_window_start: u32,
    pub cancel_window_end: u32,
    pub can_cancel_into: Vec<MoveId>,
    pub is_special: bool,
    pub is_super: bool,
    pub meter_gain: f32,
    pub meter_cost: f32,
    pub proration: f32,
    pub launches: bool,
    pub wall_bounces: bool,
    pub ground_bounces: bool,
}

impl ComboMove {
    pub fn new(id: MoveId, name: &str, input: MotionInput) -> Self {
        Self {
            id, name: name.to_string(), input, damage: 10.0, hitstun: 15, blockstun: 10,
            startup_frames: 5, active_frames: 3, recovery_frames: 10,
            cancel_window_start: 0, cancel_window_end: 0, can_cancel_into: Vec::new(),
            is_special: false, is_super: false, meter_gain: 0.0, meter_cost: 0.0,
            proration: 1.0, launches: false, wall_bounces: false, ground_bounces: false,
        }
    }
    pub fn total_frames(&self) -> u32 { self.startup_frames + self.active_frames + self.recovery_frames }
    pub fn is_cancelable_at(&self, frame: u32) -> bool { frame >= self.cancel_window_start && frame <= self.cancel_window_end }
}

#[derive(Debug, Clone)]
pub struct ComboTreeNode {
    pub move_id: MoveId,
    pub children: Vec<ComboTreeNode>,
    pub is_ender: bool,
    pub link_window: u32,
}

impl ComboTreeNode {
    pub fn new(move_id: MoveId) -> Self { Self { move_id, children: Vec::new(), is_ender: false, link_window: DEFAULT_LINK_WINDOW } }
    pub fn add_child(&mut self, child: ComboTreeNode) { self.children.push(child); }
    pub fn with_children(mut self, children: Vec<ComboTreeNode>) -> Self { self.children = children; self }
    pub fn ender(mut self) -> Self { self.is_ender = true; self }
}

#[derive(Debug, Clone)]
pub struct ComboDefinition {
    pub id: ComboId,
    pub name: String,
    pub tree: ComboTreeNode,
    pub min_hits: u32,
    pub max_damage: f32,
}

impl ComboDefinition {
    pub fn new(id: ComboId, name: &str, tree: ComboTreeNode) -> Self { Self { id, name: name.to_string(), tree, min_hits: 2, max_damage: f32::MAX } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComboState { Idle, InCombo, Dropped, Completed }

#[derive(Debug, Clone)]
pub struct ComboCounter {
    pub hit_count: u32,
    pub total_damage: f32,
    pub max_combo: u32,
    pub state: ComboState,
    pub current_proration: f32,
    pub combo_timer: u32,
    pub moves_used: Vec<MoveId>,
    pub is_valid: bool,
}

impl ComboCounter {
    pub fn new() -> Self {
        Self { hit_count: 0, total_damage: 0.0, max_combo: 0, state: ComboState::Idle, current_proration: 1.0, combo_timer: 0, moves_used: Vec::new(), is_valid: true }
    }
    pub fn register_hit(&mut self, damage: f32, proration: f32, move_id: MoveId) {
        self.hit_count += 1;
        let prorated_damage = damage * self.current_proration;
        self.total_damage += prorated_damage;
        self.current_proration *= proration;
        self.state = ComboState::InCombo;
        self.combo_timer = 0;
        self.moves_used.push(move_id);
        if self.hit_count > self.max_combo { self.max_combo = self.hit_count; }
    }
    pub fn drop(&mut self) { self.state = ComboState::Dropped; }
    pub fn complete(&mut self) { self.state = ComboState::Completed; }
    pub fn reset(&mut self) {
        self.hit_count = 0; self.total_damage = 0.0; self.current_proration = 1.0;
        self.combo_timer = 0; self.moves_used.clear(); self.state = ComboState::Idle; self.is_valid = true;
    }
    pub fn update(&mut self, hitstun_remaining: u32) {
        if self.state == ComboState::InCombo { self.combo_timer += 1; if hitstun_remaining == 0 { self.drop(); } }
    }
}

pub struct InputBuffer {
    buffer: VecDeque<InputEvent>,
    capacity: usize,
    current_frame: u64,
}

impl InputBuffer {
    pub fn new(capacity: usize) -> Self { Self { buffer: VecDeque::with_capacity(capacity), capacity, current_frame: 0 } }
    pub fn push(&mut self, event: InputEvent) { if self.buffer.len() >= self.capacity { self.buffer.pop_front(); } self.buffer.push_back(event); }
    pub fn advance_frame(&mut self) { self.current_frame += 1; }
    pub fn current_frame(&self) -> u64 { self.current_frame }
    pub fn recent(&self, frames: u32) -> Vec<&InputEvent> {
        let min_frame = self.current_frame.saturating_sub(frames as u64);
        self.buffer.iter().filter(|e| e.frame >= min_frame).collect()
    }
    pub fn has_direction_sequence(&self, dirs: &[Direction], window: u32) -> bool {
        let recent = self.recent(window);
        let mut dir_idx = 0;
        for event in &recent {
            if dir_idx < dirs.len() && event.direction == dirs[dir_idx] { dir_idx += 1; }
        }
        dir_idx >= dirs.len()
    }
    pub fn has_button_in_window(&self, button: Button, window: u32) -> bool {
        self.recent(window).iter().any(|e| e.button == Some(button) && e.pressed)
    }
    pub fn clear(&mut self) { self.buffer.clear(); }
    pub fn len(&self) -> usize { self.buffer.len() }
    pub fn is_empty(&self) -> bool { self.buffer.is_empty() }
}

pub struct InputComboSystem {
    moves: HashMap<MoveId, ComboMove>,
    combos: HashMap<ComboId, ComboDefinition>,
    input_buffer: InputBuffer,
    combo_counter: ComboCounter,
    input_window: u32,
    cancel_window: u32,
    current_move: Option<MoveId>,
    current_move_frame: u32,
    detected_moves: Vec<MoveId>,
}

impl InputComboSystem {
    pub fn new() -> Self {
        Self {
            moves: HashMap::new(), combos: HashMap::new(),
            input_buffer: InputBuffer::new(INPUT_BUFFER_SIZE),
            combo_counter: ComboCounter::new(), input_window: DEFAULT_INPUT_WINDOW,
            cancel_window: DEFAULT_CANCEL_WINDOW, current_move: None, current_move_frame: 0,
            detected_moves: Vec::new(),
        }
    }
    pub fn register_move(&mut self, m: ComboMove) { self.moves.insert(m.id, m); }
    pub fn register_combo(&mut self, c: ComboDefinition) { self.combos.insert(c.id, c); }
    pub fn push_input(&mut self, event: InputEvent) { self.input_buffer.push(event); }
    pub fn update(&mut self) {
        self.input_buffer.advance_frame();
        self.detected_moves.clear();
        if let Some(mid) = self.current_move { self.current_move_frame += 1; }
        // Detect motions.
        let move_ids: Vec<MoveId> = self.moves.keys().cloned().collect();
        for mid in move_ids {
            let m = self.moves.get(&mid).unwrap().clone();
            if self.check_motion_input(&m.input) {
                // Check cancel window.
                let can_cancel = if let Some(current) = self.current_move {
                    if let Some(cm) = self.moves.get(&current) {
                        cm.can_cancel_into.contains(&mid) && cm.is_cancelable_at(self.current_move_frame)
                    } else { true }
                } else { true };
                if can_cancel {
                    self.detected_moves.push(mid);
                    self.current_move = Some(mid);
                    self.current_move_frame = 0;
                    self.combo_counter.register_hit(m.damage, m.proration, mid);
                }
            }
        }
    }
    fn check_motion_input(&self, motion: &MotionInput) -> bool {
        match motion {
            MotionInput::Button(b) => self.input_buffer.has_button_in_window(*b, 2),
            MotionInput::DirectionButton(d, b) => {
                let recent = self.input_buffer.recent(2);
                recent.iter().any(|e| e.direction == *d && e.button == Some(*b))
            }
            MotionInput::QuarterCircleForward(b) | MotionInput::QuarterCircleBack(b) | MotionInput::DragonPunch(b) => {
                let dirs = motion.required_directions();
                self.input_buffer.has_direction_sequence(&dirs, self.input_window) && self.input_buffer.has_button_in_window(*b, 3)
            }
            _ => false,
        }
    }
    pub fn detected_moves(&self) -> &[MoveId] { &self.detected_moves }
    pub fn combo_counter(&self) -> &ComboCounter { &self.combo_counter }
    pub fn combo_counter_mut(&mut self) -> &mut ComboCounter { &mut self.combo_counter }
    pub fn reset_combo(&mut self) { self.combo_counter.reset(); self.current_move = None; self.current_move_frame = 0; }
    pub fn current_move(&self) -> Option<MoveId> { self.current_move }
    pub fn get_move(&self, id: MoveId) -> Option<&ComboMove> { self.moves.get(&id) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_numpad() {
        assert_eq!(Direction::Down.numpad(), 2);
        assert_eq!(Direction::from_numpad(6), Direction::Right);
    }

    #[test]
    fn test_input_buffer() {
        let mut buf = InputBuffer::new(32);
        buf.push(InputEvent::direction(Direction::Down, 0));
        buf.advance_frame();
        buf.push(InputEvent::direction(Direction::DownRight, 1));
        buf.advance_frame();
        buf.push(InputEvent::direction(Direction::Right, 2));
        buf.advance_frame();
        assert!(buf.has_direction_sequence(&[Direction::Down, Direction::DownRight, Direction::Right], 10));
    }

    #[test]
    fn test_combo_counter() {
        let mut counter = ComboCounter::new();
        counter.register_hit(100.0, 0.9, 1);
        counter.register_hit(80.0, 0.9, 2);
        assert_eq!(counter.hit_count, 2);
        assert!(counter.total_damage > 150.0);
        assert_eq!(counter.max_combo, 2);
    }

    #[test]
    fn test_motion_notation() {
        let m = MotionInput::QuarterCircleForward(Button::Punch);
        assert_eq!(m.notation(), "236P");
    }
}
