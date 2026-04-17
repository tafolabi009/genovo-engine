// engine/core/src/command_buffer.rs
// Command buffer: deferred commands, recording, replay, serialization, undo support.
use std::collections::VecDeque;

pub type CommandId = u64;
pub type EntityId = u32;

#[derive(Debug, Clone)]
pub enum Command {
    SpawnEntity { entity_type: String, position: [f32; 3], data: Vec<u8> },
    DestroyEntity { entity: EntityId },
    SetPosition { entity: EntityId, position: [f32; 3] },
    SetRotation { entity: EntityId, rotation: [f32; 4] },
    SetProperty { entity: EntityId, property: String, value: PropertyValue },
    PlaySound { sound_id: String, position: [f32; 3], volume: f32 },
    SpawnParticle { effect: String, position: [f32; 3] },
    ApplyDamage { source: EntityId, target: EntityId, amount: f32 },
    TriggerEvent { event_name: String, data: Vec<u8> },
    Custom { type_id: u32, data: Vec<u8> },
}

#[derive(Debug, Clone)]
pub enum PropertyValue { Bool(bool), Int(i64), Float(f64), String(String), Vec3([f32; 3]) }

#[derive(Debug, Clone)]
pub struct TimestampedCommand { pub id: CommandId, pub frame: u64, pub timestamp: f64, pub command: Command }

pub struct CommandBuffer {
    pending: VecDeque<TimestampedCommand>,
    history: VecDeque<TimestampedCommand>,
    undo_stack: Vec<TimestampedCommand>,
    redo_stack: Vec<TimestampedCommand>,
    next_id: CommandId,
    current_frame: u64,
    current_time: f64,
    max_history: usize,
    is_recording: bool,
    recorded: Vec<TimestampedCommand>,
}

impl CommandBuffer {
    pub fn new() -> Self {
        Self { pending: VecDeque::new(), history: VecDeque::new(), undo_stack: Vec::new(), redo_stack: Vec::new(), next_id: 1, current_frame: 0, current_time: 0.0, max_history: 1000, is_recording: false, recorded: Vec::new() }
    }

    pub fn enqueue(&mut self, command: Command) -> CommandId {
        let id = self.next_id; self.next_id += 1;
        let tc = TimestampedCommand { id, frame: self.current_frame, timestamp: self.current_time, command };
        if self.is_recording { self.recorded.push(tc.clone()); }
        self.pending.push_back(tc);
        id
    }

    pub fn drain(&mut self) -> Vec<TimestampedCommand> {
        let commands: Vec<_> = self.pending.drain(..).collect();
        for cmd in &commands {
            self.history.push_back(cmd.clone());
            if self.history.len() > self.max_history { self.history.pop_front(); }
        }
        commands
    }

    pub fn set_frame(&mut self, frame: u64, time: f64) { self.current_frame = frame; self.current_time = time; }
    pub fn pending_count(&self) -> usize { self.pending.len() }
    pub fn history_count(&self) -> usize { self.history.len() }

    pub fn push_undo(&mut self, command: TimestampedCommand) { self.undo_stack.push(command); self.redo_stack.clear(); }

    pub fn undo(&mut self) -> Option<TimestampedCommand> {
        if let Some(cmd) = self.undo_stack.pop() { self.redo_stack.push(cmd.clone()); Some(cmd) } else { None }
    }

    pub fn redo(&mut self) -> Option<TimestampedCommand> {
        if let Some(cmd) = self.redo_stack.pop() { self.undo_stack.push(cmd.clone()); Some(cmd) } else { None }
    }

    pub fn can_undo(&self) -> bool { !self.undo_stack.is_empty() }
    pub fn can_redo(&self) -> bool { !self.redo_stack.is_empty() }

    pub fn start_recording(&mut self) { self.is_recording = true; self.recorded.clear(); }
    pub fn stop_recording(&mut self) -> Vec<TimestampedCommand> { self.is_recording = false; std::mem::take(&mut self.recorded) }

    pub fn replay(&mut self, commands: &[TimestampedCommand]) { for cmd in commands { self.pending.push_back(cmd.clone()); } }

    pub fn serialize_commands(commands: &[TimestampedCommand]) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&(commands.len() as u32).to_le_bytes());
        for cmd in commands {
            data.extend_from_slice(&cmd.id.to_le_bytes());
            data.extend_from_slice(&cmd.frame.to_le_bytes());
            data.extend_from_slice(&cmd.timestamp.to_le_bytes());
            // Simplified: just store command type
            let type_id: u32 = match &cmd.command {
                Command::SpawnEntity { .. } => 0,
                Command::DestroyEntity { .. } => 1,
                Command::SetPosition { .. } => 2,
                Command::SetRotation { .. } => 3,
                Command::SetProperty { .. } => 4,
                Command::PlaySound { .. } => 5,
                Command::SpawnParticle { .. } => 6,
                Command::ApplyDamage { .. } => 7,
                Command::TriggerEvent { .. } => 8,
                Command::Custom { .. } => 9,
            };
            data.extend_from_slice(&type_id.to_le_bytes());
        }
        data
    }

    pub fn clear(&mut self) { self.pending.clear(); }
    pub fn clear_history(&mut self) { self.history.clear(); self.undo_stack.clear(); self.redo_stack.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_enqueue_drain() {
        let mut buf = CommandBuffer::new();
        buf.enqueue(Command::SpawnEntity { entity_type: "enemy".into(), position: [0.0,0.0,0.0], data: Vec::new() });
        assert_eq!(buf.pending_count(), 1);
        let cmds = buf.drain();
        assert_eq!(cmds.len(), 1);
        assert_eq!(buf.pending_count(), 0);
        assert_eq!(buf.history_count(), 1);
    }
    #[test]
    fn test_undo_redo() {
        let mut buf = CommandBuffer::new();
        let cmd = TimestampedCommand { id: 1, frame: 0, timestamp: 0.0, command: Command::SetPosition { entity: 0, position: [1.0,2.0,3.0] } };
        buf.push_undo(cmd);
        assert!(buf.can_undo());
        let _ = buf.undo();
        assert!(buf.can_redo());
    }
    #[test]
    fn test_recording() {
        let mut buf = CommandBuffer::new();
        buf.start_recording();
        buf.enqueue(Command::DestroyEntity { entity: 0 });
        buf.enqueue(Command::DestroyEntity { entity: 1 });
        let recorded = buf.stop_recording();
        assert_eq!(recorded.len(), 2);
    }
}
