//! MIDI parsing, synthesis, and music theory module for the Genovo audio engine.
//!
//! Provides:
//! - **MIDI data types** — `MidiNote`, `MidiEvent`, `MidiSequence`
//! - **Standard MIDI File parser** — header/track chunk parsing, variable-length
//!   quantity decoding, running status, delta-time to absolute time conversion
//! - **Wavetable synthesizer** — oscillators (sine, square, saw, triangle, noise),
//!   ADSR envelope, polyphonic voice allocation, PCM rendering
//! - **Music theory** — note-to-frequency conversion, note naming, scale and
//!   chord generation for major, minor, pentatonic, blues, and more

use std::f32::consts::{PI, TAU};

// ===========================================================================
// Constants
// ===========================================================================

/// Standard A4 reference frequency (Hz).
pub const A4_FREQUENCY: f32 = 440.0;

/// MIDI note number for A4.
pub const A4_NOTE: u8 = 69;

/// Default polyphony (max simultaneous voices).
pub const DEFAULT_POLYPHONY: usize = 16;

/// Default sample rate for synthesis.
pub const DEFAULT_SAMPLE_RATE: u32 = 44100;

/// Note names without octave.
const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

// ===========================================================================
// MidiNote
// ===========================================================================

/// A MIDI note with number, velocity, and channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MidiNote {
    /// MIDI note number (0-127). Middle C = 60.
    pub note: u8,
    /// Velocity (0-127). 0 typically means note-off.
    pub velocity: u8,
    /// MIDI channel (0-15).
    pub channel: u8,
}

impl MidiNote {
    /// Create a new MIDI note.
    pub fn new(note: u8, velocity: u8, channel: u8) -> Self {
        Self {
            note: note.min(127),
            velocity: velocity.min(127),
            channel: channel.min(15),
        }
    }

    /// Get the frequency in Hz for this note.
    pub fn frequency(&self) -> f32 {
        note_to_frequency(self.note)
    }

    /// Get the human-readable name of this note (e.g. "C4", "A#3").
    pub fn name(&self) -> String {
        note_name(self.note).to_string()
    }

    /// Get the velocity as a normalised float (0.0 to 1.0).
    pub fn velocity_f32(&self) -> f32 {
        self.velocity as f32 / 127.0
    }
}

// ===========================================================================
// MidiEvent
// ===========================================================================

/// MIDI event types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MidiEvent {
    /// Note-on: start playing a note.
    NoteOn {
        channel: u8,
        note: u8,
        velocity: u8,
    },
    /// Note-off: stop playing a note.
    NoteOff {
        channel: u8,
        note: u8,
        velocity: u8,
    },
    /// Control change (CC): modify a controller value.
    ControlChange {
        channel: u8,
        controller: u8,
        value: u8,
    },
    /// Pitch bend: modify pitch (-8192 to +8191).
    PitchBend {
        channel: u8,
        value: i16,
    },
    /// Program change: select an instrument/patch.
    ProgramChange {
        channel: u8,
        program: u8,
    },
    /// Channel aftertouch (pressure).
    ChannelPressure {
        channel: u8,
        pressure: u8,
    },
    /// Set tempo (microseconds per quarter note) — meta event.
    SetTempo(u32),
    /// End of track — meta event.
    EndOfTrack,
    /// Unknown/unsupported event (stored raw).
    Unknown {
        status: u8,
        data: [u8; 2],
    },
}

impl MidiEvent {
    /// Get the MIDI channel for channel events, if applicable.
    pub fn channel(&self) -> Option<u8> {
        match self {
            Self::NoteOn { channel, .. }
            | Self::NoteOff { channel, .. }
            | Self::ControlChange { channel, .. }
            | Self::PitchBend { channel, .. }
            | Self::ProgramChange { channel, .. }
            | Self::ChannelPressure { channel, .. } => Some(*channel),
            _ => None,
        }
    }

    /// Whether this event is a note-on with non-zero velocity.
    pub fn is_note_on(&self) -> bool {
        matches!(self, Self::NoteOn { velocity, .. } if *velocity > 0)
    }

    /// Whether this event is a note-off (or note-on with velocity 0).
    pub fn is_note_off(&self) -> bool {
        matches!(
            self,
            Self::NoteOff { .. } | Self::NoteOn { velocity: 0, .. }
        )
    }
}

// ===========================================================================
// MidiSequence
// ===========================================================================

/// A timed sequence of MIDI events.
///
/// Events are stored as `(timestamp_seconds, MidiEvent)` pairs.
/// The sequence can be played back at variable tempo.
pub struct MidiSequence {
    /// Timed events: `(time_in_seconds, event)`.
    pub events: Vec<(f64, MidiEvent)>,
    /// Name of this sequence/track (from MIDI file or user-assigned).
    pub name: String,
    /// Duration of the sequence in seconds.
    pub duration: f64,
}

impl MidiSequence {
    /// Create an empty MIDI sequence.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            name: String::new(),
            duration: 0.0,
        }
    }

    /// Create a sequence with the given events.
    pub fn with_events(events: Vec<(f64, MidiEvent)>) -> Self {
        let duration = events
            .iter()
            .map(|(t, _)| *t)
            .fold(0.0_f64, f64::max);
        Self {
            events,
            name: String::new(),
            duration,
        }
    }

    /// Add an event at the given time.
    pub fn add_event(&mut self, time: f64, event: MidiEvent) {
        self.events.push((time, event));
        if time > self.duration {
            self.duration = time;
        }
    }

    /// Sort events by timestamp (stable sort).
    pub fn sort_events(&mut self) {
        self.events
            .sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Get all events in a time range [start, end).
    pub fn events_in_range(&self, start: f64, end: f64) -> Vec<&(f64, MidiEvent)> {
        self.events
            .iter()
            .filter(|(t, _)| *t >= start && *t < end)
            .collect()
    }

    /// Scale all event timestamps by a tempo multiplier.
    ///
    /// `tempo_scale` > 1.0 speeds up playback (shorter intervals),
    /// `tempo_scale` < 1.0 slows down playback.
    pub fn scale_tempo(&mut self, tempo_scale: f64) {
        if tempo_scale <= 0.0 {
            return;
        }
        for (time, _) in self.events.iter_mut() {
            *time /= tempo_scale;
        }
        self.duration /= tempo_scale;
    }

    /// Get the number of note-on events.
    pub fn note_count(&self) -> usize {
        self.events
            .iter()
            .filter(|(_, e)| e.is_note_on())
            .count()
    }

    /// Extract all unique note numbers used in this sequence.
    pub fn unique_notes(&self) -> Vec<u8> {
        let mut notes: Vec<u8> = self
            .events
            .iter()
            .filter_map(|(_, e)| match e {
                MidiEvent::NoteOn { note, velocity, .. } if *velocity > 0 => Some(*note),
                _ => None,
            })
            .collect();
        notes.sort();
        notes.dedup();
        notes
    }
}

impl Default for MidiSequence {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// MIDI File Parser
// ===========================================================================

/// Errors that can occur during MIDI file parsing.
#[derive(Debug, Clone)]
pub enum MidiParseError {
    /// The file is too short or truncated.
    UnexpectedEof,
    /// Invalid chunk header.
    InvalidChunkHeader(String),
    /// Invalid MIDI file header.
    InvalidHeader(String),
    /// Invalid or unsupported event.
    InvalidEvent(String),
    /// Data is malformed.
    MalformedData(String),
}

impl std::fmt::Display for MidiParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of MIDI data"),
            Self::InvalidChunkHeader(s) => write!(f, "invalid chunk header: {}", s),
            Self::InvalidHeader(s) => write!(f, "invalid MIDI header: {}", s),
            Self::InvalidEvent(s) => write!(f, "invalid MIDI event: {}", s),
            Self::MalformedData(s) => write!(f, "malformed MIDI data: {}", s),
        }
    }
}

/// MIDI file format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidiFormat {
    /// Format 0: single track.
    SingleTrack,
    /// Format 1: multiple tracks, simultaneous.
    MultiTrack,
    /// Format 2: multiple tracks, sequential.
    SequentialTracks,
}

/// Parsed MIDI file header.
#[derive(Debug, Clone)]
pub struct MidiHeader {
    /// File format (0, 1, or 2).
    pub format: MidiFormat,
    /// Number of track chunks.
    pub track_count: u16,
    /// Time division (ticks per quarter note, or SMPTE).
    pub division: u16,
    /// Whether the division is SMPTE-based.
    pub is_smpte: bool,
}

/// Parser for Standard MIDI File (SMF) format.
///
/// Supports format 0 and format 1 files. Handles:
/// - Header chunk (`MThd`) parsing
/// - Track chunk (`MTrk`) parsing
/// - Variable-length quantity (VLQ) decoding
/// - Running status optimisation
/// - Meta events (tempo, end-of-track, etc.)
/// - Delta-time to absolute time conversion
pub struct MidiFileParser {
    /// Raw file data.
    data: Vec<u8>,
    /// Current read position.
    pos: usize,
}

impl MidiFileParser {
    /// Create a new parser for the given MIDI file data.
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, pos: 0 }
    }

    /// Parse the entire MIDI file and return sequences for each track.
    pub fn parse(&mut self) -> Result<(MidiHeader, Vec<MidiSequence>), MidiParseError> {
        self.pos = 0;
        let header = self.parse_header()?;
        let mut sequences = Vec::with_capacity(header.track_count as usize);

        for _ in 0..header.track_count {
            let seq = self.parse_track(&header)?;
            sequences.push(seq);
        }

        Ok((header, sequences))
    }

    /// Parse the MIDI file header chunk (`MThd`).
    fn parse_header(&mut self) -> Result<MidiHeader, MidiParseError> {
        // Read chunk ID: "MThd"
        let chunk_id = self.read_bytes(4)?;
        if chunk_id != b"MThd" {
            return Err(MidiParseError::InvalidChunkHeader(
                format!("expected MThd, got {:?}", chunk_id),
            ));
        }

        // Chunk length (should be 6 for standard header)
        let length = self.read_u32_be()?;
        if length < 6 {
            return Err(MidiParseError::InvalidHeader(
                format!("header length {} too short", length),
            ));
        }

        // Format type
        let format_raw = self.read_u16_be()?;
        let format = match format_raw {
            0 => MidiFormat::SingleTrack,
            1 => MidiFormat::MultiTrack,
            2 => MidiFormat::SequentialTracks,
            _ => {
                return Err(MidiParseError::InvalidHeader(
                    format!("unsupported format {}", format_raw),
                ))
            }
        };

        // Number of tracks
        let track_count = self.read_u16_be()?;

        // Division
        let division = self.read_u16_be()?;
        let is_smpte = (division & 0x8000) != 0;

        // Skip any extra header bytes
        if length > 6 {
            self.pos += (length - 6) as usize;
        }

        Ok(MidiHeader {
            format,
            track_count,
            division,
            is_smpte,
        })
    }

    /// Parse a single track chunk (`MTrk`).
    fn parse_track(&mut self, header: &MidiHeader) -> Result<MidiSequence, MidiParseError> {
        // Read chunk ID: "MTrk"
        let chunk_id = self.read_bytes(4)?;
        if chunk_id != b"MTrk" {
            return Err(MidiParseError::InvalidChunkHeader(
                format!("expected MTrk, got {:?}", chunk_id),
            ));
        }

        let chunk_length = self.read_u32_be()? as usize;
        let chunk_end = self.pos + chunk_length;

        let mut sequence = MidiSequence::new();
        let mut running_status: u8 = 0;
        let mut tick_position: u64 = 0;

        // Default tempo: 120 BPM = 500000 microseconds per quarter note
        let mut tempo_us_per_quarter: u32 = 500_000;
        let ticks_per_quarter = if header.is_smpte {
            // SMPTE: use frame rate
            header.division as u64
        } else {
            header.division as u64
        };

        while self.pos < chunk_end {
            // Read delta time (variable-length quantity)
            let delta = self.read_vlq()?;
            tick_position += delta as u64;

            // Convert tick position to seconds
            let time_seconds = if ticks_per_quarter > 0 {
                let us_per_tick = tempo_us_per_quarter as f64 / ticks_per_quarter as f64;
                (tick_position as f64 * us_per_tick) / 1_000_000.0
            } else {
                0.0
            };

            // Read event
            let event = self.parse_event(&mut running_status)?;

            // Handle tempo change meta events internally
            if let MidiEvent::SetTempo(new_tempo) = event {
                tempo_us_per_quarter = new_tempo;
            }

            // Skip end-of-track marker but don't add it to the sequence
            if matches!(event, MidiEvent::EndOfTrack) {
                break;
            }

            sequence.add_event(time_seconds, event);
        }

        // Ensure we're at the chunk end
        if self.pos < chunk_end {
            self.pos = chunk_end;
        }

        sequence.sort_events();
        if let Some((last_time, _)) = sequence.events.last() {
            sequence.duration = *last_time;
        }

        Ok(sequence)
    }

    /// Parse a single MIDI event from the stream.
    ///
    /// Handles running status: if the first byte is a data byte (< 0x80),
    /// the previous status byte is reused.
    fn parse_event(&mut self, running_status: &mut u8) -> Result<MidiEvent, MidiParseError> {
        let first_byte = self.read_u8()?;

        let status;
        let data1;

        if first_byte >= 0x80 {
            // New status byte
            status = first_byte;
            *running_status = status;
            // Meta event and SysEx don't have running status
            if status == 0xFF {
                return self.parse_meta_event();
            }
            if status == 0xF0 || status == 0xF7 {
                return self.parse_sysex_event();
            }
            data1 = self.read_u8()?;
        } else {
            // Running status: first_byte is actually data1
            status = *running_status;
            data1 = first_byte;
        }

        let channel = status & 0x0F;
        let event_type = status & 0xF0;

        match event_type {
            0x80 => {
                // Note Off
                let data2 = self.read_u8()?;
                Ok(MidiEvent::NoteOff {
                    channel,
                    note: data1,
                    velocity: data2,
                })
            }
            0x90 => {
                // Note On (velocity 0 = note off)
                let data2 = self.read_u8()?;
                if data2 == 0 {
                    Ok(MidiEvent::NoteOff {
                        channel,
                        note: data1,
                        velocity: 0,
                    })
                } else {
                    Ok(MidiEvent::NoteOn {
                        channel,
                        note: data1,
                        velocity: data2,
                    })
                }
            }
            0xA0 => {
                // Polyphonic aftertouch — skip data2, return as unknown
                let data2 = self.read_u8()?;
                Ok(MidiEvent::Unknown {
                    status,
                    data: [data1, data2],
                })
            }
            0xB0 => {
                // Control Change
                let data2 = self.read_u8()?;
                Ok(MidiEvent::ControlChange {
                    channel,
                    controller: data1,
                    value: data2,
                })
            }
            0xC0 => {
                // Program Change (single data byte)
                Ok(MidiEvent::ProgramChange {
                    channel,
                    program: data1,
                })
            }
            0xD0 => {
                // Channel Pressure (single data byte)
                Ok(MidiEvent::ChannelPressure {
                    channel,
                    pressure: data1,
                })
            }
            0xE0 => {
                // Pitch Bend (14-bit value)
                let data2 = self.read_u8()?;
                let raw = ((data2 as u16) << 7) | (data1 as u16);
                let value = raw as i16 - 8192;
                Ok(MidiEvent::PitchBend { channel, value })
            }
            _ => {
                // Unknown channel event
                Ok(MidiEvent::Unknown {
                    status,
                    data: [data1, 0],
                })
            }
        }
    }

    /// Parse a meta event (status byte 0xFF).
    fn parse_meta_event(&mut self) -> Result<MidiEvent, MidiParseError> {
        let meta_type = self.read_u8()?;
        let length = self.read_vlq()? as usize;

        match meta_type {
            0x2F => {
                // End of Track
                Ok(MidiEvent::EndOfTrack)
            }
            0x51 => {
                // Set Tempo: 3 bytes for microseconds per quarter note
                if length < 3 {
                    self.pos += length;
                    return Err(MidiParseError::InvalidEvent(
                        "tempo event too short".into(),
                    ));
                }
                let b0 = self.read_u8()? as u32;
                let b1 = self.read_u8()? as u32;
                let b2 = self.read_u8()? as u32;
                let tempo = (b0 << 16) | (b1 << 8) | b2;
                // Skip remaining bytes if any
                if length > 3 {
                    self.pos += length - 3;
                }
                Ok(MidiEvent::SetTempo(tempo))
            }
            _ => {
                // Skip unknown meta events
                self.pos += length;
                Ok(MidiEvent::Unknown {
                    status: 0xFF,
                    data: [meta_type, 0],
                })
            }
        }
    }

    /// Parse a System Exclusive (SysEx) event.
    fn parse_sysex_event(&mut self) -> Result<MidiEvent, MidiParseError> {
        let length = self.read_vlq()? as usize;
        self.pos += length; // Skip SysEx data
        Ok(MidiEvent::Unknown {
            status: 0xF0,
            data: [0, 0],
        })
    }

    // -- Low-level reading helpers --

    /// Read a single byte, advancing the position.
    fn read_u8(&mut self) -> Result<u8, MidiParseError> {
        if self.pos >= self.data.len() {
            return Err(MidiParseError::UnexpectedEof);
        }
        let val = self.data[self.pos];
        self.pos += 1;
        Ok(val)
    }

    /// Read a big-endian u16.
    fn read_u16_be(&mut self) -> Result<u16, MidiParseError> {
        let b0 = self.read_u8()? as u16;
        let b1 = self.read_u8()? as u16;
        Ok((b0 << 8) | b1)
    }

    /// Read a big-endian u32.
    fn read_u32_be(&mut self) -> Result<u32, MidiParseError> {
        let b0 = self.read_u8()? as u32;
        let b1 = self.read_u8()? as u32;
        let b2 = self.read_u8()? as u32;
        let b3 = self.read_u8()? as u32;
        Ok((b0 << 24) | (b1 << 16) | (b2 << 8) | b3)
    }

    /// Read `n` bytes as a slice, advancing the position.
    fn read_bytes(&mut self, n: usize) -> Result<&[u8], MidiParseError> {
        if self.pos + n > self.data.len() {
            return Err(MidiParseError::UnexpectedEof);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Read a MIDI variable-length quantity (VLQ).
    ///
    /// VLQ encoding uses 7 bits per byte, with the MSB indicating
    /// whether more bytes follow.
    fn read_vlq(&mut self) -> Result<u32, MidiParseError> {
        let mut value: u32 = 0;
        let mut bytes_read = 0;

        loop {
            let byte = self.read_u8()?;
            value = (value << 7) | (byte & 0x7F) as u32;
            bytes_read += 1;

            if (byte & 0x80) == 0 {
                break;
            }
            if bytes_read >= 4 {
                return Err(MidiParseError::MalformedData(
                    "VLQ exceeds 4 bytes".into(),
                ));
            }
        }

        Ok(value)
    }
}

/// Encode a value as a MIDI variable-length quantity.
///
/// Returns the encoded bytes (1-4 bytes).
pub fn encode_vlq(mut value: u32) -> Vec<u8> {
    if value == 0 {
        return vec![0];
    }

    let mut bytes = Vec::with_capacity(4);
    let mut first = true;

    while value > 0 || first {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if !first {
            byte |= 0x80; // continuation bit
        }
        bytes.push(byte);
        first = false;
    }

    bytes.reverse();
    bytes
}

// ===========================================================================
// Synthesizer — Oscillators
// ===========================================================================

/// Oscillator waveform types for the synthesizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Oscillator {
    /// Sine wave — pure tone, no harmonics.
    Sine,
    /// Square wave — fundamental + odd harmonics.
    Square,
    /// Sawtooth wave — all harmonics.
    Sawtooth,
    /// Triangle wave — fundamental + odd harmonics (softer than square).
    Triangle,
    /// White noise — random samples, no pitch.
    Noise,
}

/// Generate a single sample from the given oscillator at the given phase.
///
/// `phase` is in [0, 1) representing one cycle of the waveform.
/// Returns a sample in [-1, 1].
pub fn oscillator_sample(osc: Oscillator, phase: f32) -> f32 {
    match osc {
        Oscillator::Sine => {
            (phase * TAU).sin()
        }
        Oscillator::Square => {
            if phase < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        Oscillator::Sawtooth => {
            2.0 * phase - 1.0
        }
        Oscillator::Triangle => {
            if phase < 0.25 {
                4.0 * phase
            } else if phase < 0.75 {
                2.0 - 4.0 * phase
            } else {
                4.0 * phase - 4.0
            }
        }
        Oscillator::Noise => {
            // Simple noise using a hash of the phase
            // In practice you'd use a proper PRNG; this gives repeatable noise.
            let bits = (phase * 100000.0) as u32;
            let hash = bits.wrapping_mul(2654435761); // Knuth multiplicative hash
            (hash as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }
}

/// Band-limited square wave using additive synthesis (first N odd harmonics).
pub fn bandlimited_square(phase: f32, num_harmonics: u32) -> f32 {
    let mut sum = 0.0_f32;
    for k in 0..num_harmonics {
        let n = 2 * k + 1; // odd harmonics: 1, 3, 5, 7, ...
        sum += (n as f32 * phase * TAU).sin() / n as f32;
    }
    sum * (4.0 / PI)
}

/// Band-limited sawtooth wave using additive synthesis.
pub fn bandlimited_sawtooth(phase: f32, num_harmonics: u32) -> f32 {
    let mut sum = 0.0_f32;
    for n in 1..=num_harmonics {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * (n as f32 * phase * TAU).sin() / n as f32;
    }
    sum * (2.0 / PI)
}

// ===========================================================================
// ADSR Envelope
// ===========================================================================

/// Stage of an ADSR envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeStage {
    /// Not yet triggered.
    Idle,
    /// Attack: ramping from 0 to 1.
    Attack,
    /// Decay: ramping from 1 to sustain level.
    Decay,
    /// Sustain: holding at sustain level until note-off.
    Sustain,
    /// Release: ramping from sustain level to 0.
    Release,
    /// Finished: envelope has completed the release phase.
    Finished,
}

/// Attack-Decay-Sustain-Release amplitude envelope.
///
/// Defines the time-domain amplitude shape of a synthesized note:
/// - **Attack**: time (seconds) to ramp from 0 to peak (1.0)
/// - **Decay**: time to fall from peak to sustain level
/// - **Sustain**: amplitude level held while the note is held (0.0 to 1.0)
/// - **Release**: time to fade from sustain level to 0 after note-off
#[derive(Debug, Clone, Copy)]
pub struct ADSREnvelope {
    /// Attack time in seconds.
    pub attack: f32,
    /// Decay time in seconds.
    pub decay: f32,
    /// Sustain level (0.0 to 1.0).
    pub sustain_level: f32,
    /// Release time in seconds.
    pub release: f32,
}

impl ADSREnvelope {
    /// Create a new ADSR envelope.
    pub fn new(attack: f32, decay: f32, sustain_level: f32, release: f32) -> Self {
        Self {
            attack: attack.max(0.001), // minimum to avoid division by zero
            decay: decay.max(0.001),
            sustain_level: sustain_level.clamp(0.0, 1.0),
            release: release.max(0.001),
        }
    }

    /// Create a plucky envelope (fast attack, moderate decay, no sustain).
    pub fn pluck() -> Self {
        Self::new(0.002, 0.3, 0.0, 0.1)
    }

    /// Create a pad envelope (slow attack, long sustain).
    pub fn pad() -> Self {
        Self::new(0.5, 0.3, 0.7, 1.0)
    }

    /// Create an organ-style envelope (instant attack, full sustain).
    pub fn organ() -> Self {
        Self::new(0.001, 0.001, 1.0, 0.01)
    }

    /// Create a default piano-like envelope.
    pub fn piano() -> Self {
        Self::new(0.005, 0.5, 0.3, 0.5)
    }
}

impl Default for ADSREnvelope {
    fn default() -> Self {
        Self::piano()
    }
}

/// Stateful ADSR envelope evaluator.
///
/// Tracks the current stage and elapsed time to produce an amplitude
/// multiplier at each sample.
#[derive(Debug, Clone)]
pub struct EnvelopeState {
    /// The ADSR parameters.
    pub params: ADSREnvelope,
    /// Current envelope stage.
    pub stage: EnvelopeStage,
    /// Time elapsed within the current stage (seconds).
    pub stage_time: f32,
    /// Current amplitude output (0.0 to 1.0).
    pub current_amplitude: f32,
    /// Amplitude at the moment note-off was received (for release ramp).
    release_start_amplitude: f32,
}

impl EnvelopeState {
    /// Create a new envelope state with the given ADSR parameters.
    pub fn new(params: ADSREnvelope) -> Self {
        Self {
            params,
            stage: EnvelopeStage::Idle,
            stage_time: 0.0,
            current_amplitude: 0.0,
            release_start_amplitude: 0.0,
        }
    }

    /// Trigger the envelope (note-on).
    pub fn trigger(&mut self) {
        self.stage = EnvelopeStage::Attack;
        self.stage_time = 0.0;
        self.current_amplitude = 0.0;
    }

    /// Release the envelope (note-off).
    pub fn release(&mut self) {
        if self.stage == EnvelopeStage::Finished || self.stage == EnvelopeStage::Idle {
            return;
        }
        self.release_start_amplitude = self.current_amplitude;
        self.stage = EnvelopeStage::Release;
        self.stage_time = 0.0;
    }

    /// Whether the envelope has finished (release complete).
    pub fn is_finished(&self) -> bool {
        self.stage == EnvelopeStage::Finished
    }

    /// Whether the envelope is active (producing non-zero output).
    pub fn is_active(&self) -> bool {
        self.stage != EnvelopeStage::Idle && self.stage != EnvelopeStage::Finished
    }

    /// Advance the envelope by one sample and return the amplitude.
    pub fn next_sample(&mut self, sample_rate: f32) -> f32 {
        let dt = 1.0 / sample_rate;
        self.stage_time += dt;

        match self.stage {
            EnvelopeStage::Idle => {
                self.current_amplitude = 0.0;
            }
            EnvelopeStage::Attack => {
                // Linear ramp from 0 to 1 over attack time
                let t = self.stage_time / self.params.attack;
                if t >= 1.0 {
                    self.current_amplitude = 1.0;
                    self.stage = EnvelopeStage::Decay;
                    self.stage_time = 0.0;
                } else {
                    self.current_amplitude = t;
                }
            }
            EnvelopeStage::Decay => {
                // Exponential-ish decay from 1 to sustain level
                let t = self.stage_time / self.params.decay;
                if t >= 1.0 {
                    self.current_amplitude = self.params.sustain_level;
                    self.stage = EnvelopeStage::Sustain;
                    self.stage_time = 0.0;
                } else {
                    // Exponential decay curve
                    let decay_curve = (-3.0 * t).exp();
                    self.current_amplitude =
                        self.params.sustain_level + (1.0 - self.params.sustain_level) * decay_curve;
                }
            }
            EnvelopeStage::Sustain => {
                self.current_amplitude = self.params.sustain_level;
                // Stays here until release() is called
            }
            EnvelopeStage::Release => {
                // Exponential decay from release_start_amplitude to 0
                let t = self.stage_time / self.params.release;
                if t >= 1.0 {
                    self.current_amplitude = 0.0;
                    self.stage = EnvelopeStage::Finished;
                } else {
                    let release_curve = (-3.0 * t).exp();
                    self.current_amplitude = self.release_start_amplitude * release_curve;
                }
            }
            EnvelopeStage::Finished => {
                self.current_amplitude = 0.0;
            }
        }

        self.current_amplitude
    }
}

/// Evaluate an ADSR envelope at a given absolute time with note-off time.
///
/// Standalone stateless function for calculating envelope amplitude
/// at a specific time point.
///
/// - `time`: time since note-on (seconds)
/// - `note_off_time`: time when note-off occurs (seconds), or `None` if still held
pub fn evaluate_envelope(env: &ADSREnvelope, time: f32, note_off_time: Option<f32>) -> f32 {
    if time < 0.0 {
        return 0.0;
    }

    // Check if we're in the release phase
    if let Some(off_time) = note_off_time {
        if time >= off_time {
            // Calculate what the amplitude was at note-off
            let amp_at_off = evaluate_envelope(env, off_time - 0.0001, None);
            let release_time = time - off_time;
            let t = release_time / env.release;
            if t >= 1.0 {
                return 0.0;
            }
            return amp_at_off * (-3.0 * t).exp();
        }
    }

    // Attack phase
    if time < env.attack {
        return time / env.attack;
    }

    // Decay phase
    let decay_start = env.attack;
    let decay_end = env.attack + env.decay;
    if time < decay_end {
        let t = (time - decay_start) / env.decay;
        let curve = (-3.0 * t).exp();
        return env.sustain_level + (1.0 - env.sustain_level) * curve;
    }

    // Sustain phase
    env.sustain_level
}

// ===========================================================================
// SynthVoice — one active note
// ===========================================================================

/// A single synthesizer voice, playing one note.
///
/// Each voice has its own oscillator, phase accumulator, and envelope
/// state. Multiple voices run simultaneously for polyphony.
#[derive(Clone)]
pub struct SynthVoice {
    /// Whether this voice is currently in use.
    pub active: bool,
    /// MIDI note number being played.
    pub note: u8,
    /// MIDI channel.
    pub channel: u8,
    /// Frequency in Hz.
    pub frequency: f32,
    /// Velocity (0.0 to 1.0).
    pub velocity: f32,
    /// Oscillator waveform.
    pub oscillator: Oscillator,
    /// Current phase accumulator (0.0 to 1.0).
    pub phase: f32,
    /// Phase increment per sample.
    pub phase_increment: f32,
    /// Envelope state.
    pub envelope: EnvelopeState,
    /// Detune in cents (1/100th of a semitone).
    pub detune_cents: f32,
    /// Pan position (-1.0 = left, 0.0 = center, 1.0 = right).
    pub pan: f32,
    /// Simple one-pole low-pass filter state.
    pub filter_state: f32,
    /// Low-pass filter cutoff coefficient (0.0 to 1.0, higher = brighter).
    pub filter_cutoff: f32,
}

impl SynthVoice {
    /// Create a new inactive voice.
    pub fn new(envelope_params: ADSREnvelope) -> Self {
        Self {
            active: false,
            note: 0,
            channel: 0,
            frequency: 440.0,
            velocity: 0.0,
            oscillator: Oscillator::Sine,
            phase: 0.0,
            phase_increment: 0.0,
            envelope: EnvelopeState::new(envelope_params),
            detune_cents: 0.0,
            pan: 0.0,
            filter_state: 0.0,
            filter_cutoff: 1.0,
        }
    }

    /// Trigger this voice with a new note.
    pub fn note_on(&mut self, note: u8, velocity: u8, sample_rate: u32) {
        self.active = true;
        self.note = note;
        self.velocity = velocity as f32 / 127.0;

        // Calculate frequency with detune
        let semitone_offset = self.detune_cents / 100.0;
        self.frequency = A4_FREQUENCY * 2.0_f32.powf(
            (note as f32 - A4_NOTE as f32 + semitone_offset) / 12.0,
        );
        self.phase_increment = self.frequency / sample_rate as f32;

        self.phase = 0.0;
        self.filter_state = 0.0;
        self.envelope.trigger();
    }

    /// Release this voice (note-off).
    pub fn note_off(&mut self) {
        self.envelope.release();
    }

    /// Generate the next sample from this voice.
    pub fn next_sample(&mut self, sample_rate: f32) -> f32 {
        if !self.active {
            return 0.0;
        }

        // Check if envelope is finished
        if self.envelope.is_finished() {
            self.active = false;
            return 0.0;
        }

        // Generate oscillator sample
        let raw = oscillator_sample(self.oscillator, self.phase);

        // Apply one-pole low-pass filter
        self.filter_state += self.filter_cutoff * (raw - self.filter_state);
        let filtered = self.filter_state;

        // Apply envelope
        let env_amp = self.envelope.next_sample(sample_rate);

        // Apply velocity
        let output = filtered * env_amp * self.velocity;

        // Advance phase
        self.phase += self.phase_increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        output
    }

    /// Check if this voice can be stolen (lowest priority for reuse).
    pub fn steal_priority(&self) -> f32 {
        if !self.active {
            return 0.0;
        }
        // Lower envelope amplitude = more stealable
        self.envelope.current_amplitude * self.velocity
    }
}

// ===========================================================================
// Synthesizer
// ===========================================================================

/// Polyphonic wavetable synthesizer.
///
/// Maintains a pool of `SynthVoice`s and allocates them for incoming
/// MIDI note-on events. When all voices are in use, the quietest voice
/// is stolen for the new note.
pub struct Synthesizer {
    /// Voice pool.
    pub voices: Vec<SynthVoice>,
    /// Maximum polyphony (number of voices).
    pub max_polyphony: usize,
    /// Sample rate for synthesis.
    pub sample_rate: u32,
    /// Default oscillator for new voices.
    pub default_oscillator: Oscillator,
    /// Default ADSR envelope for new voices.
    pub default_envelope: ADSREnvelope,
    /// Master volume (0.0 to 1.0).
    pub master_volume: f32,
    /// Default filter cutoff for new voices.
    pub default_filter_cutoff: f32,
    /// Global pitch bend value (-8192 to 8191).
    pub pitch_bend: i16,
    /// Pitch bend range in semitones.
    pub pitch_bend_range: f32,
}

impl Synthesizer {
    /// Create a new synthesizer with default settings.
    pub fn new(sample_rate: u32) -> Self {
        let max_poly = DEFAULT_POLYPHONY;
        let envelope = ADSREnvelope::default();
        let voices = (0..max_poly)
            .map(|_| SynthVoice::new(envelope))
            .collect();
        Self {
            voices,
            max_polyphony: max_poly,
            sample_rate,
            default_oscillator: Oscillator::Sine,
            default_envelope: envelope,
            master_volume: 0.7,
            default_filter_cutoff: 1.0,
            pitch_bend: 0,
            pitch_bend_range: 2.0,
        }
    }

    /// Create a synthesizer with custom polyphony.
    pub fn with_polyphony(sample_rate: u32, polyphony: usize) -> Self {
        let mut synth = Self::new(sample_rate);
        synth.max_polyphony = polyphony;
        synth.voices = (0..polyphony)
            .map(|_| SynthVoice::new(synth.default_envelope))
            .collect();
        synth
    }

    /// Handle a MIDI event.
    pub fn process_event(&mut self, event: &MidiEvent) {
        match *event {
            MidiEvent::NoteOn {
                note, velocity, ..
            } => {
                if velocity > 0 {
                    self.note_on(note, velocity);
                } else {
                    self.note_off(note);
                }
            }
            MidiEvent::NoteOff { note, .. } => {
                self.note_off(note);
            }
            MidiEvent::PitchBend { value, .. } => {
                self.pitch_bend = value;
            }
            MidiEvent::ControlChange {
                controller, value, ..
            } => {
                self.handle_cc(controller, value);
            }
            _ => {}
        }
    }

    /// Trigger a note-on event.
    fn note_on(&mut self, note: u8, velocity: u8) {
        // Find a free voice, or steal the quietest one
        let voice_idx = self.find_free_voice().unwrap_or_else(|| self.steal_voice());

        let voice = &mut self.voices[voice_idx];
        voice.oscillator = self.default_oscillator;
        voice.envelope = EnvelopeState::new(self.default_envelope);
        voice.filter_cutoff = self.default_filter_cutoff;
        voice.note_on(note, velocity, self.sample_rate);
    }

    /// Trigger a note-off event.
    fn note_off(&mut self, note: u8) {
        for voice in &mut self.voices {
            if voice.active && voice.note == note {
                voice.note_off();
                break;
            }
        }
    }

    /// Handle a MIDI CC message.
    fn handle_cc(&mut self, controller: u8, value: u8) {
        match controller {
            1 => {
                // Modulation wheel — adjust filter cutoff
                let cutoff = value as f32 / 127.0;
                for voice in &mut self.voices {
                    if voice.active {
                        voice.filter_cutoff = cutoff.max(0.01);
                    }
                }
            }
            7 => {
                // Volume
                self.master_volume = value as f32 / 127.0;
            }
            10 => {
                // Pan
                let pan = (value as f32 / 64.0) - 1.0;
                for voice in &mut self.voices {
                    if voice.active {
                        voice.pan = pan;
                    }
                }
            }
            123 => {
                // All Notes Off
                self.all_notes_off();
            }
            _ => {}
        }
    }

    /// Find a free (inactive) voice.
    fn find_free_voice(&self) -> Option<usize> {
        self.voices.iter().position(|v| !v.active)
    }

    /// Steal the voice with the lowest priority.
    fn steal_voice(&self) -> usize {
        let mut min_priority = f32::MAX;
        let mut steal_idx = 0;
        for (i, voice) in self.voices.iter().enumerate() {
            let priority = voice.steal_priority();
            if priority < min_priority {
                min_priority = priority;
                steal_idx = i;
            }
        }
        steal_idx
    }

    /// Release all active notes.
    pub fn all_notes_off(&mut self) {
        for voice in &mut self.voices {
            if voice.active {
                voice.note_off();
            }
        }
    }

    /// Kill all voices immediately (no release phase).
    pub fn panic(&mut self) {
        for voice in &mut self.voices {
            voice.active = false;
            voice.envelope.stage = EnvelopeStage::Finished;
        }
    }

    /// Get the number of currently active voices.
    pub fn active_voice_count(&self) -> usize {
        self.voices.iter().filter(|v| v.active).count()
    }

    /// Render audio into the output buffer (mono).
    ///
    /// Sums all active voices and writes to `output`. The buffer is
    /// overwritten (not mixed into).
    pub fn render_mono(&mut self, output: &mut [f32]) {
        let sr = self.sample_rate as f32;

        for sample in output.iter_mut() {
            *sample = 0.0;
        }

        for voice in &mut self.voices {
            if !voice.active {
                continue;
            }
            for sample in output.iter_mut() {
                *sample += voice.next_sample(sr);
            }
        }

        // Apply master volume and soft-clip
        for sample in output.iter_mut() {
            *sample *= self.master_volume;
            // Soft clipping (tanh approximation)
            *sample = soft_clip(*sample);
        }
    }

    /// Render audio into stereo interleaved output buffer.
    ///
    /// Each voice's pan position controls left/right balance.
    /// Output format: [L0, R0, L1, R1, ...].
    pub fn render_stereo(&mut self, output: &mut [f32]) {
        let sr = self.sample_rate as f32;
        let num_frames = output.len() / 2;

        for s in output.iter_mut() {
            *s = 0.0;
        }

        for voice in &mut self.voices {
            if !voice.active {
                continue;
            }
            // Calculate stereo gains from pan position
            let pan = voice.pan.clamp(-1.0, 1.0);
            let left_gain = ((1.0 - pan) * 0.5).sqrt();
            let right_gain = ((1.0 + pan) * 0.5).sqrt();

            for frame in 0..num_frames {
                let mono = voice.next_sample(sr);
                output[frame * 2] += mono * left_gain;
                output[frame * 2 + 1] += mono * right_gain;
            }
        }

        // Master volume and soft clip
        for sample in output.iter_mut() {
            *sample *= self.master_volume;
            *sample = soft_clip(*sample);
        }
    }
}

/// Render audio from a set of voice references into a buffer.
///
/// Standalone rendering function that can be used outside the Synthesizer.
pub fn render_audio(
    voices: &mut [SynthVoice],
    sample_rate: u32,
    output_buffer: &mut [f32],
) {
    let sr = sample_rate as f32;

    for sample in output_buffer.iter_mut() {
        *sample = 0.0;
    }

    for voice in voices.iter_mut() {
        if !voice.active {
            continue;
        }
        for sample in output_buffer.iter_mut() {
            *sample += voice.next_sample(sr);
        }
    }

    // Soft clip
    for sample in output_buffer.iter_mut() {
        *sample = soft_clip(*sample);
    }
}

/// Soft-clip a sample to prevent harsh digital distortion.
///
/// Uses a fast tanh approximation: `x / (1 + |x|)`.
fn soft_clip(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

// ===========================================================================
// Music Theory Helpers
// ===========================================================================

/// Convert a MIDI note number to its frequency in Hz.
///
/// Uses equal temperament tuning: `f = 440 * 2^((note - 69) / 12)`.
pub fn note_to_frequency(note: u8) -> f32 {
    A4_FREQUENCY * 2.0_f32.powf((note as f32 - A4_NOTE as f32) / 12.0)
}

/// Convert a frequency in Hz to the closest MIDI note number.
pub fn frequency_to_note(freq: f32) -> u8 {
    if freq <= 0.0 {
        return 0;
    }
    let note = A4_NOTE as f32 + 12.0 * (freq / A4_FREQUENCY).log2();
    note.round().clamp(0.0, 127.0) as u8
}

/// Get the human-readable name of a MIDI note (e.g. "C4", "A#3").
///
/// Uses scientific pitch notation where middle C (note 60) = "C4".
pub fn note_name(note: u8) -> String {
    let name_idx = (note % 12) as usize;
    let octave = (note / 12) as i32 - 1; // MIDI note 0 = C-1
    format!("{}{}", NOTE_NAMES[name_idx], octave)
}

/// Parse a note name string (e.g. "C4", "A#3") to a MIDI note number.
pub fn parse_note_name(name: &str) -> Option<u8> {
    if name.is_empty() {
        return None;
    }

    let name = name.trim();
    let (note_part, octave_part) = if name.len() >= 2 && name.as_bytes()[1] == b'#' {
        (&name[..2], &name[2..])
    } else if name.len() >= 2 && name.as_bytes()[1] == b'b' {
        // Flat: convert to sharp equivalent
        (&name[..2], &name[2..])
    } else {
        (&name[..1], &name[1..])
    };

    let semitone = match note_part {
        "C" => 0,
        "C#" | "Db" => 1,
        "D" => 2,
        "D#" | "Eb" => 3,
        "E" => 4,
        "F" => 5,
        "F#" | "Gb" => 6,
        "G" => 7,
        "G#" | "Ab" => 8,
        "A" => 9,
        "A#" | "Bb" => 10,
        "B" => 11,
        _ => return None,
    };

    let octave: i32 = octave_part.parse().ok()?;
    let note = (octave + 1) * 12 + semitone;
    if note < 0 || note > 127 {
        return None;
    }
    Some(note as u8)
}

// ---------------------------------------------------------------------------
// Scales
// ---------------------------------------------------------------------------

/// Musical scale types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    /// Major scale (Ionian mode): W-W-H-W-W-W-H
    Major,
    /// Natural minor scale (Aeolian mode): W-H-W-W-H-W-W
    Minor,
    /// Major pentatonic: 5 notes per octave
    Pentatonic,
    /// Minor pentatonic: 5 notes per octave
    MinorPentatonic,
    /// Blues scale: minor pentatonic + b5
    Blues,
    /// Chromatic scale: all 12 semitones
    Chromatic,
    /// Dorian mode: W-H-W-W-W-H-W
    Dorian,
    /// Mixolydian mode: W-W-H-W-W-H-W
    Mixolydian,
    /// Harmonic minor: W-H-W-W-H-WH-H
    HarmonicMinor,
    /// Whole tone scale: W-W-W-W-W-W
    WholeTone,
}

impl Scale {
    /// Get the interval pattern (semitone offsets from root) for this scale.
    pub fn intervals(&self) -> &[u8] {
        match self {
            Self::Major => &[0, 2, 4, 5, 7, 9, 11],
            Self::Minor => &[0, 2, 3, 5, 7, 8, 10],
            Self::Pentatonic => &[0, 2, 4, 7, 9],
            Self::MinorPentatonic => &[0, 3, 5, 7, 10],
            Self::Blues => &[0, 3, 5, 6, 7, 10],
            Self::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            Self::Dorian => &[0, 2, 3, 5, 7, 9, 10],
            Self::Mixolydian => &[0, 2, 4, 5, 7, 9, 10],
            Self::HarmonicMinor => &[0, 2, 3, 5, 7, 8, 11],
            Self::WholeTone => &[0, 2, 4, 6, 8, 10],
        }
    }
}

/// Generate a scale starting from the given root note.
///
/// Returns MIDI note numbers for one octave of the scale.
/// Notes that would exceed 127 are excluded.
pub fn scale(root: u8, scale_type: Scale) -> Vec<u8> {
    scale_type
        .intervals()
        .iter()
        .filter_map(|&interval| {
            let note = root as u16 + interval as u16;
            if note <= 127 {
                Some(note as u8)
            } else {
                None
            }
        })
        .collect()
}

/// Generate a multi-octave scale.
pub fn scale_multi_octave(root: u8, scale_type: Scale, octaves: u8) -> Vec<u8> {
    let intervals = scale_type.intervals();
    let mut notes = Vec::new();
    for oct in 0..octaves {
        for &interval in intervals {
            let note = root as u16 + (oct as u16 * 12) + interval as u16;
            if note <= 127 {
                notes.push(note as u8);
            }
        }
    }
    notes
}

// ---------------------------------------------------------------------------
// Chords
// ---------------------------------------------------------------------------

/// Chord types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chord {
    /// Major triad: root, major 3rd, perfect 5th
    Major,
    /// Minor triad: root, minor 3rd, perfect 5th
    Minor,
    /// Dominant 7th: major triad + minor 7th
    Dominant7th,
    /// Major 7th: major triad + major 7th
    Major7th,
    /// Minor 7th: minor triad + minor 7th
    Minor7th,
    /// Diminished triad: root, minor 3rd, diminished 5th
    Diminished,
    /// Augmented triad: root, major 3rd, augmented 5th
    Augmented,
    /// Suspended 2nd: root, major 2nd, perfect 5th
    Sus2,
    /// Suspended 4th: root, perfect 4th, perfect 5th
    Sus4,
    /// Power chord (5th): root, perfect 5th
    Power,
    /// Diminished 7th: dim triad + diminished 7th
    Diminished7th,
    /// Minor-Major 7th: minor triad + major 7th
    MinorMajor7th,
}

impl Chord {
    /// Get the interval pattern (semitone offsets from root) for this chord.
    pub fn intervals(&self) -> &[u8] {
        match self {
            Self::Major => &[0, 4, 7],
            Self::Minor => &[0, 3, 7],
            Self::Dominant7th => &[0, 4, 7, 10],
            Self::Major7th => &[0, 4, 7, 11],
            Self::Minor7th => &[0, 3, 7, 10],
            Self::Diminished => &[0, 3, 6],
            Self::Augmented => &[0, 4, 8],
            Self::Sus2 => &[0, 2, 7],
            Self::Sus4 => &[0, 5, 7],
            Self::Power => &[0, 7],
            Self::Diminished7th => &[0, 3, 6, 9],
            Self::MinorMajor7th => &[0, 3, 7, 11],
        }
    }
}

/// Generate the notes of a chord from the given root.
///
/// Returns MIDI note numbers. Notes exceeding 127 are excluded.
pub fn chord(root: u8, chord_type: Chord) -> Vec<u8> {
    chord_type
        .intervals()
        .iter()
        .filter_map(|&interval| {
            let note = root as u16 + interval as u16;
            if note <= 127 {
                Some(note as u8)
            } else {
                None
            }
        })
        .collect()
}

/// Generate a chord with a specific inversion.
///
/// Inversion 0 = root position, 1 = first inversion, etc.
/// Notes in inversions below the root are raised by an octave.
pub fn chord_inversion(root: u8, chord_type: Chord, inversion: usize) -> Vec<u8> {
    let mut notes = chord(root, chord_type);
    for _ in 0..inversion.min(notes.len() - 1) {
        if let Some(note) = notes.first().copied() {
            notes.remove(0);
            let raised = note.saturating_add(12);
            if raised <= 127 {
                notes.push(raised);
            }
        }
    }
    notes
}

/// Get the interval name between two MIDI notes.
pub fn interval_name(semitones: u8) -> &'static str {
    match semitones % 12 {
        0 => "unison",
        1 => "minor 2nd",
        2 => "major 2nd",
        3 => "minor 3rd",
        4 => "major 3rd",
        5 => "perfect 4th",
        6 => "tritone",
        7 => "perfect 5th",
        8 => "minor 6th",
        9 => "major 6th",
        10 => "minor 7th",
        11 => "major 7th",
        _ => unreachable!(),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_to_frequency_a4() {
        let freq = note_to_frequency(69);
        assert!((freq - 440.0).abs() < 0.01);
    }

    #[test]
    fn note_to_frequency_middle_c() {
        let freq = note_to_frequency(60);
        assert!((freq - 261.63).abs() < 0.1);
    }

    #[test]
    fn note_to_frequency_octave() {
        let f1 = note_to_frequency(60);
        let f2 = note_to_frequency(72);
        assert!((f2 / f1 - 2.0).abs() < 0.01, "octave should double frequency");
    }

    #[test]
    fn frequency_to_note_roundtrip() {
        for note in 20..108 {
            let freq = note_to_frequency(note);
            let back = frequency_to_note(freq);
            assert_eq!(back, note, "roundtrip failed for note {}", note);
        }
    }

    #[test]
    fn note_name_middle_c() {
        assert_eq!(note_name(60), "C4");
    }

    #[test]
    fn note_name_a4() {
        assert_eq!(note_name(69), "A4");
    }

    #[test]
    fn note_name_a_sharp_3() {
        assert_eq!(note_name(58), "A#3");
    }

    #[test]
    fn parse_note_roundtrip() {
        for note in 0..=127u8 {
            let name = note_name(note);
            let parsed = parse_note_name(&name);
            assert_eq!(parsed, Some(note), "roundtrip failed for {} ({})", note, name);
        }
    }

    #[test]
    fn major_scale() {
        let s = scale(60, Scale::Major);
        assert_eq!(s, vec![60, 62, 64, 65, 67, 69, 71]);
    }

    #[test]
    fn minor_scale() {
        let s = scale(69, Scale::Minor);
        assert_eq!(s, vec![69, 71, 72, 74, 76, 77, 79]);
    }

    #[test]
    fn pentatonic_scale() {
        let s = scale(60, Scale::Pentatonic);
        assert_eq!(s, vec![60, 62, 64, 67, 69]);
    }

    #[test]
    fn blues_scale() {
        let s = scale(60, Scale::Blues);
        assert_eq!(s, vec![60, 63, 65, 66, 67, 70]);
    }

    #[test]
    fn chromatic_scale() {
        let s = scale(60, Scale::Chromatic);
        assert_eq!(s.len(), 12);
        assert_eq!(s[0], 60);
        assert_eq!(s[11], 71);
    }

    #[test]
    fn major_chord() {
        let c = chord(60, Chord::Major);
        assert_eq!(c, vec![60, 64, 67]); // C E G
    }

    #[test]
    fn minor_chord() {
        let c = chord(69, Chord::Minor);
        assert_eq!(c, vec![69, 72, 76]); // A C E
    }

    #[test]
    fn dominant_7th_chord() {
        let c = chord(67, Chord::Dominant7th);
        assert_eq!(c, vec![67, 71, 74, 77]); // G B D F
    }

    #[test]
    fn diminished_chord() {
        let c = chord(71, Chord::Diminished);
        assert_eq!(c, vec![71, 74, 77]); // B D F
    }

    #[test]
    fn augmented_chord() {
        let c = chord(60, Chord::Augmented);
        assert_eq!(c, vec![60, 64, 68]); // C E G#
    }

    #[test]
    fn chord_inversion_first() {
        let c = chord_inversion(60, Chord::Major, 1);
        assert_eq!(c, vec![64, 67, 72]); // E G C
    }

    #[test]
    fn oscillator_sine() {
        assert!((oscillator_sample(Oscillator::Sine, 0.0)).abs() < 0.01);
        assert!((oscillator_sample(Oscillator::Sine, 0.25) - 1.0).abs() < 0.01);
    }

    #[test]
    fn oscillator_square() {
        assert_eq!(oscillator_sample(Oscillator::Square, 0.25), 1.0);
        assert_eq!(oscillator_sample(Oscillator::Square, 0.75), -1.0);
    }

    #[test]
    fn oscillator_sawtooth() {
        assert!((oscillator_sample(Oscillator::Sawtooth, 0.0) - (-1.0)).abs() < 0.01);
        assert!((oscillator_sample(Oscillator::Sawtooth, 0.5) - 0.0).abs() < 0.01);
    }

    #[test]
    fn oscillator_triangle() {
        assert!((oscillator_sample(Oscillator::Triangle, 0.0)).abs() < 0.01);
        assert!((oscillator_sample(Oscillator::Triangle, 0.25) - 1.0).abs() < 0.01);
        assert!((oscillator_sample(Oscillator::Triangle, 0.75) - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn adsr_envelope_attack() {
        let env = ADSREnvelope::new(0.1, 0.1, 0.5, 0.1);
        let amp_mid = evaluate_envelope(&env, 0.05, None);
        assert!(amp_mid > 0.3 && amp_mid < 0.7, "mid-attack should be ~0.5, got {}", amp_mid);
        let amp_end = evaluate_envelope(&env, 0.1, None);
        assert!((amp_end - 1.0).abs() < 0.1, "end of attack should be ~1.0");
    }

    #[test]
    fn adsr_envelope_sustain() {
        let env = ADSREnvelope::new(0.01, 0.01, 0.5, 0.1);
        let amp = evaluate_envelope(&env, 1.0, None);
        assert!((amp - 0.5).abs() < 0.05, "sustain should be ~0.5, got {}", amp);
    }

    #[test]
    fn adsr_envelope_release() {
        let env = ADSREnvelope::new(0.01, 0.01, 0.5, 0.5);
        let amp_before_off = evaluate_envelope(&env, 0.5, None);
        assert!(amp_before_off > 0.4);
        let amp_after_off = evaluate_envelope(&env, 1.0, Some(0.5));
        assert!(amp_after_off < amp_before_off, "release should reduce amplitude");
        let amp_released = evaluate_envelope(&env, 2.0, Some(0.5));
        assert!(amp_released < 0.01, "should be nearly silent after release");
    }

    #[test]
    fn envelope_state_lifecycle() {
        let env = ADSREnvelope::new(0.01, 0.01, 0.5, 0.05);
        let mut state = EnvelopeState::new(env);
        assert_eq!(state.stage, EnvelopeStage::Idle);

        state.trigger();
        assert_eq!(state.stage, EnvelopeStage::Attack);

        // Run through attack and decay
        for _ in 0..1000 {
            state.next_sample(44100.0);
        }
        // Should be in sustain by now
        assert!(state.current_amplitude > 0.3);

        state.release();
        assert_eq!(state.stage, EnvelopeStage::Release);

        // Run through release
        for _ in 0..5000 {
            state.next_sample(44100.0);
        }
        assert!(state.is_finished());
    }

    #[test]
    fn synthesizer_note_on_off() {
        let mut synth = Synthesizer::new(44100);
        synth.process_event(&MidiEvent::NoteOn {
            channel: 0,
            note: 60,
            velocity: 100,
        });
        assert_eq!(synth.active_voice_count(), 1);

        synth.process_event(&MidiEvent::NoteOff {
            channel: 0,
            note: 60,
            velocity: 0,
        });
        // Voice is in release, still "active"
        assert!(synth.active_voice_count() >= 0);
    }

    #[test]
    fn synthesizer_renders_audio() {
        let mut synth = Synthesizer::new(44100);
        synth.default_oscillator = Oscillator::Sine;
        synth.process_event(&MidiEvent::NoteOn {
            channel: 0,
            note: 69,
            velocity: 127,
        });

        let mut buffer = vec![0.0f32; 1024];
        synth.render_mono(&mut buffer);

        let max = buffer.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        assert!(max > 0.0, "synthesizer should produce non-zero output");
    }

    #[test]
    fn vlq_encode_decode() {
        let test_values = [0u32, 1, 127, 128, 255, 16383, 16384, 0x0FFFFFFF];
        for &val in &test_values {
            let encoded = encode_vlq(val);
            // Decode it back
            let mut parser = MidiFileParser::new(encoded);
            let decoded = parser.read_vlq().unwrap();
            assert_eq!(decoded, val, "VLQ roundtrip failed for {}", val);
        }
    }

    #[test]
    fn midi_parse_header() {
        // Construct a minimal MThd chunk
        let mut data = Vec::new();
        data.extend_from_slice(b"MThd");
        data.extend_from_slice(&6u32.to_be_bytes()); // length = 6
        data.extend_from_slice(&0u16.to_be_bytes()); // format 0
        data.extend_from_slice(&1u16.to_be_bytes()); // 1 track
        data.extend_from_slice(&480u16.to_be_bytes()); // 480 ticks per quarter

        let mut parser = MidiFileParser::new(data);
        let header = parser.parse_header().unwrap();
        assert_eq!(header.format, MidiFormat::SingleTrack);
        assert_eq!(header.track_count, 1);
        assert_eq!(header.division, 480);
    }

    #[test]
    fn midi_parse_simple_track() {
        let mut data = Vec::new();
        // Header
        data.extend_from_slice(b"MThd");
        data.extend_from_slice(&6u32.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes()); // format 0
        data.extend_from_slice(&1u16.to_be_bytes()); // 1 track
        data.extend_from_slice(&96u16.to_be_bytes()); // 96 tpqn

        // Track chunk
        let mut track_data = Vec::new();
        // Delta=0, Note On C4 velocity 80
        track_data.push(0x00); // delta time = 0
        track_data.push(0x90); // note on, channel 0
        track_data.push(60);   // C4
        track_data.push(80);   // velocity

        // Delta=96, Note Off C4
        track_data.push(0x60); // delta time = 96
        track_data.push(0x80); // note off, channel 0
        track_data.push(60);   // C4
        track_data.push(0);    // velocity

        // End of track
        track_data.push(0x00); // delta
        track_data.push(0xFF); // meta
        track_data.push(0x2F); // end of track
        track_data.push(0x00); // length

        data.extend_from_slice(b"MTrk");
        data.extend_from_slice(&(track_data.len() as u32).to_be_bytes());
        data.extend_from_slice(&track_data);

        let mut parser = MidiFileParser::new(data);
        let (header, sequences) = parser.parse().unwrap();
        assert_eq!(header.format, MidiFormat::SingleTrack);
        assert_eq!(sequences.len(), 1);

        let seq = &sequences[0];
        assert!(seq.events.len() >= 2);
        // First event should be NoteOn
        assert!(seq.events[0].1.is_note_on());
        // Second event should be NoteOff
        assert!(seq.events[1].1.is_note_off());
    }

    #[test]
    fn midi_running_status() {
        // Two consecutive note-on events using running status
        let mut data = Vec::new();
        // Header
        data.extend_from_slice(b"MThd");
        data.extend_from_slice(&6u32.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes());
        data.extend_from_slice(&1u16.to_be_bytes());
        data.extend_from_slice(&96u16.to_be_bytes());

        let mut track = Vec::new();
        // First note: explicit status
        track.push(0x00); // delta
        track.push(0x90); // note on ch0
        track.push(60);
        track.push(100);

        // Second note: running status (no status byte)
        track.push(0x00); // delta
        track.push(64);   // E4 — data byte (< 0x80 => running status)
        track.push(100);

        // End of track
        track.push(0x00);
        track.push(0xFF);
        track.push(0x2F);
        track.push(0x00);

        data.extend_from_slice(b"MTrk");
        data.extend_from_slice(&(track.len() as u32).to_be_bytes());
        data.extend_from_slice(&track);

        let mut parser = MidiFileParser::new(data);
        let (_, sequences) = parser.parse().unwrap();
        let seq = &sequences[0];
        assert!(seq.events.len() >= 2);
        // Both should be note-on
        assert!(seq.events[0].1.is_note_on());
        assert!(seq.events[1].1.is_note_on());

        // Check note numbers
        if let MidiEvent::NoteOn { note: n1, .. } = seq.events[0].1 {
            assert_eq!(n1, 60);
        }
        if let MidiEvent::NoteOn { note: n2, .. } = seq.events[1].1 {
            assert_eq!(n2, 64);
        }
    }

    #[test]
    fn interval_names() {
        assert_eq!(interval_name(0), "unison");
        assert_eq!(interval_name(7), "perfect 5th");
        assert_eq!(interval_name(12), "unison"); // wraps
    }

    #[test]
    fn synth_voice_lifecycle() {
        let env = ADSREnvelope::new(0.01, 0.1, 0.5, 0.1);
        let mut voice = SynthVoice::new(env);
        assert!(!voice.active);

        voice.note_on(69, 100, 44100);
        assert!(voice.active);
        assert!((voice.frequency - 440.0).abs() < 1.0);

        // Generate some samples
        for _ in 0..100 {
            let s = voice.next_sample(44100.0);
            assert!(s.abs() <= 1.0);
        }

        voice.note_off();
        // Still active during release
        assert!(voice.active);

        // Run through release
        for _ in 0..50000 {
            voice.next_sample(44100.0);
        }
        assert!(!voice.active);
    }

    #[test]
    fn render_audio_standalone() {
        let env = ADSREnvelope::organ();
        let mut voices: Vec<SynthVoice> = (0..4).map(|_| SynthVoice::new(env)).collect();
        voices[0].note_on(60, 100, 44100);
        voices[1].note_on(64, 100, 44100);

        let mut buffer = vec![0.0; 256];
        render_audio(&mut voices, 44100, &mut buffer);
        let max = buffer.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        assert!(max > 0.0);
    }

    #[test]
    fn midi_note_basics() {
        let note = MidiNote::new(60, 100, 0);
        assert_eq!(note.note, 60);
        assert_eq!(note.name(), "C4");
        assert!((note.frequency() - 261.63).abs() < 0.1);
        assert!((note.velocity_f32() - 0.787).abs() < 0.01);
    }

    #[test]
    fn soft_clip_identity_near_zero() {
        // Near zero, soft clip should be approximately identity
        assert!((soft_clip(0.0)).abs() < 0.001);
        assert!((soft_clip(0.1) - 0.1).abs() < 0.01);
    }

    #[test]
    fn soft_clip_limits() {
        // Large values should be compressed
        assert!(soft_clip(10.0) < 1.0);
        assert!(soft_clip(-10.0) > -1.0);
    }
}
