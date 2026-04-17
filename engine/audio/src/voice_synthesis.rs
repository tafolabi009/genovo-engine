//! # Simple Speech Synthesis
//!
//! A lightweight formant-based speech synthesis system for the Genovo engine.
//! Generates speech-like audio from text using phoneme decomposition and
//! vocal tract modelling.
//!
//! ## Features
//!
//! - **Phoneme library** — Defines formant frequencies, bandwidths, and
//!   durations for English phonemes.
//! - **Text-to-phoneme** — Rule-based conversion from English text to phoneme
//!   sequences.
//! - **Formant synthesis** — Cascade formant synthesizer modelling the vocal
//!   tract resonances (F1, F2, F3).
//! - **Pitch and speed control** — Adjustable fundamental frequency and
//!   speaking rate.
//! - **Voice effects** — Robot voice (quantized pitch), alien voice (shifted
//!   formants), whisper (noise excitation), and more.

use std::collections::HashMap;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Phoneme
// ---------------------------------------------------------------------------

/// A phoneme identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phoneme {
    // Vowels
    /// /a/ as in "father"
    AA,
    /// /ae/ as in "cat"
    AE,
    /// /ah/ as in "but"
    AH,
    /// /aw/ as in "cow"
    AW,
    /// /ay/ as in "bite"
    AY,
    /// /eh/ as in "bed"
    EH,
    /// /ee/ as in "see"
    EE,
    /// /er/ as in "bird"
    ER,
    /// /ih/ as in "bit"
    IH,
    /// /iy/ as in "bee"
    IY,
    /// /oh/ as in "boat"
    OH,
    /// /oo/ as in "food"
    OO,
    /// /ow/ as in "go"
    OW,
    /// /oy/ as in "boy"
    OY,
    /// /uh/ as in "book"
    UH,

    // Consonants — Stops
    /// /b/ as in "bat"
    B,
    /// /d/ as in "dog"
    D,
    /// /g/ as in "get"
    G,
    /// /k/ as in "cat"
    K,
    /// /p/ as in "pat"
    P,
    /// /t/ as in "top"
    T,

    // Consonants — Fricatives
    /// /f/ as in "fan"
    F,
    /// /s/ as in "sit"
    S,
    /// /sh/ as in "ship"
    SH,
    /// /th/ as in "think"
    TH,
    /// /v/ as in "van"
    V,
    /// /z/ as in "zip"
    Z,
    /// /zh/ as in "measure"
    ZH,
    /// /h/ as in "hat"
    H,

    // Consonants — Nasals
    /// /m/ as in "man"
    M,
    /// /n/ as in "no"
    N,
    /// /ng/ as in "sing"
    NG,

    // Consonants — Liquids and Glides
    /// /l/ as in "let"
    L,
    /// /r/ as in "red"
    R,
    /// /w/ as in "wet"
    W,
    /// /y/ as in "yes"
    Y,

    // Special
    /// /ch/ as in "church"
    CH,
    /// /j/ as in "judge"
    J,

    /// Silence / pause.
    Silence,
}

impl Phoneme {
    /// Returns a human-readable name.
    pub fn name(&self) -> &str {
        match self {
            Phoneme::AA => "AA",
            Phoneme::AE => "AE",
            Phoneme::AH => "AH",
            Phoneme::AW => "AW",
            Phoneme::AY => "AY",
            Phoneme::EH => "EH",
            Phoneme::EE => "EE",
            Phoneme::ER => "ER",
            Phoneme::IH => "IH",
            Phoneme::IY => "IY",
            Phoneme::OH => "OH",
            Phoneme::OO => "OO",
            Phoneme::OW => "OW",
            Phoneme::OY => "OY",
            Phoneme::UH => "UH",
            Phoneme::B => "B",
            Phoneme::D => "D",
            Phoneme::G => "G",
            Phoneme::K => "K",
            Phoneme::P => "P",
            Phoneme::T => "T",
            Phoneme::F => "F",
            Phoneme::S => "S",
            Phoneme::SH => "SH",
            Phoneme::TH => "TH",
            Phoneme::V => "V",
            Phoneme::Z => "Z",
            Phoneme::ZH => "ZH",
            Phoneme::H => "H",
            Phoneme::M => "M",
            Phoneme::N => "N",
            Phoneme::NG => "NG",
            Phoneme::L => "L",
            Phoneme::R => "R",
            Phoneme::W => "W",
            Phoneme::Y => "Y",
            Phoneme::CH => "CH",
            Phoneme::J => "J",
            Phoneme::Silence => "SIL",
        }
    }

    /// Returns true if this is a vowel phoneme.
    pub fn is_vowel(&self) -> bool {
        matches!(
            self,
            Phoneme::AA
                | Phoneme::AE
                | Phoneme::AH
                | Phoneme::AW
                | Phoneme::AY
                | Phoneme::EH
                | Phoneme::EE
                | Phoneme::ER
                | Phoneme::IH
                | Phoneme::IY
                | Phoneme::OH
                | Phoneme::OO
                | Phoneme::OW
                | Phoneme::OY
                | Phoneme::UH
        )
    }

    /// Returns true if this is a consonant.
    pub fn is_consonant(&self) -> bool {
        !self.is_vowel() && *self != Phoneme::Silence
    }

    /// Returns true if this is a voiced phoneme.
    pub fn is_voiced(&self) -> bool {
        match self {
            Phoneme::B | Phoneme::D | Phoneme::G | Phoneme::V | Phoneme::Z
            | Phoneme::ZH | Phoneme::M | Phoneme::N | Phoneme::NG | Phoneme::L
            | Phoneme::R | Phoneme::W | Phoneme::Y | Phoneme::J => true,
            p if p.is_vowel() => true,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// FormantParams
// ---------------------------------------------------------------------------

/// Formant parameters for a phoneme.
///
/// The vocal tract is modelled as a series of resonant filters (formants).
/// F1 and F2 primarily determine vowel identity; F3 adds naturalness.
#[derive(Debug, Clone)]
pub struct FormantParams {
    /// First formant frequency (Hz).
    pub f1: f32,
    /// First formant bandwidth (Hz).
    pub b1: f32,
    /// Second formant frequency (Hz).
    pub f2: f32,
    /// Second formant bandwidth (Hz).
    pub b2: f32,
    /// Third formant frequency (Hz).
    pub f3: f32,
    /// Third formant bandwidth (Hz).
    pub b3: f32,
    /// Amplitude of the phoneme [0, 1].
    pub amplitude: f32,
    /// Duration in seconds.
    pub duration: f32,
    /// Whether to use voiced (glottal pulse) or unvoiced (noise) excitation.
    pub voiced: bool,
    /// Noise mix factor [0, 1]. 0 = pure voice, 1 = pure noise.
    pub noise_mix: f32,
}

impl FormantParams {
    /// Create a new formant parameter set.
    pub fn new(
        f1: f32,
        b1: f32,
        f2: f32,
        b2: f32,
        f3: f32,
        b3: f32,
        amplitude: f32,
        duration: f32,
        voiced: bool,
    ) -> Self {
        Self {
            f1,
            b1,
            f2,
            b2,
            f3,
            b3,
            amplitude,
            duration,
            voiced,
            noise_mix: if voiced { 0.0 } else { 1.0 },
        }
    }

    /// Linearly interpolate between two formant parameter sets.
    pub fn lerp(a: &FormantParams, b: &FormantParams, t: f32) -> FormantParams {
        let t = t.clamp(0.0, 1.0);
        FormantParams {
            f1: a.f1 + (b.f1 - a.f1) * t,
            b1: a.b1 + (b.b1 - a.b1) * t,
            f2: a.f2 + (b.f2 - a.f2) * t,
            b2: a.b2 + (b.b2 - a.b2) * t,
            f3: a.f3 + (b.f3 - a.f3) * t,
            b3: a.b3 + (b.b3 - a.b3) * t,
            amplitude: a.amplitude + (b.amplitude - a.amplitude) * t,
            duration: a.duration + (b.duration - a.duration) * t,
            voiced: if t < 0.5 { a.voiced } else { b.voiced },
            noise_mix: a.noise_mix + (b.noise_mix - a.noise_mix) * t,
        }
    }
}

// ---------------------------------------------------------------------------
// PhonemeLibrary
// ---------------------------------------------------------------------------

/// Library of formant parameters for each phoneme.
pub struct PhonemeLibrary {
    entries: HashMap<Phoneme, FormantParams>,
}

impl PhonemeLibrary {
    /// Create a library with default English phoneme data.
    pub fn english() -> Self {
        let mut entries = HashMap::new();

        // Vowels — formant frequencies from standard linguistic data.
        entries.insert(Phoneme::AA, FormantParams::new(730.0, 90.0, 1090.0, 110.0, 2440.0, 120.0, 1.0, 0.12, true));
        entries.insert(Phoneme::AE, FormantParams::new(660.0, 60.0, 1720.0, 100.0, 2410.0, 120.0, 1.0, 0.10, true));
        entries.insert(Phoneme::AH, FormantParams::new(520.0, 70.0, 1190.0, 110.0, 2390.0, 120.0, 0.9, 0.08, true));
        entries.insert(Phoneme::AW, FormantParams::new(680.0, 80.0, 1060.0, 100.0, 2380.0, 120.0, 1.0, 0.14, true));
        entries.insert(Phoneme::AY, FormantParams::new(710.0, 80.0, 1100.0, 100.0, 2440.0, 120.0, 1.0, 0.14, true));
        entries.insert(Phoneme::EH, FormantParams::new(530.0, 60.0, 1840.0, 100.0, 2480.0, 120.0, 0.9, 0.08, true));
        entries.insert(Phoneme::EE, FormantParams::new(270.0, 60.0, 2290.0, 100.0, 3010.0, 120.0, 0.8, 0.10, true));
        entries.insert(Phoneme::ER, FormantParams::new(490.0, 70.0, 1350.0, 100.0, 1690.0, 120.0, 0.8, 0.10, true));
        entries.insert(Phoneme::IH, FormantParams::new(390.0, 50.0, 1990.0, 100.0, 2550.0, 120.0, 0.85, 0.07, true));
        entries.insert(Phoneme::IY, FormantParams::new(280.0, 50.0, 2250.0, 100.0, 2890.0, 120.0, 0.8, 0.10, true));
        entries.insert(Phoneme::OH, FormantParams::new(570.0, 60.0, 840.0, 100.0, 2410.0, 120.0, 1.0, 0.10, true));
        entries.insert(Phoneme::OO, FormantParams::new(300.0, 50.0, 870.0, 100.0, 2240.0, 120.0, 0.9, 0.10, true));
        entries.insert(Phoneme::OW, FormantParams::new(490.0, 60.0, 1100.0, 100.0, 2440.0, 120.0, 1.0, 0.12, true));
        entries.insert(Phoneme::OY, FormantParams::new(570.0, 60.0, 840.0, 100.0, 2410.0, 120.0, 1.0, 0.14, true));
        entries.insert(Phoneme::UH, FormantParams::new(440.0, 70.0, 1020.0, 100.0, 2240.0, 120.0, 0.85, 0.08, true));

        // Consonants — simplified formant data.
        entries.insert(Phoneme::B, FormantParams::new(200.0, 80.0, 1100.0, 200.0, 2150.0, 300.0, 0.3, 0.04, true));
        entries.insert(Phoneme::D, FormantParams::new(300.0, 80.0, 1700.0, 200.0, 2600.0, 300.0, 0.3, 0.03, true));
        entries.insert(Phoneme::G, FormantParams::new(250.0, 80.0, 2000.0, 200.0, 2700.0, 300.0, 0.3, 0.04, true));
        entries.insert(Phoneme::K, FormantParams::new(300.0, 80.0, 2000.0, 200.0, 2700.0, 300.0, 0.2, 0.04, false));
        entries.insert(Phoneme::P, FormantParams::new(200.0, 80.0, 1100.0, 200.0, 2150.0, 300.0, 0.2, 0.03, false));
        entries.insert(Phoneme::T, FormantParams::new(300.0, 80.0, 1700.0, 200.0, 2600.0, 300.0, 0.2, 0.03, false));
        entries.insert(Phoneme::F, FormantParams::new(400.0, 200.0, 1500.0, 500.0, 2500.0, 500.0, 0.15, 0.06, false));
        entries.insert(Phoneme::S, FormantParams::new(400.0, 200.0, 1800.0, 500.0, 5000.0, 500.0, 0.2, 0.06, false));
        entries.insert(Phoneme::SH, FormantParams::new(400.0, 200.0, 1800.0, 500.0, 3500.0, 500.0, 0.2, 0.06, false));
        entries.insert(Phoneme::TH, FormantParams::new(400.0, 200.0, 1500.0, 500.0, 2500.0, 500.0, 0.1, 0.06, false));
        entries.insert(Phoneme::V, FormantParams::new(300.0, 100.0, 1500.0, 300.0, 2500.0, 400.0, 0.2, 0.05, true));
        entries.insert(Phoneme::Z, FormantParams::new(300.0, 100.0, 1800.0, 300.0, 5000.0, 400.0, 0.25, 0.05, true));
        entries.insert(Phoneme::ZH, FormantParams::new(300.0, 100.0, 1800.0, 300.0, 3500.0, 400.0, 0.2, 0.05, true));
        entries.insert(Phoneme::H, FormantParams::new(500.0, 400.0, 1500.0, 600.0, 2500.0, 600.0, 0.1, 0.04, false));
        entries.insert(Phoneme::M, FormantParams::new(280.0, 60.0, 1000.0, 200.0, 2200.0, 300.0, 0.5, 0.06, true));
        entries.insert(Phoneme::N, FormantParams::new(280.0, 60.0, 1700.0, 200.0, 2500.0, 300.0, 0.5, 0.05, true));
        entries.insert(Phoneme::NG, FormantParams::new(280.0, 60.0, 2000.0, 200.0, 2700.0, 300.0, 0.4, 0.06, true));
        entries.insert(Phoneme::L, FormantParams::new(350.0, 80.0, 1100.0, 200.0, 2400.0, 300.0, 0.4, 0.05, true));
        entries.insert(Phoneme::R, FormantParams::new(420.0, 80.0, 1300.0, 200.0, 1600.0, 300.0, 0.4, 0.05, true));
        entries.insert(Phoneme::W, FormantParams::new(290.0, 70.0, 610.0, 200.0, 2150.0, 300.0, 0.4, 0.04, true));
        entries.insert(Phoneme::Y, FormantParams::new(280.0, 60.0, 2250.0, 200.0, 2890.0, 300.0, 0.4, 0.04, true));
        entries.insert(Phoneme::CH, FormantParams::new(400.0, 200.0, 1800.0, 500.0, 3500.0, 500.0, 0.15, 0.06, false));
        entries.insert(Phoneme::J, FormantParams::new(300.0, 100.0, 1800.0, 300.0, 3500.0, 400.0, 0.2, 0.06, true));
        entries.insert(Phoneme::Silence, FormantParams::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, false));

        Self { entries }
    }

    /// Look up formant parameters for a phoneme.
    pub fn get(&self, phoneme: &Phoneme) -> Option<&FormantParams> {
        self.entries.get(phoneme)
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add or replace a phoneme entry.
    pub fn set(&mut self, phoneme: Phoneme, params: FormantParams) {
        self.entries.insert(phoneme, params);
    }
}

// ---------------------------------------------------------------------------
// Text-to-Phoneme converter
// ---------------------------------------------------------------------------

/// Rule-based text-to-phoneme converter for English.
///
/// Uses a simplified letter-to-phoneme mapping. This is not production-grade
/// but handles common words reasonably well for game dialogue.
pub struct TextToPhoneme {
    /// Letter cluster rules mapping string patterns to phoneme sequences.
    rules: Vec<(String, Vec<Phoneme>)>,
}

impl TextToPhoneme {
    /// Create a new converter with default English rules.
    pub fn english() -> Self {
        let mut rules: Vec<(String, Vec<Phoneme>)> = Vec::new();

        // Multi-character rules (must be checked before single-character).
        rules.push(("th".to_string(), vec![Phoneme::TH]));
        rules.push(("sh".to_string(), vec![Phoneme::SH]));
        rules.push(("ch".to_string(), vec![Phoneme::CH]));
        rules.push(("ng".to_string(), vec![Phoneme::NG]));
        rules.push(("ph".to_string(), vec![Phoneme::F]));
        rules.push(("wh".to_string(), vec![Phoneme::W]));
        rules.push(("ck".to_string(), vec![Phoneme::K]));
        rules.push(("ee".to_string(), vec![Phoneme::IY]));
        rules.push(("oo".to_string(), vec![Phoneme::OO]));
        rules.push(("ea".to_string(), vec![Phoneme::IY]));
        rules.push(("ou".to_string(), vec![Phoneme::AW]));
        rules.push(("oi".to_string(), vec![Phoneme::OY]));
        rules.push(("oy".to_string(), vec![Phoneme::OY]));
        rules.push(("ai".to_string(), vec![Phoneme::AY]));
        rules.push(("ay".to_string(), vec![Phoneme::AY]));
        rules.push(("ow".to_string(), vec![Phoneme::OW]));
        rules.push(("aw".to_string(), vec![Phoneme::AW]));
        rules.push(("au".to_string(), vec![Phoneme::AW]));
        rules.push(("igh".to_string(), vec![Phoneme::AY]));
        rules.push(("tion".to_string(), vec![Phoneme::SH, Phoneme::AH, Phoneme::N]));
        rules.push(("sion".to_string(), vec![Phoneme::ZH, Phoneme::AH, Phoneme::N]));
        rules.push(("qu".to_string(), vec![Phoneme::K, Phoneme::W]));

        // Single-character rules.
        rules.push(("a".to_string(), vec![Phoneme::AE]));
        rules.push(("b".to_string(), vec![Phoneme::B]));
        rules.push(("c".to_string(), vec![Phoneme::K]));
        rules.push(("d".to_string(), vec![Phoneme::D]));
        rules.push(("e".to_string(), vec![Phoneme::EH]));
        rules.push(("f".to_string(), vec![Phoneme::F]));
        rules.push(("g".to_string(), vec![Phoneme::G]));
        rules.push(("h".to_string(), vec![Phoneme::H]));
        rules.push(("i".to_string(), vec![Phoneme::IH]));
        rules.push(("j".to_string(), vec![Phoneme::J]));
        rules.push(("k".to_string(), vec![Phoneme::K]));
        rules.push(("l".to_string(), vec![Phoneme::L]));
        rules.push(("m".to_string(), vec![Phoneme::M]));
        rules.push(("n".to_string(), vec![Phoneme::N]));
        rules.push(("o".to_string(), vec![Phoneme::OH]));
        rules.push(("p".to_string(), vec![Phoneme::P]));
        rules.push(("r".to_string(), vec![Phoneme::R]));
        rules.push(("s".to_string(), vec![Phoneme::S]));
        rules.push(("t".to_string(), vec![Phoneme::T]));
        rules.push(("u".to_string(), vec![Phoneme::AH]));
        rules.push(("v".to_string(), vec![Phoneme::V]));
        rules.push(("w".to_string(), vec![Phoneme::W]));
        rules.push(("x".to_string(), vec![Phoneme::K, Phoneme::S]));
        rules.push(("y".to_string(), vec![Phoneme::Y]));
        rules.push(("z".to_string(), vec![Phoneme::Z]));

        Self { rules }
    }

    /// Convert text to a sequence of phonemes.
    pub fn convert(&self, text: &str) -> Vec<Phoneme> {
        let lower = text.to_lowercase();
        let mut phonemes = Vec::new();
        let chars: Vec<char> = lower.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i] == ' ' || chars[i] == ',' || chars[i] == '.' {
                phonemes.push(Phoneme::Silence);
                i += 1;
                continue;
            }

            if !chars[i].is_alphabetic() {
                i += 1;
                continue;
            }

            // Try multi-character rules (longest match first).
            let mut matched = false;
            for (pattern, result) in &self.rules {
                let pat_len = pattern.len();
                if i + pat_len <= chars.len() {
                    let slice: String = chars[i..i + pat_len].iter().collect();
                    if slice == *pattern {
                        phonemes.extend_from_slice(result);
                        i += pat_len;
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                // Skip unknown characters.
                i += 1;
            }
        }

        phonemes
    }
}

// ---------------------------------------------------------------------------
// FormantFilter
// ---------------------------------------------------------------------------

/// A single second-order resonant filter modelling one formant.
#[derive(Debug, Clone)]
struct FormantFilter {
    /// Center frequency.
    frequency: f32,
    /// Bandwidth.
    bandwidth: f32,
    /// Filter coefficients.
    a1: f32,
    a2: f32,
    b0: f32,
    /// State variables.
    y1: f32,
    y2: f32,
}

impl FormantFilter {
    /// Create a new formant filter.
    fn new(frequency: f32, bandwidth: f32, sample_rate: f32) -> Self {
        let mut filter = Self {
            frequency,
            bandwidth,
            a1: 0.0,
            a2: 0.0,
            b0: 1.0,
            y1: 0.0,
            y2: 0.0,
        };
        filter.update_coefficients(sample_rate);
        filter
    }

    /// Recompute filter coefficients.
    fn update_coefficients(&mut self, sample_rate: f32) {
        let r = (-PI * self.bandwidth / sample_rate).exp();
        let theta = 2.0 * PI * self.frequency / sample_rate;

        self.a1 = -2.0 * r * theta.cos();
        self.a2 = r * r;
        self.b0 = 1.0 - r;
    }

    /// Set frequency and bandwidth.
    fn set_params(&mut self, frequency: f32, bandwidth: f32, sample_rate: f32) {
        self.frequency = frequency;
        self.bandwidth = bandwidth;
        self.update_coefficients(sample_rate);
    }

    /// Process one sample.
    fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = output;
        output
    }

    /// Reset state.
    fn reset(&mut self) {
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// GlottalSource
// ---------------------------------------------------------------------------

/// Simplified glottal pulse source (excitation signal for voiced sounds).
#[derive(Debug, Clone)]
struct GlottalSource {
    /// Fundamental frequency (pitch) in Hz.
    frequency: f32,
    /// Current phase [0, 1).
    phase: f32,
    /// Sample rate.
    sample_rate: f32,
}

impl GlottalSource {
    fn new(frequency: f32, sample_rate: f32) -> Self {
        Self {
            frequency,
            phase: 0.0,
            sample_rate,
        }
    }

    /// Set the fundamental frequency.
    fn set_frequency(&mut self, freq: f32) {
        self.frequency = freq.max(20.0);
    }

    /// Generate one sample.
    fn next_sample(&mut self) -> f32 {
        let phase_inc = self.frequency / self.sample_rate;
        self.phase += phase_inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // Rosenberg glottal pulse model (simplified).
        let t = self.phase;
        if t < 0.4 {
            // Opening phase.
            let x = t / 0.4;
            3.0 * x * x - 2.0 * x * x * x
        } else if t < 0.6 {
            // Closing phase.
            let x = (t - 0.4) / 0.2;
            1.0 - x * x
        } else {
            // Closed phase.
            0.0
        }
    }

    /// Reset phase.
    fn reset(&mut self) {
        self.phase = 0.0;
    }
}

// ---------------------------------------------------------------------------
// NoiseSource
// ---------------------------------------------------------------------------

/// Simple noise source for unvoiced sounds.
#[derive(Debug, Clone)]
struct NoiseSource {
    state: u32,
}

impl NoiseSource {
    fn new() -> Self {
        Self { state: 12345 }
    }

    fn next_sample(&mut self) -> f32 {
        // xorshift32.
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        // Map to [-1, 1].
        (self.state as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// VoiceEffect
// ---------------------------------------------------------------------------

/// Voice effect types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoiceEffect {
    /// No effect.
    None,
    /// Robot voice: quantized pitch steps.
    Robot {
        /// Number of pitch quantization steps.
        steps: u32,
    },
    /// Alien voice: shifted formants.
    Alien {
        /// Formant frequency shift multiplier.
        shift: f32,
    },
    /// Whisper: noise-only excitation.
    Whisper,
    /// Deep voice: lowered pitch.
    Deep {
        /// Pitch multiplier (< 1.0 for deeper).
        factor: f32,
    },
    /// Chipmunk: raised pitch.
    Chipmunk {
        /// Pitch multiplier (> 1.0 for higher).
        factor: f32,
    },
    /// Megaphone: band-limited, distorted.
    Megaphone,
}

// ---------------------------------------------------------------------------
// VoiceSynthesizer
// ---------------------------------------------------------------------------

/// Main speech synthesizer.
///
/// Produces audio samples from a sequence of phonemes using formant synthesis.
///
/// # Example
///
/// ```ignore
/// use genovo_audio::voice_synthesis::*;
///
/// let library = PhonemeLibrary::english();
/// let converter = TextToPhoneme::english();
/// let mut synth = VoiceSynthesizer::new(44100, &library);
///
/// synth.set_pitch(150.0); // Hz
/// synth.set_speed(1.0);
///
/// let phonemes = converter.convert("hello world");
/// let samples = synth.synthesize(&phonemes, &library);
/// ```
pub struct VoiceSynthesizer {
    /// Sample rate.
    sample_rate: u32,
    /// Fundamental frequency (pitch) in Hz.
    pitch: f32,
    /// Speaking rate multiplier (1.0 = normal).
    speed: f32,
    /// Volume [0, 1].
    volume: f32,
    /// Glottal source.
    glottal: GlottalSource,
    /// Noise source.
    noise: NoiseSource,
    /// Formant filters (F1, F2, F3).
    formants: [FormantFilter; 3],
    /// Active voice effect.
    effect: VoiceEffect,
    /// Transition time between phonemes (in seconds).
    transition_time: f32,
}

impl VoiceSynthesizer {
    /// Create a new synthesizer.
    pub fn new(sample_rate: u32, _library: &PhonemeLibrary) -> Self {
        let sr = sample_rate as f32;
        Self {
            sample_rate,
            pitch: 120.0,
            speed: 1.0,
            volume: 0.8,
            glottal: GlottalSource::new(120.0, sr),
            noise: NoiseSource::new(),
            formants: [
                FormantFilter::new(500.0, 80.0, sr),
                FormantFilter::new(1500.0, 100.0, sr),
                FormantFilter::new(2500.0, 120.0, sr),
            ],
            effect: VoiceEffect::None,
            transition_time: 0.01,
        }
    }

    /// Set the fundamental frequency (pitch).
    pub fn set_pitch(&mut self, hz: f32) {
        self.pitch = hz.clamp(50.0, 500.0);
        self.glottal.set_frequency(self.pitch);
    }

    /// Set speaking rate.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(0.25, 4.0);
    }

    /// Set volume.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
    }

    /// Set voice effect.
    pub fn set_effect(&mut self, effect: VoiceEffect) {
        self.effect = effect;
    }

    /// Set the transition time between phonemes.
    pub fn set_transition_time(&mut self, seconds: f32) {
        self.transition_time = seconds.clamp(0.001, 0.1);
    }

    /// Synthesize audio from a phoneme sequence.
    pub fn synthesize(
        &mut self,
        phonemes: &[Phoneme],
        library: &PhonemeLibrary,
    ) -> Vec<f32> {
        let sr = self.sample_rate as f32;
        let mut output = Vec::new();

        // Reset filters.
        for f in &mut self.formants {
            f.reset();
        }
        self.glottal.reset();

        for (idx, phoneme) in phonemes.iter().enumerate() {
            let params = match library.get(phoneme) {
                Some(p) => p.clone(),
                None => continue,
            };

            let duration = params.duration / self.speed;
            let num_samples = (duration * sr) as usize;

            // Apply voice effects to the parameters.
            let effective_params = self.apply_effect_to_params(&params);

            // Compute pitch for this phoneme.
            let phoneme_pitch = self.compute_pitch(idx, phonemes.len());
            self.glottal.set_frequency(phoneme_pitch);

            // Update formant filter frequencies.
            self.formants[0].set_params(effective_params.f1, effective_params.b1, sr);
            self.formants[1].set_params(effective_params.f2, effective_params.b2, sr);
            self.formants[2].set_params(effective_params.f3, effective_params.b3, sr);

            // Generate samples for this phoneme.
            for i in 0..num_samples {
                let t = i as f32 / num_samples as f32;

                // Excitation: blend between glottal pulse and noise.
                let glottal_sample = self.glottal.next_sample();
                let noise_sample = self.noise.next_sample();
                let noise_mix = effective_params.noise_mix;

                let excitation = glottal_sample * (1.0 - noise_mix)
                    + noise_sample * noise_mix;

                // Pass through formant filter cascade.
                let mut sample = excitation;
                for formant in &mut self.formants {
                    sample = formant.process(sample);
                }

                // Envelope: smooth attack/release.
                let envelope = self.compute_envelope(t, duration);
                sample *= envelope * effective_params.amplitude * self.volume;

                // Clamp.
                sample = sample.clamp(-1.0, 1.0);

                output.push(sample);
            }
        }

        // Apply post-processing effects.
        self.post_process(&mut output);

        output
    }

    /// Compute pitch with natural variation.
    fn compute_pitch(&self, phoneme_idx: usize, total: usize) -> f32 {
        let base = self.pitch;

        match self.effect {
            VoiceEffect::Robot { steps } => {
                // Quantize to discrete steps.
                let step_size = base / steps as f32;
                let quantized = (base / step_size).round() * step_size;
                quantized
            }
            VoiceEffect::Deep { factor } => base * factor,
            VoiceEffect::Chipmunk { factor } => base * factor,
            _ => {
                // Natural intonation: slight rise at beginning, fall at end.
                let position = phoneme_idx as f32 / total.max(1) as f32;
                let intonation = if position < 0.3 {
                    1.0 + position * 0.1
                } else if position > 0.7 {
                    1.0 - (position - 0.7) * 0.15
                } else {
                    1.0
                };
                base * intonation
            }
        }
    }

    /// Apply voice effect to formant parameters.
    fn apply_effect_to_params(&self, params: &FormantParams) -> FormantParams {
        let mut result = params.clone();

        match self.effect {
            VoiceEffect::Alien { shift } => {
                result.f1 *= shift;
                result.f2 *= shift;
                result.f3 *= shift;
            }
            VoiceEffect::Whisper => {
                result.noise_mix = 1.0;
                result.voiced = false;
            }
            VoiceEffect::Megaphone => {
                // Narrow bandwidth for "telephone" quality.
                result.b1 = result.b1 * 2.0;
                result.b2 = result.b2 * 2.0;
                result.b3 = result.b3 * 3.0;
                result.noise_mix = (result.noise_mix + 0.1).min(1.0);
            }
            _ => {}
        }

        result
    }

    /// Compute amplitude envelope for a phoneme.
    fn compute_envelope(&self, t: f32, _duration: f32) -> f32 {
        let attack = 0.1_f32; // First 10% is attack.
        let release = 0.1_f32; // Last 10% is release.

        if t < attack {
            t / attack
        } else if t > (1.0 - release) {
            (1.0 - t) / release
        } else {
            1.0
        }
    }

    /// Post-process the output buffer with effects.
    fn post_process(&self, samples: &mut [f32]) {
        match self.effect {
            VoiceEffect::Megaphone => {
                // Soft clipping for distortion.
                for sample in samples.iter_mut() {
                    *sample = (*sample * 2.0).tanh() * 0.7;
                }
            }
            VoiceEffect::Robot { .. } => {
                // Add slight ring modulation.
                let mod_freq = 50.0;
                for (i, sample) in samples.iter_mut().enumerate() {
                    let t = i as f32 / self.sample_rate as f32;
                    let modulator = (2.0 * PI * mod_freq * t).sin() * 0.3 + 0.7;
                    *sample *= modulator;
                }
            }
            _ => {}
        }
    }

    /// Returns the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the current pitch.
    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Returns the current speed.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Returns the current volume.
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Returns the active effect.
    pub fn effect(&self) -> &VoiceEffect {
        &self.effect
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phoneme_library() {
        let lib = PhonemeLibrary::english();
        assert!(!lib.is_empty());

        let aa = lib.get(&Phoneme::AA).unwrap();
        assert!(aa.f1 > 0.0);
        assert!(aa.voiced);
    }

    #[test]
    fn test_phoneme_classification() {
        assert!(Phoneme::AA.is_vowel());
        assert!(!Phoneme::AA.is_consonant());
        assert!(Phoneme::B.is_consonant());
        assert!(Phoneme::B.is_voiced());
        assert!(!Phoneme::S.is_voiced());
    }

    #[test]
    fn test_text_to_phoneme() {
        let converter = TextToPhoneme::english();
        let phonemes = converter.convert("hello");
        assert!(!phonemes.is_empty());
    }

    #[test]
    fn test_text_to_phoneme_with_space() {
        let converter = TextToPhoneme::english();
        let phonemes = converter.convert("hi there");
        assert!(phonemes.contains(&Phoneme::Silence));
    }

    #[test]
    fn test_synthesizer_basic() {
        let lib = PhonemeLibrary::english();
        let mut synth = VoiceSynthesizer::new(22050, &lib);

        let phonemes = vec![Phoneme::AA, Phoneme::EH];
        let samples = synth.synthesize(&phonemes, &lib);

        assert!(!samples.is_empty());
        // All samples should be in [-1, 1].
        for &s in &samples {
            assert!(s >= -1.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_synthesizer_silence() {
        let lib = PhonemeLibrary::english();
        let mut synth = VoiceSynthesizer::new(22050, &lib);

        let phonemes = vec![Phoneme::Silence];
        let samples = synth.synthesize(&phonemes, &lib);
        // Silence should produce near-zero samples.
        let max_amp: f32 = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
        assert!(max_amp < 0.01);
    }

    #[test]
    fn test_synthesizer_effects() {
        let lib = PhonemeLibrary::english();
        let mut synth = VoiceSynthesizer::new(22050, &lib);

        let phonemes = vec![Phoneme::AA];

        synth.set_effect(VoiceEffect::Robot { steps: 4 });
        let robot_samples = synth.synthesize(&phonemes, &lib);
        assert!(!robot_samples.is_empty());

        synth.set_effect(VoiceEffect::Whisper);
        let whisper_samples = synth.synthesize(&phonemes, &lib);
        assert!(!whisper_samples.is_empty());

        synth.set_effect(VoiceEffect::Alien { shift: 1.5 });
        let alien_samples = synth.synthesize(&phonemes, &lib);
        assert!(!alien_samples.is_empty());
    }

    #[test]
    fn test_synthesizer_pitch_speed() {
        let lib = PhonemeLibrary::english();
        let mut synth = VoiceSynthesizer::new(22050, &lib);

        synth.set_pitch(200.0);
        assert!((synth.pitch() - 200.0).abs() < 1e-5);

        synth.set_speed(2.0);
        assert!((synth.speed() - 2.0).abs() < 1e-5);

        let phonemes = vec![Phoneme::AA];
        let fast_samples = synth.synthesize(&phonemes, &lib);

        synth.set_speed(0.5);
        let slow_samples = synth.synthesize(&phonemes, &lib);

        // Slower speed should produce more samples.
        assert!(slow_samples.len() > fast_samples.len());
    }

    #[test]
    fn test_formant_lerp() {
        let a = FormantParams::new(300.0, 50.0, 1000.0, 100.0, 2500.0, 120.0, 1.0, 0.1, true);
        let b = FormantParams::new(700.0, 90.0, 1500.0, 110.0, 2800.0, 130.0, 0.5, 0.2, true);
        let mid = FormantParams::lerp(&a, &b, 0.5);

        assert!((mid.f1 - 500.0).abs() < 1e-3);
        assert!((mid.amplitude - 0.75).abs() < 1e-3);
    }

    #[test]
    fn test_glottal_source() {
        let mut source = GlottalSource::new(100.0, 22050.0);
        let mut samples = Vec::new();
        for _ in 0..22050 {
            samples.push(source.next_sample());
        }
        // Should have periodic pulses.
        let max = samples.iter().fold(0.0_f32, |a, &b| a.max(b));
        assert!(max > 0.5);
    }

    #[test]
    fn test_full_pipeline() {
        let lib = PhonemeLibrary::english();
        let converter = TextToPhoneme::english();
        let mut synth = VoiceSynthesizer::new(22050, &lib);

        let phonemes = converter.convert("hi");
        let samples = synth.synthesize(&phonemes, &lib);

        assert!(!samples.is_empty());
    }
}
