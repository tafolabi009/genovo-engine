// engine/ai/src/emotion_model.rs
//
// AI emotion system using the PAD (Pleasure-Arousal-Dominance) model.
//
// Provides:
// - Full PAD emotion space with clamping and interpolation
// - Emotion blending and exponential decay toward baseline
// - Emotion triggers with intensity, duration, and cooldown
// - Mood computation from emotion history (rolling average)
// - Personality traits modifying emotion responses
// - Emotion-to-behavior mapping (fearful -> flee, angry -> attack)
// - Social emotion spreading (group morale)
// - Facial expression blend shape output
// - Emotion memory and history tracking
// - Emotion contagion between nearby entities

use std::collections::HashMap;
use std::fmt;

pub type EmotionTriggerId = u32;
pub type EntityId = u64;

pub const PAD_MIN: f32 = -1.0;
pub const PAD_MAX: f32 = 1.0;
pub const DEFAULT_DECAY_RATE: f32 = 0.1;
pub const DEFAULT_MOOD_BLEND: f32 = 0.01;
pub const EMOTION_THRESHOLD: f32 = 0.15;
pub const MOOD_HISTORY_SIZE: usize = 60;
pub const MAX_ACTIVE_EMOTIONS: usize = 8;
pub const CONTAGION_DEFAULT_RADIUS: f32 = 10.0;
pub const CONTAGION_DEFAULT_STRENGTH: f32 = 0.3;

// ---------------------------------------------------------------------------
// PAD State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PadState {
    pub pleasure: f32,
    pub arousal: f32,
    pub dominance: f32,
}

impl PadState {
    pub const NEUTRAL: Self = Self { pleasure: 0.0, arousal: 0.0, dominance: 0.0 };

    pub fn new(p: f32, a: f32, d: f32) -> Self {
        Self {
            pleasure: p.clamp(PAD_MIN, PAD_MAX),
            arousal: a.clamp(PAD_MIN, PAD_MAX),
            dominance: d.clamp(PAD_MIN, PAD_MAX),
        }
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self::new(
            self.pleasure + (other.pleasure - self.pleasure) * t,
            self.arousal + (other.arousal - self.arousal) * t,
            self.dominance + (other.dominance - self.dominance) * t,
        )
    }

    pub fn add(self, other: Self) -> Self {
        Self::new(
            self.pleasure + other.pleasure,
            self.arousal + other.arousal,
            self.dominance + other.dominance,
        )
    }

    pub fn sub(self, other: Self) -> Self {
        Self::new(
            self.pleasure - other.pleasure,
            self.arousal - other.arousal,
            self.dominance - other.dominance,
        )
    }

    pub fn scale(self, s: f32) -> Self {
        Self::new(
            self.pleasure * s,
            self.arousal * s,
            self.dominance * s,
        )
    }

    pub fn magnitude(self) -> f32 {
        (self.pleasure * self.pleasure + self.arousal * self.arousal + self.dominance * self.dominance).sqrt()
    }

    /// Exponential decay toward a target (baseline).
    pub fn decay_toward(self, target: Self, rate: f32, dt: f32) -> Self {
        let factor = (-rate * dt).exp();
        Self::new(
            target.pleasure + (self.pleasure - target.pleasure) * factor,
            target.arousal + (self.arousal - target.arousal) * factor,
            target.dominance + (self.dominance - target.dominance) * factor,
        )
    }

    /// Simple decay toward neutral (zero).
    pub fn decay(self, rate: f32, dt: f32) -> Self {
        self.decay_toward(Self::NEUTRAL, rate, dt)
    }

    /// Euclidean distance to another PAD state.
    pub fn distance(self, other: Self) -> f32 {
        let dp = self.pleasure - other.pleasure;
        let da = self.arousal - other.arousal;
        let dd = self.dominance - other.dominance;
        (dp * dp + da * da + dd * dd).sqrt()
    }

    /// Clamp all components to PAD range.
    pub fn clamped(self) -> Self {
        Self {
            pleasure: self.pleasure.clamp(PAD_MIN, PAD_MAX),
            arousal: self.arousal.clamp(PAD_MIN, PAD_MAX),
            dominance: self.dominance.clamp(PAD_MIN, PAD_MAX),
        }
    }

    /// Weighted blend of multiple PAD states.
    pub fn weighted_blend(states: &[(Self, f32)]) -> Self {
        if states.is_empty() {
            return Self::NEUTRAL;
        }
        let total_weight: f32 = states.iter().map(|&(_, w)| w).sum();
        if total_weight < 1e-9 {
            return Self::NEUTRAL;
        }
        let mut p = 0.0f32;
        let mut a = 0.0f32;
        let mut d = 0.0f32;
        for &(state, weight) in states {
            let w = weight / total_weight;
            p += state.pleasure * w;
            a += state.arousal * w;
            d += state.dominance * w;
        }
        Self::new(p, a, d)
    }
}

impl Default for PadState {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

impl fmt::Display for PadState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PAD({:.2}, {:.2}, {:.2})", self.pleasure, self.arousal, self.dominance)
    }
}

// ---------------------------------------------------------------------------
// Emotion Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmotionType {
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Trust,
    Anticipation,
    Contempt,
    Shame,
    Pride,
    Love,
    Awe,
    Boredom,
    Neutral,
}

impl EmotionType {
    pub fn pad_center(&self) -> PadState {
        match self {
            Self::Joy => PadState::new(0.8, 0.5, 0.5),
            Self::Sadness => PadState::new(-0.7, -0.3, -0.5),
            Self::Anger => PadState::new(-0.6, 0.8, 0.6),
            Self::Fear => PadState::new(-0.7, 0.7, -0.7),
            Self::Surprise => PadState::new(0.2, 0.8, 0.0),
            Self::Disgust => PadState::new(-0.6, 0.3, 0.3),
            Self::Trust => PadState::new(0.5, 0.0, 0.3),
            Self::Anticipation => PadState::new(0.3, 0.5, 0.3),
            Self::Contempt => PadState::new(-0.3, 0.1, 0.7),
            Self::Shame => PadState::new(-0.5, -0.1, -0.6),
            Self::Pride => PadState::new(0.6, 0.3, 0.7),
            Self::Love => PadState::new(0.9, 0.4, 0.2),
            Self::Awe => PadState::new(0.4, 0.7, -0.3),
            Self::Boredom => PadState::new(-0.2, -0.7, -0.2),
            Self::Neutral => PadState::NEUTRAL,
        }
    }

    /// Get the "opposite" emotion (complementary in PAD space).
    pub fn opposite(&self) -> Self {
        match self {
            Self::Joy => Self::Sadness,
            Self::Sadness => Self::Joy,
            Self::Anger => Self::Fear,
            Self::Fear => Self::Anger,
            Self::Surprise => Self::Boredom,
            Self::Disgust => Self::Trust,
            Self::Trust => Self::Disgust,
            Self::Anticipation => Self::Boredom,
            Self::Contempt => Self::Awe,
            Self::Shame => Self::Pride,
            Self::Pride => Self::Shame,
            Self::Love => Self::Contempt,
            Self::Awe => Self::Boredom,
            Self::Boredom => Self::Anticipation,
            Self::Neutral => Self::Neutral,
        }
    }

    /// All non-neutral emotion types.
    pub fn all() -> &'static [EmotionType] {
        &[
            EmotionType::Joy, EmotionType::Sadness, EmotionType::Anger,
            EmotionType::Fear, EmotionType::Surprise, EmotionType::Disgust,
            EmotionType::Trust, EmotionType::Anticipation, EmotionType::Contempt,
            EmotionType::Shame, EmotionType::Pride, EmotionType::Love,
            EmotionType::Awe, EmotionType::Boredom,
        ]
    }
}

impl fmt::Display for EmotionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Joy => write!(f, "Joy"),
            Self::Sadness => write!(f, "Sadness"),
            Self::Anger => write!(f, "Anger"),
            Self::Fear => write!(f, "Fear"),
            Self::Surprise => write!(f, "Surprise"),
            Self::Disgust => write!(f, "Disgust"),
            Self::Trust => write!(f, "Trust"),
            Self::Anticipation => write!(f, "Anticipation"),
            Self::Contempt => write!(f, "Contempt"),
            Self::Shame => write!(f, "Shame"),
            Self::Pride => write!(f, "Pride"),
            Self::Love => write!(f, "Love"),
            Self::Awe => write!(f, "Awe"),
            Self::Boredom => write!(f, "Boredom"),
            Self::Neutral => write!(f, "Neutral"),
        }
    }
}

// ---------------------------------------------------------------------------
// Active Emotion Instance
// ---------------------------------------------------------------------------

/// An active emotion with intensity and remaining duration.
#[derive(Debug, Clone)]
pub struct ActiveEmotion {
    pub emotion_type: EmotionType,
    pub intensity: f32,
    pub max_intensity: f32,
    pub duration: f32,
    pub remaining: f32,
    pub decay_rate: f32,
    pub source: Option<String>,
}

impl ActiveEmotion {
    pub fn new(emotion_type: EmotionType, intensity: f32, duration: f32) -> Self {
        Self {
            emotion_type,
            intensity,
            max_intensity: intensity,
            duration,
            remaining: duration,
            decay_rate: DEFAULT_DECAY_RATE,
            source: None,
        }
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    /// Update the emotion: decay intensity and reduce remaining duration.
    pub fn update(&mut self, dt: f32) {
        self.remaining -= dt;
        // Exponential decay on intensity.
        self.intensity *= (-self.decay_rate * dt).exp();
        // Also decay as we approach the end of duration.
        if self.duration > 0.0 {
            let life_fraction = (self.remaining / self.duration).max(0.0);
            self.intensity = self.intensity.min(self.max_intensity * life_fraction);
        }
    }

    /// Whether this emotion has expired.
    pub fn is_expired(&self) -> bool {
        self.remaining <= 0.0 || self.intensity < 0.01
    }

    /// Get the PAD contribution of this active emotion.
    pub fn pad_contribution(&self) -> PadState {
        self.emotion_type.pad_center().scale(self.intensity)
    }
}

// ---------------------------------------------------------------------------
// Emotion Trigger
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EmotionTrigger {
    pub id: EmotionTriggerId,
    pub event_name: String,
    pub pad_impulse: PadState,
    pub intensity: f32,
    pub duration: f32,
    pub decay_rate: f32,
    pub cooldown: f32,
    pub last_triggered: f32,
    /// Personality sensitivity multiplier keys (trait_name -> multiplier).
    pub personality_modifiers: HashMap<String, f32>,
}

impl EmotionTrigger {
    pub fn new(id: EmotionTriggerId, event: &str, impulse: PadState, intensity: f32) -> Self {
        Self {
            id,
            event_name: event.to_string(),
            pad_impulse: impulse,
            intensity,
            duration: 10.0,
            decay_rate: DEFAULT_DECAY_RATE,
            cooldown: 1.0,
            last_triggered: -100.0,
            personality_modifiers: HashMap::new(),
        }
    }

    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = duration;
        self
    }

    pub fn with_cooldown(mut self, cooldown: f32) -> Self {
        self.cooldown = cooldown;
        self
    }

    pub fn with_personality_modifier(mut self, trait_name: &str, multiplier: f32) -> Self {
        self.personality_modifiers.insert(trait_name.to_string(), multiplier);
        self
    }

    pub fn can_trigger(&self, time: f32) -> bool {
        time - self.last_triggered >= self.cooldown
    }
}

// ---------------------------------------------------------------------------
// Personality Traits
// ---------------------------------------------------------------------------

/// Personality traits that modify how strongly emotions are felt and expressed.
/// Based on the Big Five (OCEAN) model.
#[derive(Debug, Clone)]
pub struct PersonalityTraits {
    /// Openness to experience (curiosity, creativity). High = stronger Awe, Surprise.
    pub openness: f32,
    /// Conscientiousness (organization, dependability). High = stronger Shame, Pride.
    pub conscientiousness: f32,
    /// Extraversion (sociability, energy). High = stronger Joy, Love; dampens Boredom.
    pub extraversion: f32,
    /// Agreeableness (cooperation, trust). High = stronger Trust, weaker Anger.
    pub agreeableness: f32,
    /// Neuroticism (emotional instability). High = stronger Fear, Sadness, Anger.
    pub neuroticism: f32,
}

impl PersonalityTraits {
    /// Create a "default" personality (all traits at 0.5).
    pub fn balanced() -> Self {
        Self {
            openness: 0.5,
            conscientiousness: 0.5,
            extraversion: 0.5,
            agreeableness: 0.5,
            neuroticism: 0.5,
        }
    }

    /// Create a brave/warrior personality.
    pub fn brave() -> Self {
        Self {
            openness: 0.4,
            conscientiousness: 0.7,
            extraversion: 0.6,
            agreeableness: 0.3,
            neuroticism: 0.2,
        }
    }

    /// Create a cowardly/nervous personality.
    pub fn cowardly() -> Self {
        Self {
            openness: 0.3,
            conscientiousness: 0.4,
            extraversion: 0.3,
            agreeableness: 0.6,
            neuroticism: 0.9,
        }
    }

    /// Create an aggressive personality.
    pub fn aggressive() -> Self {
        Self {
            openness: 0.3,
            conscientiousness: 0.3,
            extraversion: 0.7,
            agreeableness: 0.1,
            neuroticism: 0.6,
        }
    }

    /// Create a friendly/social personality.
    pub fn friendly() -> Self {
        Self {
            openness: 0.7,
            conscientiousness: 0.5,
            extraversion: 0.8,
            agreeableness: 0.9,
            neuroticism: 0.2,
        }
    }

    /// Get a PAD bias from personality (the "resting" emotional state).
    pub fn baseline_pad(&self) -> PadState {
        PadState::new(
            (self.extraversion - 0.5) * 0.4 + (self.agreeableness - 0.5) * 0.3 - (self.neuroticism - 0.5) * 0.3,
            (self.neuroticism - 0.5) * 0.4 + (self.extraversion - 0.5) * 0.2,
            (self.extraversion - 0.5) * 0.3 + (0.5 - self.agreeableness) * 0.2 + (0.5 - self.neuroticism) * 0.2,
        )
    }

    /// Get the intensity multiplier for a given emotion type.
    pub fn emotion_multiplier(&self, emotion: EmotionType) -> f32 {
        match emotion {
            EmotionType::Joy => 0.7 + self.extraversion * 0.6,
            EmotionType::Sadness => 0.5 + self.neuroticism * 0.8,
            EmotionType::Anger => 0.5 + self.neuroticism * 0.4 + (1.0 - self.agreeableness) * 0.4,
            EmotionType::Fear => 0.3 + self.neuroticism * 0.9 - (1.0 - self.agreeableness) * 0.2,
            EmotionType::Surprise => 0.5 + self.openness * 0.5,
            EmotionType::Disgust => 0.5 + (1.0 - self.openness) * 0.3 + self.neuroticism * 0.2,
            EmotionType::Trust => 0.4 + self.agreeableness * 0.6,
            EmotionType::Anticipation => 0.5 + self.openness * 0.3 + self.extraversion * 0.2,
            EmotionType::Contempt => 0.3 + (1.0 - self.agreeableness) * 0.5,
            EmotionType::Shame => 0.3 + self.conscientiousness * 0.5 + self.neuroticism * 0.3,
            EmotionType::Pride => 0.4 + self.conscientiousness * 0.4 + self.extraversion * 0.2,
            EmotionType::Love => 0.4 + self.agreeableness * 0.4 + self.extraversion * 0.2,
            EmotionType::Awe => 0.3 + self.openness * 0.7,
            EmotionType::Boredom => 0.3 + (1.0 - self.openness) * 0.4 + (1.0 - self.extraversion) * 0.3,
            EmotionType::Neutral => 1.0,
        }
    }

    /// Get the trait value by name (for trigger personality_modifiers).
    pub fn trait_value(&self, name: &str) -> f32 {
        match name {
            "openness" => self.openness,
            "conscientiousness" => self.conscientiousness,
            "extraversion" => self.extraversion,
            "agreeableness" => self.agreeableness,
            "neuroticism" => self.neuroticism,
            _ => 0.5,
        }
    }
}

impl Default for PersonalityTraits {
    fn default() -> Self {
        Self::balanced()
    }
}

// ---------------------------------------------------------------------------
// Facial Expression
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct FacialExpression {
    pub brow_raise: f32,
    pub brow_furrow: f32,
    pub eye_wide: f32,
    pub eye_squint: f32,
    pub mouth_smile: f32,
    pub mouth_frown: f32,
    pub mouth_open: f32,
    pub nostril_flare: f32,
    pub jaw_clench: f32,
    pub lip_tremble: f32,
    pub cheek_raise: f32,
    pub nose_wrinkle: f32,
}

impl Default for FacialExpression {
    fn default() -> Self {
        Self {
            brow_raise: 0.0,
            brow_furrow: 0.0,
            eye_wide: 0.0,
            eye_squint: 0.0,
            mouth_smile: 0.0,
            mouth_frown: 0.0,
            mouth_open: 0.0,
            nostril_flare: 0.0,
            jaw_clench: 0.0,
            lip_tremble: 0.0,
            cheek_raise: 0.0,
            nose_wrinkle: 0.0,
        }
    }
}

impl FacialExpression {
    pub fn from_emotion(emotion: EmotionType, intensity: f32) -> Self {
        let i = intensity.clamp(0.0, 1.0);
        match emotion {
            EmotionType::Joy => Self {
                mouth_smile: i,
                eye_squint: i * 0.3,
                brow_raise: i * 0.2,
                cheek_raise: i * 0.5,
                ..Default::default()
            },
            EmotionType::Sadness => Self {
                mouth_frown: i,
                brow_furrow: i * 0.5,
                eye_squint: i * 0.2,
                lip_tremble: i * 0.3,
                ..Default::default()
            },
            EmotionType::Anger => Self {
                brow_furrow: i,
                eye_squint: i * 0.4,
                nostril_flare: i * 0.5,
                mouth_frown: i * 0.3,
                jaw_clench: i * 0.6,
                ..Default::default()
            },
            EmotionType::Fear => Self {
                eye_wide: i,
                brow_raise: i * 0.7,
                mouth_open: i * 0.4,
                lip_tremble: i * 0.5,
                ..Default::default()
            },
            EmotionType::Surprise => Self {
                eye_wide: i,
                brow_raise: i,
                mouth_open: i * 0.6,
                ..Default::default()
            },
            EmotionType::Disgust => Self {
                nostril_flare: i * 0.6,
                mouth_frown: i * 0.5,
                brow_furrow: i * 0.3,
                nose_wrinkle: i * 0.7,
                ..Default::default()
            },
            EmotionType::Trust => Self {
                mouth_smile: i * 0.4,
                eye_squint: i * 0.1,
                cheek_raise: i * 0.2,
                ..Default::default()
            },
            EmotionType::Anticipation => Self {
                eye_wide: i * 0.3,
                brow_raise: i * 0.3,
                mouth_open: i * 0.1,
                ..Default::default()
            },
            EmotionType::Contempt => Self {
                mouth_smile: i * 0.2, // asymmetric smirk
                eye_squint: i * 0.3,
                brow_raise: i * 0.15,
                ..Default::default()
            },
            EmotionType::Shame => Self {
                brow_furrow: i * 0.3,
                eye_squint: i * 0.4,
                mouth_frown: i * 0.2,
                ..Default::default()
            },
            EmotionType::Pride => Self {
                brow_raise: i * 0.2,
                mouth_smile: i * 0.4,
                cheek_raise: i * 0.3,
                ..Default::default()
            },
            EmotionType::Love => Self {
                mouth_smile: i * 0.6,
                eye_squint: i * 0.2,
                cheek_raise: i * 0.4,
                ..Default::default()
            },
            EmotionType::Awe => Self {
                eye_wide: i * 0.6,
                mouth_open: i * 0.5,
                brow_raise: i * 0.6,
                ..Default::default()
            },
            EmotionType::Boredom => Self {
                eye_squint: i * 0.3,
                mouth_frown: i * 0.1,
                ..Default::default()
            },
            _ => Self::default(),
        }
    }

    /// Blend from two PAD-space emotion states to produce a combined expression.
    pub fn from_pad(pad: PadState) -> Self {
        // Determine the two closest emotions and blend their expressions.
        let emotions = EmotionType::all();
        let mut sorted: Vec<(EmotionType, f32)> = emotions.iter()
            .map(|&e| (e, pad.distance(e.pad_center())))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let magnitude = pad.magnitude().min(1.0);

        if sorted.len() >= 2 && sorted[0].1 < 2.0 {
            let (e1, d1) = sorted[0];
            let (e2, d2) = sorted[1];
            let total = d1 + d2;
            if total < 1e-9 {
                return Self::from_emotion(e1, magnitude);
            }
            let w1 = 1.0 - d1 / total;
            let w2 = 1.0 - d2 / total;
            let expr1 = Self::from_emotion(e1, magnitude * w1);
            let expr2 = Self::from_emotion(e2, magnitude * w2);
            expr1.lerp(&expr2, 0.5)
        } else if !sorted.is_empty() {
            Self::from_emotion(sorted[0].0, magnitude)
        } else {
            Self::default()
        }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            brow_raise: self.brow_raise + (other.brow_raise - self.brow_raise) * t,
            brow_furrow: self.brow_furrow + (other.brow_furrow - self.brow_furrow) * t,
            eye_wide: self.eye_wide + (other.eye_wide - self.eye_wide) * t,
            eye_squint: self.eye_squint + (other.eye_squint - self.eye_squint) * t,
            mouth_smile: self.mouth_smile + (other.mouth_smile - self.mouth_smile) * t,
            mouth_frown: self.mouth_frown + (other.mouth_frown - self.mouth_frown) * t,
            mouth_open: self.mouth_open + (other.mouth_open - self.mouth_open) * t,
            nostril_flare: self.nostril_flare + (other.nostril_flare - self.nostril_flare) * t,
            jaw_clench: self.jaw_clench + (other.jaw_clench - self.jaw_clench) * t,
            lip_tremble: self.lip_tremble + (other.lip_tremble - self.lip_tremble) * t,
            cheek_raise: self.cheek_raise + (other.cheek_raise - self.cheek_raise) * t,
            nose_wrinkle: self.nose_wrinkle + (other.nose_wrinkle - self.nose_wrinkle) * t,
        }
    }

    /// Convert to an array of blend shape weights for rendering.
    pub fn as_blend_shapes(&self) -> [f32; 12] {
        [
            self.brow_raise,
            self.brow_furrow,
            self.eye_wide,
            self.eye_squint,
            self.mouth_smile,
            self.mouth_frown,
            self.mouth_open,
            self.nostril_flare,
            self.jaw_clench,
            self.lip_tremble,
            self.cheek_raise,
            self.nose_wrinkle,
        ]
    }

    /// Get the dominant expression weight (largest blend shape value).
    pub fn dominant_weight(&self) -> f32 {
        let shapes = self.as_blend_shapes();
        shapes.iter().cloned().fold(0.0f32, f32::max)
    }
}

// ---------------------------------------------------------------------------
// Behavior Modifier
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BehaviorModifier {
    pub aggression: f32,
    pub caution: f32,
    pub sociability: f32,
    pub focus: f32,
    pub speed_multiplier: f32,
    pub accuracy_multiplier: f32,
    pub pain_tolerance: f32,
    pub flee_threshold: f32,
}

impl Default for BehaviorModifier {
    fn default() -> Self {
        Self {
            aggression: 0.0,
            caution: 0.0,
            sociability: 0.0,
            focus: 0.0,
            speed_multiplier: 1.0,
            accuracy_multiplier: 1.0,
            pain_tolerance: 0.5,
            flee_threshold: 0.3,
        }
    }
}

impl BehaviorModifier {
    pub fn from_pad(pad: PadState, personality: &PersonalityTraits) -> Self {
        let base_aggression = (-pad.pleasure * 0.5 + pad.arousal * 0.3 + pad.dominance * 0.3).clamp(-1.0, 1.0);
        let base_caution = (-pad.pleasure * 0.2 - pad.dominance * 0.5 + pad.arousal * 0.2).clamp(-1.0, 1.0);

        Self {
            aggression: (base_aggression * (1.0 + (1.0 - personality.agreeableness) * 0.5)).clamp(-1.0, 1.0),
            caution: (base_caution * (1.0 + personality.neuroticism * 0.5)).clamp(-1.0, 1.0),
            sociability: (pad.pleasure * 0.5 - pad.arousal * 0.1 + pad.dominance * 0.1
                + (personality.extraversion - 0.5) * 0.4).clamp(-1.0, 1.0),
            focus: (pad.arousal * 0.4 + pad.dominance * 0.2
                + (personality.conscientiousness - 0.5) * 0.3).clamp(-1.0, 1.0),
            speed_multiplier: 1.0 + pad.arousal * 0.2,
            accuracy_multiplier: 1.0 - pad.arousal * 0.15 + pad.dominance * 0.1,
            pain_tolerance: (0.5 + pad.dominance * 0.3 - personality.neuroticism * 0.2).clamp(0.0, 1.0),
            flee_threshold: (0.3 + personality.neuroticism * 0.3 - pad.dominance * 0.2).clamp(0.1, 0.9),
        }
    }

    /// Suggest the primary behavioral response.
    pub fn suggested_behavior(&self) -> SuggestedBehavior {
        if self.aggression > 0.7 {
            SuggestedBehavior::Attack
        } else if self.caution > 0.7 || self.aggression < -0.3 {
            SuggestedBehavior::Flee
        } else if self.sociability > 0.5 {
            SuggestedBehavior::Socialize
        } else if self.focus > 0.5 {
            SuggestedBehavior::Focus
        } else if self.caution > 0.3 {
            SuggestedBehavior::Defend
        } else {
            SuggestedBehavior::Idle
        }
    }
}

/// High-level behavior suggestion based on emotional state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestedBehavior {
    Attack,
    Flee,
    Defend,
    Socialize,
    Focus,
    Idle,
}

// ---------------------------------------------------------------------------
// Mood History
// ---------------------------------------------------------------------------

/// Tracks mood over time for computing long-term mood trends.
#[derive(Debug, Clone)]
struct MoodHistory {
    samples: Vec<(f32, PadState)>, // (timestamp, pad_state)
    max_samples: usize,
}

impl MoodHistory {
    fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::new(),
            max_samples,
        }
    }

    fn record(&mut self, time: f32, pad: PadState) {
        self.samples.push((time, pad));
        if self.samples.len() > self.max_samples {
            self.samples.remove(0);
        }
    }

    /// Compute the rolling average PAD state over the last `window` seconds.
    fn rolling_average(&self, current_time: f32, window: f32) -> PadState {
        let cutoff = current_time - window;
        let relevant: Vec<&(f32, PadState)> = self.samples.iter()
            .filter(|(t, _)| *t >= cutoff)
            .collect();

        if relevant.is_empty() {
            return PadState::NEUTRAL;
        }

        let mut sum_p = 0.0f32;
        let mut sum_a = 0.0f32;
        let mut sum_d = 0.0f32;
        for (_, pad) in &relevant {
            sum_p += pad.pleasure;
            sum_a += pad.arousal;
            sum_d += pad.dominance;
        }
        let n = relevant.len() as f32;
        PadState::new(sum_p / n, sum_a / n, sum_d / n)
    }

    /// Compute the trend: is the mood improving or worsening?
    fn pleasure_trend(&self, current_time: f32, window: f32) -> f32 {
        let cutoff = current_time - window;
        let relevant: Vec<&(f32, PadState)> = self.samples.iter()
            .filter(|(t, _)| *t >= cutoff)
            .collect();

        if relevant.len() < 2 {
            return 0.0;
        }

        // Simple linear regression on pleasure values.
        let n = relevant.len() as f32;
        let mut sum_t = 0.0f32;
        let mut sum_p = 0.0f32;
        let mut sum_tp = 0.0f32;
        let mut sum_tt = 0.0f32;
        for &(t, pad) in &relevant {
            sum_t += t;
            sum_p += pad.pleasure;
            sum_tp += t * pad.pleasure;
            sum_tt += t * t;
        }
        let denom = n * sum_tt - sum_t * sum_t;
        if denom.abs() < 1e-9 {
            return 0.0;
        }
        (n * sum_tp - sum_t * sum_p) / denom
    }
}

// ---------------------------------------------------------------------------
// Emotion Model
// ---------------------------------------------------------------------------

pub struct EmotionModel {
    current_pad: PadState,
    mood: PadState,
    triggers: HashMap<String, Vec<EmotionTrigger>>,
    active_emotions: Vec<ActiveEmotion>,
    decay_rate: f32,
    mood_blend_rate: f32,
    current_emotion: EmotionType,
    emotion_intensity: f32,
    facial_expression: FacialExpression,
    behavior_modifier: BehaviorModifier,
    personality: PersonalityTraits,
    mood_history: MoodHistory,
    time: f32,
    /// Sample interval for mood history (seconds).
    mood_sample_interval: f32,
    last_mood_sample: f32,
}

impl EmotionModel {
    pub fn new() -> Self {
        Self {
            current_pad: PadState::NEUTRAL,
            mood: PadState::NEUTRAL,
            triggers: HashMap::new(),
            active_emotions: Vec::new(),
            decay_rate: DEFAULT_DECAY_RATE,
            mood_blend_rate: DEFAULT_MOOD_BLEND,
            current_emotion: EmotionType::Neutral,
            emotion_intensity: 0.0,
            facial_expression: FacialExpression::default(),
            behavior_modifier: BehaviorModifier::default(),
            personality: PersonalityTraits::balanced(),
            mood_history: MoodHistory::new(MOOD_HISTORY_SIZE),
            time: 0.0,
            mood_sample_interval: 1.0,
            last_mood_sample: 0.0,
        }
    }

    pub fn with_personality(mut self, personality: PersonalityTraits) -> Self {
        self.personality = personality;
        self
    }

    pub fn set_personality(&mut self, personality: PersonalityTraits) {
        self.personality = personality;
    }

    pub fn personality(&self) -> &PersonalityTraits {
        &self.personality
    }

    pub fn set_personality_bias(&mut self, bias: PadState) {
        // Kept for backward compatibility; prefer set_personality.
        let _ = bias;
    }

    pub fn register_trigger(&mut self, trigger: EmotionTrigger) {
        self.triggers
            .entry(trigger.event_name.clone())
            .or_default()
            .push(trigger);
    }

    /// Trigger an event by name, which may activate one or more emotion triggers.
    pub fn trigger_event(&mut self, event: &str, intensity_scale: f32) {
        if let Some(triggers) = self.triggers.get_mut(event) {
            let mut impulses_to_add: Vec<(PadState, f32, f32)> = Vec::new();

            for trigger in triggers.iter_mut() {
                if trigger.can_trigger(self.time) {
                    // Apply personality modifiers to the trigger intensity.
                    let mut personality_multiplier = 1.0f32;
                    for (trait_name, &modifier) in &trigger.personality_modifiers {
                        let trait_val = self.personality.trait_value(trait_name);
                        personality_multiplier *= 1.0 + (trait_val - 0.5) * modifier;
                    }

                    let final_intensity = trigger.intensity * intensity_scale * personality_multiplier;
                    impulses_to_add.push((
                        trigger.pad_impulse.scale(final_intensity),
                        trigger.duration,
                        trigger.decay_rate,
                    ));
                    trigger.last_triggered = self.time;
                }
            }

            for (impulse, duration, decay_rate) in impulses_to_add {
                self.current_pad = self.current_pad.add(impulse).clamped();
                // Also add as an active emotion.
                let emotion = self.classify_pad(impulse);
                let mut active = ActiveEmotion::new(emotion, impulse.magnitude(), duration);
                active.decay_rate = decay_rate;
                active.source = Some(event.to_string());
                self.add_active_emotion(active);
            }
        }
    }

    /// Apply a direct PAD impulse.
    pub fn apply_impulse(&mut self, impulse: PadState) {
        self.current_pad = self.current_pad.add(impulse).clamped();
    }

    /// Apply a specific emotion with intensity and duration.
    pub fn apply_emotion(&mut self, emotion_type: EmotionType, intensity: f32, duration: f32) {
        let multiplier = self.personality.emotion_multiplier(emotion_type);
        let adjusted_intensity = intensity * multiplier;
        let impulse = emotion_type.pad_center().scale(adjusted_intensity);
        self.current_pad = self.current_pad.add(impulse).clamped();

        let mut active = ActiveEmotion::new(emotion_type, adjusted_intensity, duration);
        active.source = Some(format!("direct_{}", emotion_type));
        self.add_active_emotion(active);
    }

    /// Add an active emotion, respecting the maximum limit.
    fn add_active_emotion(&mut self, emotion: ActiveEmotion) {
        if self.active_emotions.len() >= MAX_ACTIVE_EMOTIONS {
            // Remove the weakest active emotion.
            if let Some(weakest_idx) = self.active_emotions.iter()
                .enumerate()
                .min_by(|a, b| a.1.intensity.partial_cmp(&b.1.intensity).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
            {
                if self.active_emotions[weakest_idx].intensity < emotion.intensity {
                    self.active_emotions.swap_remove(weakest_idx);
                    self.active_emotions.push(emotion);
                }
            }
        } else {
            self.active_emotions.push(emotion);
        }
    }

    /// Classify a PAD impulse to find the closest emotion type.
    fn classify_pad(&self, pad: PadState) -> EmotionType {
        let mut best = EmotionType::Neutral;
        let mut best_dist = f32::MAX;
        for &e in EmotionType::all() {
            let dist = pad.distance(e.pad_center());
            if dist < best_dist {
                best_dist = dist;
                best = e;
            }
        }
        best
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;

        // Update active emotions.
        for emotion in &mut self.active_emotions {
            emotion.update(dt);
        }
        self.active_emotions.retain(|e| !e.is_expired());

        // Compute the current PAD state from active emotions + baseline.
        let baseline = self.personality.baseline_pad();

        // Blend active emotion contributions.
        let active_contributions: Vec<(PadState, f32)> = self.active_emotions.iter()
            .map(|e| (e.pad_contribution(), e.intensity))
            .collect();

        if !active_contributions.is_empty() {
            let active_pad = PadState::weighted_blend(&active_contributions);
            // Mix active emotions with current state.
            self.current_pad = self.current_pad.lerp(
                self.current_pad.add(active_pad).clamped(),
                (dt * 2.0).min(1.0),
            );
        }

        // Decay toward personality baseline.
        self.current_pad = self.current_pad.decay_toward(baseline, self.decay_rate, dt);

        // Update mood (slowly tracks current state using exponential moving average).
        self.mood = self.mood.lerp(self.current_pad, 1.0 - (-self.mood_blend_rate * dt).exp());

        // Record mood history at intervals.
        if self.time - self.last_mood_sample >= self.mood_sample_interval {
            self.mood_history.record(self.time, self.current_pad);
            self.last_mood_sample = self.time;
        }

        // Classify emotion.
        self.classify_emotion();

        // Update facial expression (smooth interpolation).
        let target_expr = FacialExpression::from_pad(self.current_pad);
        self.facial_expression = self.facial_expression.lerp(&target_expr, (5.0 * dt).min(1.0));

        // Update behavior modifier.
        self.behavior_modifier = BehaviorModifier::from_pad(self.current_pad, &self.personality);
    }

    fn classify_emotion(&mut self) {
        let mut best = EmotionType::Neutral;
        let mut best_dist = f32::MAX;
        for &e in EmotionType::all() {
            let center = e.pad_center();
            let dist = self.current_pad.distance(center);
            if dist < best_dist {
                best_dist = dist;
                best = e;
            }
        }
        let mag = self.current_pad.magnitude();
        if mag < EMOTION_THRESHOLD {
            best = EmotionType::Neutral;
        }
        self.current_emotion = best;
        self.emotion_intensity = mag.min(1.0);
    }

    /// Get the current instantaneous PAD state.
    pub fn current_pad(&self) -> PadState {
        self.current_pad
    }

    /// Get the long-term mood PAD state.
    pub fn mood(&self) -> PadState {
        self.mood
    }

    /// Get the rolling average mood over the last N seconds.
    pub fn average_mood(&self, window_seconds: f32) -> PadState {
        self.mood_history.rolling_average(self.time, window_seconds)
    }

    /// Get the pleasure trend (positive = improving, negative = worsening).
    pub fn mood_trend(&self, window_seconds: f32) -> f32 {
        self.mood_history.pleasure_trend(self.time, window_seconds)
    }

    pub fn current_emotion(&self) -> EmotionType {
        self.current_emotion
    }

    pub fn emotion_intensity(&self) -> f32 {
        self.emotion_intensity
    }

    pub fn facial_expression(&self) -> &FacialExpression {
        &self.facial_expression
    }

    pub fn behavior_modifier(&self) -> &BehaviorModifier {
        &self.behavior_modifier
    }

    pub fn suggested_behavior(&self) -> SuggestedBehavior {
        self.behavior_modifier.suggested_behavior()
    }

    /// Get the list of currently active emotions.
    pub fn active_emotions(&self) -> &[ActiveEmotion] {
        &self.active_emotions
    }

    /// Get the strongest active emotion.
    pub fn strongest_emotion(&self) -> Option<&ActiveEmotion> {
        self.active_emotions.iter()
            .max_by(|a, b| a.intensity.partial_cmp(&b.intensity).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the blend shape array for rendering.
    pub fn blend_shapes(&self) -> [f32; 12] {
        self.facial_expression.as_blend_shapes()
    }

    /// Reset to neutral state.
    pub fn reset(&mut self) {
        self.current_pad = self.personality.baseline_pad();
        self.mood = self.current_pad;
        self.active_emotions.clear();
        self.current_emotion = EmotionType::Neutral;
        self.emotion_intensity = 0.0;
        self.facial_expression = FacialExpression::default();
        self.behavior_modifier = BehaviorModifier::default();
    }
}

// ---------------------------------------------------------------------------
// Social Emotion System (Group Morale / Emotion Spreading)
// ---------------------------------------------------------------------------

/// An entity tracked by the social emotion system.
pub struct SocialEntity {
    pub id: EntityId,
    pub model: EmotionModel,
    pub position: [f32; 3],
    pub group_id: Option<u32>,
}

/// Manages emotion spreading between nearby entities in a group.
pub struct SocialEmotionSystem {
    entities: Vec<SocialEntity>,
    contagion_radius: f32,
    contagion_strength: f32,
    /// Group morale per group_id.
    group_morale: HashMap<u32, f32>,
    update_interval: f32,
    update_timer: f32,
}

impl SocialEmotionSystem {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            contagion_radius: CONTAGION_DEFAULT_RADIUS,
            contagion_strength: CONTAGION_DEFAULT_STRENGTH,
            group_morale: HashMap::new(),
            update_interval: 0.5,
            update_timer: 0.0,
        }
    }

    pub fn set_contagion_params(&mut self, radius: f32, strength: f32) {
        self.contagion_radius = radius;
        self.contagion_strength = strength;
    }

    pub fn add_entity(&mut self, entity: SocialEntity) {
        self.entities.push(entity);
    }

    pub fn remove_entity(&mut self, id: EntityId) {
        self.entities.retain(|e| e.id != id);
    }

    pub fn get_entity(&self, id: EntityId) -> Option<&SocialEntity> {
        self.entities.iter().find(|e| e.id == id)
    }

    pub fn get_entity_mut(&mut self, id: EntityId) -> Option<&mut SocialEntity> {
        self.entities.iter_mut().find(|e| e.id == id)
    }

    pub fn update(&mut self, dt: f32) {
        // Update individual emotion models.
        for entity in &mut self.entities {
            entity.model.update(dt);
        }

        self.update_timer += dt;
        if self.update_timer < self.update_interval {
            return;
        }
        self.update_timer = 0.0;

        // Compute contagion: nearby entities influence each other.
        let radius_sq = self.contagion_radius * self.contagion_radius;
        let n = self.entities.len();

        // Collect impulses to apply (to avoid borrow issues).
        let mut impulses: Vec<PadState> = vec![PadState::NEUTRAL; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.entities[i].position[0] - self.entities[j].position[0];
                let dy = self.entities[i].position[1] - self.entities[j].position[1];
                let dz = self.entities[i].position[2] - self.entities[j].position[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq > radius_sq {
                    continue;
                }

                // Same group gets stronger contagion.
                let same_group = match (self.entities[i].group_id, self.entities[j].group_id) {
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                };
                let group_mult = if same_group { 2.0 } else { 0.5 };

                let dist = dist_sq.sqrt();
                let falloff = 1.0 - dist / self.contagion_radius;
                let strength = self.contagion_strength * falloff * group_mult;

                // i influences j and vice versa.
                let pad_i = self.entities[i].model.current_pad();
                let pad_j = self.entities[j].model.current_pad();

                impulses[j] = impulses[j].add(pad_i.sub(pad_j).scale(strength * self.update_interval));
                impulses[i] = impulses[i].add(pad_j.sub(pad_i).scale(strength * self.update_interval));
            }
        }

        // Apply impulses.
        for (i, impulse) in impulses.into_iter().enumerate() {
            if impulse.magnitude() > 0.001 {
                self.entities[i].model.apply_impulse(impulse);
            }
        }

        // Compute group morale.
        self.group_morale.clear();
        let mut group_counts: HashMap<u32, (f32, f32)> = HashMap::new(); // (sum_pleasure, count)
        for entity in &self.entities {
            if let Some(gid) = entity.group_id {
                let entry = group_counts.entry(gid).or_insert((0.0, 0.0));
                entry.0 += entity.model.current_pad().pleasure;
                entry.1 += 1.0;
            }
        }
        for (gid, (sum, count)) in &group_counts {
            self.group_morale.insert(*gid, sum / count);
        }
    }

    /// Get the morale for a group (average pleasure of all group members).
    pub fn group_morale(&self, group_id: u32) -> f32 {
        self.group_morale.get(&group_id).copied().unwrap_or(0.0)
    }

    /// Get the overall morale across all entities.
    pub fn overall_morale(&self) -> f32 {
        if self.entities.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.entities.iter()
            .map(|e| e.model.current_pad().pleasure)
            .sum();
        sum / self.entities.len() as f32
    }

    /// Apply a morale event to all entities in a group (e.g., "ally defeated").
    pub fn group_event(&mut self, group_id: u32, event: &str, intensity: f32) {
        for entity in &mut self.entities {
            if entity.group_id == Some(group_id) {
                entity.model.trigger_event(event, intensity);
            }
        }
    }

    /// Number of entities in the system.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}

impl Default for SocialEmotionSystem {
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

    #[test]
    fn test_pad_state() {
        let a = PadState::new(0.5, 0.3, 0.2);
        let b = PadState::new(0.1, 0.1, 0.1);
        let c = a.add(b);
        assert!((c.pleasure - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_pad_decay_toward() {
        let state = PadState::new(0.8, 0.5, 0.3);
        let target = PadState::NEUTRAL;
        let decayed = state.decay_toward(target, 1.0, 1.0);
        // Should be closer to neutral.
        assert!(decayed.magnitude() < state.magnitude());
    }

    #[test]
    fn test_pad_weighted_blend() {
        let states = vec![
            (PadState::new(1.0, 0.0, 0.0), 1.0),
            (PadState::new(-1.0, 0.0, 0.0), 1.0),
        ];
        let blended = PadState::weighted_blend(&states);
        assert!(blended.pleasure.abs() < 1e-6); // Should cancel out.
    }

    #[test]
    fn test_emotion_classification() {
        let mut model = EmotionModel::new();
        model.apply_impulse(PadState::new(0.8, 0.5, 0.5));
        model.update(0.016);
        assert_eq!(model.current_emotion(), EmotionType::Joy);
    }

    #[test]
    fn test_trigger() {
        let mut model = EmotionModel::new();
        model.register_trigger(
            EmotionTrigger::new(0, "damage_taken", PadState::new(-0.5, 0.7, -0.3), 1.0)
                .with_duration(5.0),
        );
        model.trigger_event("damage_taken", 1.0);
        model.update(0.016);
        assert!(model.current_pad().arousal > 0.0);
        assert!(!model.active_emotions().is_empty());
    }

    #[test]
    fn test_personality_multiplier() {
        let brave = PersonalityTraits::brave();
        let cowardly = PersonalityTraits::cowardly();

        let fear_brave = brave.emotion_multiplier(EmotionType::Fear);
        let fear_cowardly = cowardly.emotion_multiplier(EmotionType::Fear);
        assert!(fear_cowardly > fear_brave); // Cowardly feels fear more.
    }

    #[test]
    fn test_behavior_suggestion() {
        let angry = BehaviorModifier::from_pad(
            PadState::new(-0.6, 0.8, 0.6),
            &PersonalityTraits::aggressive(),
        );
        assert_eq!(angry.suggested_behavior(), SuggestedBehavior::Attack);

        let fearful = BehaviorModifier::from_pad(
            PadState::new(-0.7, 0.7, -0.7),
            &PersonalityTraits::cowardly(),
        );
        assert_eq!(fearful.suggested_behavior(), SuggestedBehavior::Flee);
    }

    #[test]
    fn test_active_emotion_decay() {
        let mut emotion = ActiveEmotion::new(EmotionType::Anger, 1.0, 5.0);
        emotion.update(3.0);
        assert!(emotion.intensity < 1.0);
        assert!(!emotion.is_expired());

        emotion.update(3.0);
        assert!(emotion.is_expired());
    }

    #[test]
    fn test_facial_expression_blend() {
        let joy = FacialExpression::from_emotion(EmotionType::Joy, 0.8);
        assert!(joy.mouth_smile > 0.5);

        let anger = FacialExpression::from_emotion(EmotionType::Anger, 0.9);
        assert!(anger.brow_furrow > 0.5);

        let blended = joy.lerp(&anger, 0.5);
        assert!(blended.mouth_smile > 0.0);
        assert!(blended.brow_furrow > 0.0);
    }

    #[test]
    fn test_social_emotion_system() {
        let mut system = SocialEmotionSystem::new();
        system.set_contagion_params(20.0, 0.5);

        let mut model_a = EmotionModel::new();
        model_a.apply_impulse(PadState::new(0.8, 0.3, 0.3)); // happy

        let model_b = EmotionModel::new(); // neutral

        system.add_entity(SocialEntity {
            id: 1,
            model: model_a,
            position: [0.0, 0.0, 0.0],
            group_id: Some(0),
        });
        system.add_entity(SocialEntity {
            id: 2,
            model: model_b,
            position: [5.0, 0.0, 0.0],
            group_id: Some(0),
        });

        // After some updates, entity B should become slightly happier due to contagion.
        for _ in 0..10 {
            system.update(0.5);
        }

        let entity_b = system.get_entity(2).unwrap();
        assert!(entity_b.model.current_pad().pleasure > 0.0);
    }

    #[test]
    fn test_blend_shapes() {
        let model = EmotionModel::new();
        let shapes = model.blend_shapes();
        assert_eq!(shapes.len(), 12);
    }

    #[test]
    fn test_mood_trend() {
        let mut model = EmotionModel::new();
        // Apply progressively happier states.
        for i in 0..20 {
            model.apply_impulse(PadState::new(0.05, 0.0, 0.0));
            model.update(1.0);
        }
        // Mood trend should be positive.
        let trend = model.mood_trend(15.0);
        assert!(trend >= 0.0);
    }
}
