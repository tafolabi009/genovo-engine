// engine/ai/src/emotion_model.rs
//
// AI emotion system using the PAD (Pleasure-Arousal-Dominance) model.
//
// Provides emotion decay, emotion triggers (events to emotions), emotion-to-
// behavior mapping, facial expression output, and mood persistence.

use std::collections::HashMap;
use std::fmt;

pub type EmotionTriggerId = u32;

pub const PAD_MIN: f32 = -1.0;
pub const PAD_MAX: f32 = 1.0;
pub const DEFAULT_DECAY_RATE: f32 = 0.1;
pub const DEFAULT_MOOD_BLEND: f32 = 0.01;
pub const EMOTION_THRESHOLD: f32 = 0.15;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PadState {
    pub pleasure: f32,
    pub arousal: f32,
    pub dominance: f32,
}

impl PadState {
    pub const NEUTRAL: Self = Self { pleasure: 0.0, arousal: 0.0, dominance: 0.0 };
    pub fn new(p: f32, a: f32, d: f32) -> Self { Self { pleasure: p.clamp(PAD_MIN, PAD_MAX), arousal: a.clamp(PAD_MIN, PAD_MAX), dominance: d.clamp(PAD_MIN, PAD_MAX) } }
    pub fn lerp(self, other: Self, t: f32) -> Self { Self::new(self.pleasure + (other.pleasure - self.pleasure) * t, self.arousal + (other.arousal - self.arousal) * t, self.dominance + (other.dominance - self.dominance) * t) }
    pub fn add(self, other: Self) -> Self { Self::new(self.pleasure + other.pleasure, self.arousal + other.arousal, self.dominance + other.dominance) }
    pub fn scale(self, s: f32) -> Self { Self::new(self.pleasure * s, self.arousal * s, self.dominance * s) }
    pub fn magnitude(self) -> f32 { (self.pleasure * self.pleasure + self.arousal * self.arousal + self.dominance * self.dominance).sqrt() }
    pub fn decay(self, rate: f32, dt: f32) -> Self { let factor = (-rate * dt).exp(); self.scale(factor) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmotionType { Joy, Sadness, Anger, Fear, Surprise, Disgust, Trust, Anticipation, Contempt, Shame, Pride, Love, Awe, Boredom, Neutral }

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
}

impl fmt::Display for EmotionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Joy => write!(f, "Joy"), Self::Sadness => write!(f, "Sadness"),
            Self::Anger => write!(f, "Anger"), Self::Fear => write!(f, "Fear"),
            Self::Surprise => write!(f, "Surprise"), Self::Disgust => write!(f, "Disgust"),
            Self::Trust => write!(f, "Trust"), Self::Anticipation => write!(f, "Anticipation"),
            Self::Contempt => write!(f, "Contempt"), Self::Shame => write!(f, "Shame"),
            Self::Pride => write!(f, "Pride"), Self::Love => write!(f, "Love"),
            Self::Awe => write!(f, "Awe"), Self::Boredom => write!(f, "Boredom"),
            Self::Neutral => write!(f, "Neutral"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmotionTrigger {
    pub id: EmotionTriggerId,
    pub event_name: String,
    pub pad_impulse: PadState,
    pub intensity: f32,
    pub decay_rate: f32,
    pub cooldown: f32,
    pub last_triggered: f32,
}

impl EmotionTrigger {
    pub fn new(id: EmotionTriggerId, event: &str, impulse: PadState, intensity: f32) -> Self {
        Self { id, event_name: event.to_string(), pad_impulse: impulse, intensity, decay_rate: DEFAULT_DECAY_RATE, cooldown: 1.0, last_triggered: -100.0 }
    }
    pub fn can_trigger(&self, time: f32) -> bool { time - self.last_triggered >= self.cooldown }
}

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
}

impl Default for FacialExpression {
    fn default() -> Self { Self { brow_raise: 0.0, brow_furrow: 0.0, eye_wide: 0.0, eye_squint: 0.0, mouth_smile: 0.0, mouth_frown: 0.0, mouth_open: 0.0, nostril_flare: 0.0 } }
}

impl FacialExpression {
    pub fn from_emotion(emotion: EmotionType, intensity: f32) -> Self {
        let i = intensity.clamp(0.0, 1.0);
        match emotion {
            EmotionType::Joy => Self { mouth_smile: i, eye_squint: i * 0.3, brow_raise: i * 0.2, ..Default::default() },
            EmotionType::Sadness => Self { mouth_frown: i, brow_furrow: i * 0.5, eye_squint: i * 0.2, ..Default::default() },
            EmotionType::Anger => Self { brow_furrow: i, eye_squint: i * 0.4, nostril_flare: i * 0.5, mouth_frown: i * 0.3, ..Default::default() },
            EmotionType::Fear => Self { eye_wide: i, brow_raise: i * 0.7, mouth_open: i * 0.4, ..Default::default() },
            EmotionType::Surprise => Self { eye_wide: i, brow_raise: i, mouth_open: i * 0.6, ..Default::default() },
            EmotionType::Disgust => Self { nostril_flare: i * 0.6, mouth_frown: i * 0.5, brow_furrow: i * 0.3, ..Default::default() },
            _ => Self::default(),
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct BehaviorModifier {
    pub aggression: f32,
    pub caution: f32,
    pub sociability: f32,
    pub focus: f32,
    pub speed_multiplier: f32,
    pub accuracy_multiplier: f32,
}

impl Default for BehaviorModifier {
    fn default() -> Self { Self { aggression: 0.0, caution: 0.0, sociability: 0.0, focus: 0.0, speed_multiplier: 1.0, accuracy_multiplier: 1.0 } }
}

impl BehaviorModifier {
    pub fn from_pad(pad: PadState) -> Self {
        Self {
            aggression: (-pad.pleasure * 0.5 + pad.arousal * 0.3 + pad.dominance * 0.3).clamp(-1.0, 1.0),
            caution: (-pad.pleasure * 0.2 - pad.dominance * 0.5 + pad.arousal * 0.2).clamp(-1.0, 1.0),
            sociability: (pad.pleasure * 0.5 - pad.arousal * 0.1 + pad.dominance * 0.1).clamp(-1.0, 1.0),
            focus: (pad.arousal * 0.4 + pad.dominance * 0.2).clamp(-1.0, 1.0),
            speed_multiplier: 1.0 + pad.arousal * 0.2,
            accuracy_multiplier: 1.0 - pad.arousal * 0.15 + pad.dominance * 0.1,
        }
    }
}

pub struct EmotionModel {
    current_pad: PadState,
    mood: PadState,
    triggers: HashMap<String, Vec<EmotionTrigger>>,
    decay_rate: f32,
    mood_blend_rate: f32,
    current_emotion: EmotionType,
    emotion_intensity: f32,
    facial_expression: FacialExpression,
    behavior_modifier: BehaviorModifier,
    personality_bias: PadState,
    time: f32,
}

impl EmotionModel {
    pub fn new() -> Self {
        Self {
            current_pad: PadState::NEUTRAL, mood: PadState::NEUTRAL,
            triggers: HashMap::new(), decay_rate: DEFAULT_DECAY_RATE,
            mood_blend_rate: DEFAULT_MOOD_BLEND, current_emotion: EmotionType::Neutral,
            emotion_intensity: 0.0, facial_expression: FacialExpression::default(),
            behavior_modifier: BehaviorModifier::default(),
            personality_bias: PadState::NEUTRAL, time: 0.0,
        }
    }

    pub fn set_personality_bias(&mut self, bias: PadState) { self.personality_bias = bias; }
    pub fn register_trigger(&mut self, trigger: EmotionTrigger) { self.triggers.entry(trigger.event_name.clone()).or_default().push(trigger); }

    pub fn trigger_event(&mut self, event: &str, intensity_scale: f32) {
        if let Some(triggers) = self.triggers.get_mut(event) {
            for trigger in triggers.iter_mut() {
                if trigger.can_trigger(self.time) {
                    let impulse = trigger.pad_impulse.scale(trigger.intensity * intensity_scale);
                    self.current_pad = self.current_pad.add(impulse);
                    self.current_pad = PadState::new(self.current_pad.pleasure.clamp(PAD_MIN, PAD_MAX), self.current_pad.arousal.clamp(PAD_MIN, PAD_MAX), self.current_pad.dominance.clamp(PAD_MIN, PAD_MAX));
                    trigger.last_triggered = self.time;
                }
            }
        }
    }

    pub fn apply_impulse(&mut self, impulse: PadState) { self.current_pad = self.current_pad.add(impulse); }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        // Decay toward personality bias.
        let target = self.personality_bias;
        let diff = PadState::new(target.pleasure - self.current_pad.pleasure, target.arousal - self.current_pad.arousal, target.dominance - self.current_pad.dominance);
        let decay_amount = diff.scale(self.decay_rate * dt);
        self.current_pad = self.current_pad.add(decay_amount);
        // Update mood (slowly tracks current state).
        self.mood = self.mood.lerp(self.current_pad, self.mood_blend_rate);
        // Classify emotion.
        self.classify_emotion();
        // Update facial expression.
        let target_expr = FacialExpression::from_emotion(self.current_emotion, self.emotion_intensity);
        self.facial_expression = self.facial_expression.lerp(&target_expr, 5.0 * dt);
        // Update behavior modifier.
        self.behavior_modifier = BehaviorModifier::from_pad(self.current_pad);
    }

    fn classify_emotion(&mut self) {
        let emotions = [EmotionType::Joy, EmotionType::Sadness, EmotionType::Anger, EmotionType::Fear, EmotionType::Surprise, EmotionType::Disgust, EmotionType::Trust, EmotionType::Anticipation, EmotionType::Contempt, EmotionType::Shame, EmotionType::Pride, EmotionType::Love, EmotionType::Awe, EmotionType::Boredom];
        let mut best = EmotionType::Neutral;
        let mut best_dist = f32::MAX;
        for &e in &emotions {
            let center = e.pad_center();
            let dp = self.current_pad.pleasure - center.pleasure;
            let da = self.current_pad.arousal - center.arousal;
            let dd = self.current_pad.dominance - center.dominance;
            let dist = dp * dp + da * da + dd * dd;
            if dist < best_dist { best_dist = dist; best = e; }
        }
        let mag = self.current_pad.magnitude();
        if mag < EMOTION_THRESHOLD { best = EmotionType::Neutral; }
        self.current_emotion = best;
        self.emotion_intensity = mag.min(1.0);
    }

    pub fn current_pad(&self) -> PadState { self.current_pad }
    pub fn mood(&self) -> PadState { self.mood }
    pub fn current_emotion(&self) -> EmotionType { self.current_emotion }
    pub fn emotion_intensity(&self) -> f32 { self.emotion_intensity }
    pub fn facial_expression(&self) -> &FacialExpression { &self.facial_expression }
    pub fn behavior_modifier(&self) -> &BehaviorModifier { &self.behavior_modifier }
}

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
    fn test_emotion_classification() {
        let mut model = EmotionModel::new();
        model.apply_impulse(PadState::new(0.8, 0.5, 0.5));
        model.update(0.016);
        assert_eq!(model.current_emotion(), EmotionType::Joy);
    }

    #[test]
    fn test_trigger() {
        let mut model = EmotionModel::new();
        model.register_trigger(EmotionTrigger::new(0, "damage_taken", PadState::new(-0.5, 0.7, -0.3), 1.0));
        model.trigger_event("damage_taken", 1.0);
        model.update(0.016);
        assert!(model.current_pad().arousal > 0.0);
    }
}
