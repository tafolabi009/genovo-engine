//! Digital Signal Processing (DSP) effects chain for the Genovo audio engine.
//!
//! Provides a modular effects system where individual DSP effects can be
//! chained together to process audio in sequence. Includes implementations
//! of common audio effects used in games:
//!
//! - **Filters**: Low-pass, high-pass, band-pass
//! - **Reverb**: Schroeder reverb (comb + allpass filters)
//! - **Delay**: Simple delay with feedback
//! - **Chorus**: Modulated delay for stereo width
//! - **Compressor**: Dynamic range compression
//! - **Equalizer**: Parametric EQ with multiple bands
//! - **Distortion**: Waveshaping / hard clipping
//! - **Tremolo**: Amplitude modulation
//! - **Phaser**: Allpass chain with modulated frequency
//! - **LFO**: Low-frequency oscillator for modulation

use std::f32::consts::TAU;

// ===========================================================================
// DspEffect trait
// ===========================================================================

/// Trait for audio DSP effects that process sample buffers.
///
/// All effects must be `Send + Sync` for use in audio processing threads.
/// The `process` method modifies samples in-place for zero-allocation
/// processing.
pub trait DspEffect: Send + Sync {
    /// Process a buffer of audio samples in-place.
    ///
    /// `samples` is a mono buffer of f32 samples in [-1, 1].
    /// `sample_rate` is provided for time-dependent effects.
    fn process(&mut self, samples: &mut [f32], sample_rate: u32);

    /// Get the name of this effect (for debugging/UI).
    fn name(&self) -> &str;

    /// Reset the effect's internal state (e.g., clear delay buffers).
    fn reset(&mut self);

    /// Whether this effect is currently bypassed (no processing).
    fn bypassed(&self) -> bool {
        false
    }
}

// ===========================================================================
// DspChain
// ===========================================================================

/// An ordered chain of DSP effects processed sequentially.
///
/// Each effect's output feeds into the next effect's input, forming a
/// signal processing pipeline. Effects can be individually bypassed.
pub struct DspChain {
    /// The effects in processing order.
    effects: Vec<Box<dyn DspEffect>>,
    /// Whether the entire chain is bypassed.
    pub bypassed: bool,
    /// Wet/dry mix [0, 1]. 0 = fully dry, 1 = fully wet.
    pub wet_mix: f32,
}

impl DspChain {
    /// Create a new empty DSP chain.
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
            bypassed: false,
            wet_mix: 1.0,
        }
    }

    /// Add an effect to the end of the chain.
    pub fn add(&mut self, effect: Box<dyn DspEffect>) {
        self.effects.push(effect);
    }

    /// Remove the effect at the given index.
    pub fn remove(&mut self, index: usize) {
        if index < self.effects.len() {
            self.effects.remove(index);
        }
    }

    /// Number of effects in the chain.
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Process samples through the entire chain.
    pub fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        if self.bypassed || self.effects.is_empty() {
            return;
        }

        if (self.wet_mix - 1.0).abs() < f32::EPSILON {
            // Fully wet: process in-place.
            for effect in &mut self.effects {
                if !effect.bypassed() {
                    effect.process(samples, sample_rate);
                }
            }
        } else {
            // Mix wet and dry: need a copy.
            let dry: Vec<f32> = samples.to_vec();
            for effect in &mut self.effects {
                if !effect.bypassed() {
                    effect.process(samples, sample_rate);
                }
            }
            // Blend.
            let dry_mix = 1.0 - self.wet_mix;
            for (i, s) in samples.iter_mut().enumerate() {
                *s = dry[i] * dry_mix + *s * self.wet_mix;
            }
        }
    }

    /// Reset all effects in the chain.
    pub fn reset(&mut self) {
        for effect in &mut self.effects {
            effect.reset();
        }
    }
}

impl Default for DspChain {
    fn default() -> Self {
        Self::new()
    }
}

// Allow Debug for DspChain even though dyn DspEffect doesn't impl Debug.
impl std::fmt::Debug for DspChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DspChain")
            .field("effect_count", &self.effects.len())
            .field("bypassed", &self.bypassed)
            .field("wet_mix", &self.wet_mix)
            .finish()
    }
}

// ===========================================================================
// LFO — Low Frequency Oscillator
// ===========================================================================

/// Waveform shape for the LFO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LfoWaveform {
    Sine,
    Triangle,
    Square,
    Sawtooth,
}

impl Default for LfoWaveform {
    fn default() -> Self {
        Self::Sine
    }
}

/// Low-frequency oscillator for modulating effect parameters.
///
/// Produces a periodic signal that can control filter cutoff, delay time,
/// amplitude, and other parameters. The output is in [-1, 1].
#[derive(Debug, Clone)]
pub struct Lfo {
    /// Oscillation frequency in Hz (typically 0.1 - 20 Hz).
    pub frequency: f32,
    /// Waveform shape.
    pub waveform: LfoWaveform,
    /// Current phase [0, 1).
    phase: f32,
    /// Depth/amplitude of the modulation [0, 1].
    pub depth: f32,
}

impl Lfo {
    /// Create a new LFO.
    pub fn new(frequency: f32, waveform: LfoWaveform) -> Self {
        Self {
            frequency,
            waveform,
            phase: 0.0,
            depth: 1.0,
        }
    }

    /// Advance the LFO by one sample and return the current value.
    pub fn tick(&mut self, sample_rate: u32) -> f32 {
        let value = self.evaluate();
        self.phase += self.frequency / sample_rate as f32;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        value * self.depth
    }

    /// Evaluate the LFO at the current phase without advancing.
    fn evaluate(&self) -> f32 {
        match self.waveform {
            LfoWaveform::Sine => (self.phase * TAU).sin(),
            LfoWaveform::Triangle => {
                let t = self.phase;
                if t < 0.25 {
                    t * 4.0
                } else if t < 0.75 {
                    2.0 - t * 4.0
                } else {
                    t * 4.0 - 4.0
                }
            }
            LfoWaveform::Square => {
                if self.phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            LfoWaveform::Sawtooth => 2.0 * self.phase - 1.0,
        }
    }

    /// Reset the LFO phase to zero.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

impl Default for Lfo {
    fn default() -> Self {
        Self::new(1.0, LfoWaveform::Sine)
    }
}

// ===========================================================================
// LowPassFilter
// ===========================================================================

/// First-order IIR low-pass filter.
///
/// Attenuates frequencies above the cutoff frequency. The transfer function is:
///
/// ```text
/// y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
/// ```
///
/// where `alpha = dt / (RC + dt)` and `RC = 1 / (2*pi*cutoff)`.
#[derive(Debug, Clone)]
pub struct LowPassFilter {
    /// Cutoff frequency in Hz.
    pub cutoff: f32,
    /// Previous output sample (filter state).
    prev_output: f32,
    /// Whether this filter is bypassed.
    pub bypass: bool,
}

impl LowPassFilter {
    /// Create a new low-pass filter with the given cutoff frequency.
    pub fn new(cutoff: f32) -> Self {
        Self {
            cutoff: cutoff.max(1.0),
            prev_output: 0.0,
            bypass: false,
        }
    }

    /// Compute the filter coefficient alpha for the given sample rate.
    fn alpha(&self, sample_rate: u32) -> f32 {
        let dt = 1.0 / sample_rate as f32;
        let rc = 1.0 / (TAU * self.cutoff);
        dt / (rc + dt)
    }
}

impl DspEffect for LowPassFilter {
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        let alpha = self.alpha(sample_rate);
        for sample in samples.iter_mut() {
            self.prev_output = alpha * *sample + (1.0 - alpha) * self.prev_output;
            *sample = self.prev_output;
        }
    }

    fn name(&self) -> &str {
        "LowPassFilter"
    }

    fn reset(&mut self) {
        self.prev_output = 0.0;
    }

    fn bypassed(&self) -> bool {
        self.bypass
    }
}

// ===========================================================================
// HighPassFilter
// ===========================================================================

/// First-order IIR high-pass filter.
///
/// Attenuates frequencies below the cutoff frequency. Implemented as the
/// complement of the low-pass filter.
///
/// ```text
/// y[n] = alpha * (y[n-1] + x[n] - x[n-1])
/// ```
#[derive(Debug, Clone)]
pub struct HighPassFilter {
    /// Cutoff frequency in Hz.
    pub cutoff: f32,
    /// Previous input sample.
    prev_input: f32,
    /// Previous output sample.
    prev_output: f32,
    /// Whether this filter is bypassed.
    pub bypass: bool,
}

impl HighPassFilter {
    /// Create a new high-pass filter.
    pub fn new(cutoff: f32) -> Self {
        Self {
            cutoff: cutoff.max(1.0),
            prev_input: 0.0,
            prev_output: 0.0,
            bypass: false,
        }
    }

    fn alpha(&self, sample_rate: u32) -> f32 {
        let dt = 1.0 / sample_rate as f32;
        let rc = 1.0 / (TAU * self.cutoff);
        rc / (rc + dt)
    }
}

impl DspEffect for HighPassFilter {
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        let alpha = self.alpha(sample_rate);
        for sample in samples.iter_mut() {
            let x = *sample;
            self.prev_output = alpha * (self.prev_output + x - self.prev_input);
            self.prev_input = x;
            *sample = self.prev_output;
        }
    }

    fn name(&self) -> &str {
        "HighPassFilter"
    }

    fn reset(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }

    fn bypassed(&self) -> bool {
        self.bypass
    }
}

// ===========================================================================
// BandPassFilter
// ===========================================================================

/// Band-pass filter implemented as cascaded low-pass and high-pass filters.
///
/// Passes frequencies between `low_cutoff` and `high_cutoff` while
/// attenuating frequencies outside that range.
#[derive(Debug, Clone)]
pub struct BandPassFilter {
    /// Low-pass component (sets the upper bound).
    low_pass: LowPassFilter,
    /// High-pass component (sets the lower bound).
    high_pass: HighPassFilter,
    /// Whether this filter is bypassed.
    pub bypass: bool,
}

impl BandPassFilter {
    /// Create a new band-pass filter.
    pub fn new(low_cutoff: f32, high_cutoff: f32) -> Self {
        Self {
            high_pass: HighPassFilter::new(low_cutoff),
            low_pass: LowPassFilter::new(high_cutoff),
            bypass: false,
        }
    }
}

impl DspEffect for BandPassFilter {
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        self.high_pass.process(samples, sample_rate);
        self.low_pass.process(samples, sample_rate);
    }

    fn name(&self) -> &str {
        "BandPassFilter"
    }

    fn reset(&mut self) {
        self.low_pass.reset();
        self.high_pass.reset();
    }

    fn bypassed(&self) -> bool {
        self.bypass
    }
}

// ===========================================================================
// Delay
// ===========================================================================

/// Simple delay effect with feedback.
///
/// ```text
/// y[n] = x[n] + feedback * buffer[write_pos - delay_samples]
/// ```
///
/// Creates echoes/repeats of the input signal.
#[derive(Debug, Clone)]
pub struct Delay {
    /// Delay time in seconds.
    pub delay_time: f32,
    /// Feedback amount [0, 1). Higher values = more repeats.
    /// Values >= 1.0 will cause runaway feedback (intentionally allowed
    /// for creative effects but use with caution).
    pub feedback: f32,
    /// Wet/dry mix [0, 1].
    pub mix: f32,
    /// Circular buffer for delayed samples.
    buffer: Vec<f32>,
    /// Current write position in the buffer.
    write_pos: usize,
    /// Whether this effect is bypassed.
    pub bypass: bool,
}

impl Delay {
    /// Create a new delay effect.
    pub fn new(delay_time: f32, feedback: f32) -> Self {
        Self {
            delay_time: delay_time.max(0.0),
            feedback: feedback.clamp(0.0, 0.99),
            mix: 0.5,
            buffer: vec![0.0; 192000], // Up to ~4s at 48kHz.
            write_pos: 0,
            bypass: false,
        }
    }

    /// Set the delay time. Does not clear the buffer.
    pub fn set_delay_time(&mut self, time: f32) {
        self.delay_time = time.max(0.0);
    }
}

impl DspEffect for Delay {
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        let delay_samples = (self.delay_time * sample_rate as f32) as usize;
        let buf_len = self.buffer.len();

        if delay_samples == 0 || delay_samples >= buf_len {
            return;
        }

        for sample in samples.iter_mut() {
            // Read from the delay buffer.
            let read_pos = (self.write_pos + buf_len - delay_samples) % buf_len;
            let delayed = self.buffer[read_pos];

            // Write input + feedback to the buffer.
            self.buffer[self.write_pos] = *sample + delayed * self.feedback;

            // Mix wet and dry.
            *sample = *sample * (1.0 - self.mix) + delayed * self.mix;

            self.write_pos = (self.write_pos + 1) % buf_len;
        }
    }

    fn name(&self) -> &str {
        "Delay"
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }

    fn bypassed(&self) -> bool {
        self.bypass
    }
}

// ===========================================================================
// Comb and Allpass filters (building blocks for Reverb)
// ===========================================================================

/// Comb filter for reverb implementation.
///
/// ```text
/// y[n] = x[n] + g * y[n - delay]
/// ```
#[derive(Debug, Clone)]
struct CombFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
    damp: f32,
    prev_output: f32,
}

impl CombFilter {
    fn new(delay_samples: usize, feedback: f32, damp: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            feedback,
            damp,
            prev_output: 0.0,
        }
    }

    fn process_sample(&mut self, input: f32) -> f32 {
        let buf_len = self.buffer.len();
        let output = self.buffer[self.write_pos];

        // Low-pass filtered feedback for dampening high frequencies.
        self.prev_output = output * (1.0 - self.damp) + self.prev_output * self.damp;
        self.buffer[self.write_pos] = input + self.prev_output * self.feedback;

        self.write_pos = (self.write_pos + 1) % buf_len;
        output
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.prev_output = 0.0;
    }
}

/// Allpass filter for reverb implementation.
///
/// ```text
/// y[n] = -g * x[n] + x[n - delay] + g * y[n - delay]
/// ```
#[derive(Debug, Clone)]
struct AllpassFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
}

impl AllpassFilter {
    fn new(delay_samples: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            feedback,
        }
    }

    fn process_sample(&mut self, input: f32) -> f32 {
        let buf_len = self.buffer.len();
        let delayed = self.buffer[self.write_pos];
        let output = -self.feedback * input + delayed;
        self.buffer[self.write_pos] = input + self.feedback * output;
        self.write_pos = (self.write_pos + 1) % buf_len;
        output
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

// ===========================================================================
// Reverb (Schroeder)
// ===========================================================================

/// Schroeder reverb effect.
///
/// Combines 4 parallel comb filters followed by 2 series allpass filters
/// to simulate room reverberation. This is a classic and computationally
/// inexpensive reverb algorithm suitable for games.
///
/// # Signal Flow
/// ```text
///            +--> Comb 1 --+
///  input --> +--> Comb 2 --+--> sum --> Allpass 1 --> Allpass 2 --> output
///            +--> Comb 3 --+
///            +--> Comb 4 --+
/// ```
#[derive(Debug, Clone)]
pub struct Reverb {
    /// Comb filters (parallel).
    combs: [CombFilter; 4],
    /// Allpass filters (series).
    allpasses: [AllpassFilter; 2],
    /// Pre-delay in samples.
    pre_delay_buffer: Vec<f32>,
    pre_delay_pos: usize,
    /// Pre-delay time in seconds.
    pub pre_delay: f32,
    /// Decay time in seconds (RT60).
    pub decay_time: f32,
    /// Wet/dry mix [0, 1].
    pub mix: f32,
    /// High-frequency damping [0, 1].
    pub damping: f32,
    /// Whether bypassed.
    pub bypass: bool,
}

impl Reverb {
    /// Create a new Schroeder reverb.
    ///
    /// * `decay_time` — approximate RT60 (time for the reverb to decay by 60dB)
    /// * `damping` — high-frequency absorption [0, 1]
    /// * `mix` — wet/dry mix [0, 1]
    pub fn new(decay_time: f32, damping: f32, mix: f32) -> Self {
        // Comb filter delay lengths in samples (at 44100 Hz).
        // These are prime-ish numbers chosen to avoid resonance.
        let comb_delays = [1557, 1617, 1491, 1422];
        let allpass_delays = [225, 556];

        let feedback = 0.84; // Approximate feedback for typical RT60.

        Self {
            combs: [
                CombFilter::new(comb_delays[0], feedback, damping),
                CombFilter::new(comb_delays[1], feedback, damping),
                CombFilter::new(comb_delays[2], feedback, damping),
                CombFilter::new(comb_delays[3], feedback, damping),
            ],
            allpasses: [
                AllpassFilter::new(allpass_delays[0], 0.5),
                AllpassFilter::new(allpass_delays[1], 0.5),
            ],
            pre_delay_buffer: vec![0.0; 48000], // 1s max pre-delay at 48kHz.
            pre_delay_pos: 0,
            pre_delay: 0.02,
            decay_time,
            mix: mix.clamp(0.0, 1.0),
            damping: damping.clamp(0.0, 1.0),
            bypass: false,
        }
    }

    /// Create a "small room" preset.
    pub fn small_room() -> Self {
        Self::new(0.5, 0.6, 0.3)
    }

    /// Create a "large hall" preset.
    pub fn large_hall() -> Self {
        Self::new(2.0, 0.3, 0.5)
    }

    /// Create a "cathedral" preset.
    pub fn cathedral() -> Self {
        Self::new(4.0, 0.2, 0.6)
    }
}

impl DspEffect for Reverb {
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        let pre_delay_samples = (self.pre_delay * sample_rate as f32) as usize;
        let pre_delay_samples = pre_delay_samples.min(self.pre_delay_buffer.len().saturating_sub(1));

        for sample in samples.iter_mut() {
            let dry = *sample;

            // Pre-delay.
            let pre_delayed = if pre_delay_samples > 0 {
                let read_pos = (self.pre_delay_pos + self.pre_delay_buffer.len() - pre_delay_samples)
                    % self.pre_delay_buffer.len();
                let delayed = self.pre_delay_buffer[read_pos];
                self.pre_delay_buffer[self.pre_delay_pos] = dry;
                self.pre_delay_pos = (self.pre_delay_pos + 1) % self.pre_delay_buffer.len();
                delayed
            } else {
                dry
            };

            // Parallel comb filters.
            let mut comb_sum = 0.0f32;
            for comb in &mut self.combs {
                comb_sum += comb.process_sample(pre_delayed);
            }
            comb_sum *= 0.25; // Average the four combs.

            // Series allpass filters.
            let mut output = comb_sum;
            for allpass in &mut self.allpasses {
                output = allpass.process_sample(output);
            }

            // Mix wet and dry.
            *sample = dry * (1.0 - self.mix) + output * self.mix;
        }
    }

    fn name(&self) -> &str {
        "Reverb"
    }

    fn reset(&mut self) {
        for comb in &mut self.combs {
            comb.reset();
        }
        for allpass in &mut self.allpasses {
            allpass.reset();
        }
        self.pre_delay_buffer.fill(0.0);
        self.pre_delay_pos = 0;
    }

    fn bypassed(&self) -> bool {
        self.bypass
    }
}

// ===========================================================================
// Chorus
// ===========================================================================

/// Chorus effect: modulated delay for stereo width and richness.
///
/// Creates a thickened, shimmering sound by mixing the dry signal with a
/// slightly delayed copy whose delay time is modulated by an LFO.
#[derive(Debug, Clone)]
pub struct Chorus {
    /// Base delay time in seconds (typically 10-30ms).
    pub base_delay: f32,
    /// Modulation depth in seconds (how much the delay varies).
    pub depth: f32,
    /// Wet/dry mix [0, 1].
    pub mix: f32,
    /// LFO for delay modulation.
    lfo: Lfo,
    /// Delay buffer.
    buffer: Vec<f32>,
    /// Write position.
    write_pos: usize,
    /// Whether bypassed.
    pub bypass: bool,
}

impl Chorus {
    /// Create a new chorus effect.
    pub fn new(rate: f32, depth: f32, mix: f32) -> Self {
        Self {
            base_delay: 0.02, // 20ms base delay.
            depth,
            mix: mix.clamp(0.0, 1.0),
            lfo: Lfo::new(rate, LfoWaveform::Sine),
            buffer: vec![0.0; 48000], // 1s at 48kHz.
            write_pos: 0,
            bypass: false,
        }
    }
}

impl DspEffect for Chorus {
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        let buf_len = self.buffer.len();

        for sample in samples.iter_mut() {
            let dry = *sample;

            // Write to delay buffer.
            self.buffer[self.write_pos] = dry;

            // Compute modulated delay time.
            let mod_val = self.lfo.tick(sample_rate);
            let delay_seconds = self.base_delay + self.depth * mod_val;
            let delay_samples = (delay_seconds * sample_rate as f32).max(0.0);

            // Read from delay buffer with linear interpolation.
            let delay_int = delay_samples as usize;
            let delay_frac = delay_samples - delay_int as f32;

            let read_pos_0 = (self.write_pos + buf_len - delay_int) % buf_len;
            let read_pos_1 = (read_pos_0 + buf_len - 1) % buf_len;

            let delayed = self.buffer[read_pos_0] * (1.0 - delay_frac)
                + self.buffer[read_pos_1] * delay_frac;

            // Mix.
            *sample = dry * (1.0 - self.mix) + delayed * self.mix;

            self.write_pos = (self.write_pos + 1) % buf_len;
        }
    }

    fn name(&self) -> &str {
        "Chorus"
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.lfo.reset();
    }

    fn bypassed(&self) -> bool {
        self.bypass
    }
}

// ===========================================================================
// Compressor
// ===========================================================================

/// Detection mode for the compressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMode {
    /// Peak level detection (reacts to instantaneous peaks).
    Peak,
    /// RMS level detection (reacts to average power, smoother).
    Rms,
}

impl Default for DetectionMode {
    fn default() -> Self {
        Self::Rms
    }
}

/// Dynamic range compressor.
///
/// Reduces the dynamic range of the audio signal by attenuating loud
/// portions above the threshold. Essential for preventing clipping and
/// maintaining consistent volume in game audio.
///
/// # Gain computation
///
/// For signals above the threshold:
/// ```text
/// gain_db = threshold + (input_db - threshold) / ratio
/// ```
///
/// The gain is smoothed by attack and release envelope followers.
#[derive(Debug, Clone)]
pub struct Compressor {
    /// Threshold in dB (signals above this are compressed).
    pub threshold: f32,
    /// Compression ratio (e.g., 4.0 means 4:1 compression).
    pub ratio: f32,
    /// Attack time in seconds (how fast compression engages).
    pub attack: f32,
    /// Release time in seconds (how fast compression releases).
    pub release: f32,
    /// Knee width in dB (0 = hard knee, >0 = soft knee).
    pub knee: f32,
    /// Makeup gain in dB (compensate for volume reduction).
    pub makeup_gain: f32,
    /// Detection mode (peak or RMS).
    pub detection: DetectionMode,
    /// Current envelope level (internal state).
    envelope: f32,
    /// RMS accumulator.
    rms_sum: f32,
    /// RMS window position.
    rms_count: u32,
    /// RMS window size.
    rms_window: u32,
    /// Whether bypassed.
    pub bypass: bool,
}

impl Compressor {
    /// Create a new compressor.
    pub fn new(threshold: f32, ratio: f32, attack: f32, release: f32) -> Self {
        Self {
            threshold,
            ratio: ratio.max(1.0),
            attack: attack.max(0.0001),
            release: release.max(0.0001),
            knee: 0.0,
            makeup_gain: 0.0,
            detection: DetectionMode::Rms,
            envelope: 0.0,
            rms_sum: 0.0,
            rms_count: 0,
            rms_window: 128,
            bypass: false,
        }
    }

    /// Convert a linear amplitude to decibels.
    #[inline]
    fn to_db(amplitude: f32) -> f32 {
        20.0 * amplitude.abs().max(1e-10).log10()
    }

    /// Convert decibels to a linear amplitude.
    #[inline]
    fn from_db(db: f32) -> f32 {
        10.0f32.powf(db / 20.0)
    }

    /// Compute the gain reduction for a given input level in dB.
    fn compute_gain(&self, input_db: f32) -> f32 {
        if self.knee > 0.0 {
            // Soft knee.
            let half_knee = self.knee / 2.0;
            if input_db < self.threshold - half_knee {
                0.0 // Below knee: no compression.
            } else if input_db > self.threshold + half_knee {
                // Above knee: full compression.
                self.threshold + (input_db - self.threshold) / self.ratio - input_db
            } else {
                // In the knee: smooth transition.
                let x = input_db - self.threshold + half_knee;
                let gain = x * x / (2.0 * self.knee);
                (gain / self.ratio) - gain
            }
        } else {
            // Hard knee.
            if input_db <= self.threshold {
                0.0
            } else {
                self.threshold + (input_db - self.threshold) / self.ratio - input_db
            }
        }
    }
}

impl DspEffect for Compressor {
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        let attack_coeff = (-1.0 / (self.attack * sample_rate as f32)).exp();
        let release_coeff = (-1.0 / (self.release * sample_rate as f32)).exp();
        let makeup_linear = Self::from_db(self.makeup_gain);

        for sample in samples.iter_mut() {
            // Detect level.
            let level = match self.detection {
                DetectionMode::Peak => sample.abs(),
                DetectionMode::Rms => {
                    self.rms_sum += *sample * *sample;
                    self.rms_count += 1;
                    if self.rms_count >= self.rms_window {
                        let rms = (self.rms_sum / self.rms_window as f32).sqrt();
                        self.rms_sum = 0.0;
                        self.rms_count = 0;
                        rms
                    } else {
                        (self.rms_sum / self.rms_count as f32).sqrt()
                    }
                }
            };

            let input_db = Self::to_db(level);
            let gain_reduction_db = self.compute_gain(input_db);

            // Envelope following with attack/release.
            let target_gain = Self::from_db(gain_reduction_db);
            let coeff = if target_gain < self.envelope {
                attack_coeff
            } else {
                release_coeff
            };
            self.envelope = coeff * self.envelope + (1.0 - coeff) * target_gain;

            // Apply gain.
            *sample *= Self::from_db(gain_reduction_db) * makeup_linear;
        }
    }

    fn name(&self) -> &str {
        "Compressor"
    }

    fn reset(&mut self) {
        self.envelope = 0.0;
        self.rms_sum = 0.0;
        self.rms_count = 0;
    }

    fn bypassed(&self) -> bool {
        self.bypass
    }
}

// ===========================================================================
// Equalizer
// ===========================================================================

/// A single band in the parametric equalizer.
#[derive(Debug, Clone)]
pub struct EqBand {
    /// Center frequency in Hz.
    pub frequency: f32,
    /// Gain in dB (-12 to +12 typical).
    pub gain: f32,
    /// Q factor (bandwidth). Higher Q = narrower band.
    pub q: f32,
    // Biquad filter coefficients.
    a0: f32, a1: f32, a2: f32,
    b0: f32, b1: f32, b2: f32,
    // Filter state.
    x1: f32, x2: f32,
    y1: f32, y2: f32,
}

impl EqBand {
    /// Create a new EQ band.
    pub fn new(frequency: f32, gain: f32, q: f32) -> Self {
        let mut band = Self {
            frequency,
            gain,
            q: q.max(0.1),
            a0: 1.0, a1: 0.0, a2: 0.0,
            b0: 1.0, b1: 0.0, b2: 0.0,
            x1: 0.0, x2: 0.0,
            y1: 0.0, y2: 0.0,
        };
        band.compute_coefficients(44100);
        band
    }

    /// Recompute biquad coefficients for the given sample rate.
    pub fn compute_coefficients(&mut self, sample_rate: u32) {
        let a = 10.0f32.powf(self.gain / 40.0);
        let w0 = TAU * self.frequency / sample_rate as f32;
        let sin_w0 = w0.sin();
        let cos_w0 = w0.cos();
        let alpha = sin_w0 / (2.0 * self.q);

        // Peaking EQ biquad coefficients.
        self.b0 = 1.0 + alpha * a;
        self.b1 = -2.0 * cos_w0;
        self.b2 = 1.0 - alpha * a;
        self.a0 = 1.0 + alpha / a;
        self.a1 = -2.0 * cos_w0;
        self.a2 = 1.0 - alpha / a;

        // Normalize.
        self.b0 /= self.a0;
        self.b1 /= self.a0;
        self.b2 /= self.a0;
        self.a1 /= self.a0;
        self.a2 /= self.a0;
        self.a0 = 1.0;
    }

    /// Process a single sample through this band.
    fn process_sample(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Parametric equalizer with multiple bands.
///
/// Each band is an independent peaking EQ filter that can boost or cut
/// a specific frequency range. Bands are processed in series.
#[derive(Debug, Clone)]
pub struct Equalizer {
    /// EQ bands.
    pub bands: Vec<EqBand>,
    /// Whether bypassed.
    pub bypass: bool,
    /// Whether coefficients need recomputation.
    dirty: bool,
}

impl Equalizer {
    /// Create a new equalizer with the given bands.
    pub fn new(bands: Vec<EqBand>) -> Self {
        Self {
            bands,
            bypass: false,
            dirty: false,
        }
    }

    /// Create a standard 3-band EQ (low, mid, high).
    pub fn three_band(low_gain: f32, mid_gain: f32, high_gain: f32) -> Self {
        Self::new(vec![
            EqBand::new(200.0, low_gain, 1.0),
            EqBand::new(1000.0, mid_gain, 1.0),
            EqBand::new(5000.0, high_gain, 1.0),
        ])
    }

    /// Add a band.
    pub fn add_band(&mut self, band: EqBand) {
        self.bands.push(band);
        self.dirty = true;
    }
}

impl DspEffect for Equalizer {
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        if self.dirty {
            for band in &mut self.bands {
                band.compute_coefficients(sample_rate);
            }
            self.dirty = false;
        }

        for sample in samples.iter_mut() {
            let mut s = *sample;
            for band in &mut self.bands {
                s = band.process_sample(s);
            }
            *sample = s;
        }
    }

    fn name(&self) -> &str {
        "Equalizer"
    }

    fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
    }

    fn bypassed(&self) -> bool {
        self.bypass
    }
}

// ===========================================================================
// Distortion
// ===========================================================================

/// Distortion effect using waveshaping.
///
/// Applies a nonlinear transfer function to the signal, adding harmonics
/// and creating a "crunchy" or "fuzzy" sound. Two modes are available:
///
/// - **Tanh**: Smooth soft clipping via `tanh(gain * x)` (warm overdrive)
/// - **HardClip**: Hard clipping at the threshold (harsh digital distortion)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionMode {
    /// Smooth waveshaping: `tanh(gain * x)`.
    Tanh,
    /// Hard clipping at `[-threshold, threshold]`.
    HardClip,
}

impl Default for DistortionMode {
    fn default() -> Self {
        Self::Tanh
    }
}

/// Distortion effect.
#[derive(Debug, Clone)]
pub struct Distortion {
    /// Drive/gain amount. Higher = more distortion.
    pub drive: f32,
    /// Distortion mode.
    pub mode: DistortionMode,
    /// Output level [0, 1] to compensate for increased loudness.
    pub output_level: f32,
    /// Wet/dry mix [0, 1].
    pub mix: f32,
    /// Whether bypassed.
    pub bypass: bool,
}

impl Distortion {
    /// Create a new distortion effect.
    pub fn new(drive: f32, mode: DistortionMode) -> Self {
        Self {
            drive: drive.max(1.0),
            mode,
            output_level: 0.7,
            mix: 1.0,
            bypass: false,
        }
    }
}

impl DspEffect for Distortion {
    fn process(&mut self, samples: &mut [f32], _sample_rate: u32) {
        for sample in samples.iter_mut() {
            let dry = *sample;
            let wet = match self.mode {
                DistortionMode::Tanh => (self.drive * *sample).tanh(),
                DistortionMode::HardClip => {
                    let amplified = *sample * self.drive;
                    amplified.clamp(-1.0, 1.0)
                }
            };
            *sample = (dry * (1.0 - self.mix) + wet * self.mix) * self.output_level;
        }
    }

    fn name(&self) -> &str {
        "Distortion"
    }

    fn reset(&mut self) {}

    fn bypassed(&self) -> bool {
        self.bypass
    }
}

// ===========================================================================
// Tremolo
// ===========================================================================

/// Tremolo effect: amplitude modulation with an LFO.
///
/// Varies the volume of the signal periodically, creating a pulsating
/// effect. Different waveforms produce different characters (sine for
/// smooth, square for choppy "stutter" effect).
#[derive(Debug, Clone)]
pub struct Tremolo {
    /// Modulation LFO.
    pub lfo: Lfo,
    /// Modulation depth [0, 1]. 1 = full silence at trough.
    pub depth: f32,
    /// Whether bypassed.
    pub bypass: bool,
}

impl Tremolo {
    /// Create a new tremolo effect.
    pub fn new(rate: f32, depth: f32) -> Self {
        Self {
            lfo: Lfo::new(rate, LfoWaveform::Sine),
            depth: depth.clamp(0.0, 1.0),
            bypass: false,
        }
    }

    /// Set the LFO waveform.
    pub fn with_waveform(mut self, waveform: LfoWaveform) -> Self {
        self.lfo.waveform = waveform;
        self
    }
}

impl DspEffect for Tremolo {
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        for sample in samples.iter_mut() {
            let mod_val = self.lfo.tick(sample_rate);
            // Map LFO [-1, 1] to amplitude [1 - depth, 1].
            let amplitude = 1.0 - self.depth * 0.5 * (1.0 + mod_val);
            *sample *= amplitude;
        }
    }

    fn name(&self) -> &str {
        "Tremolo"
    }

    fn reset(&mut self) {
        self.lfo.reset();
    }

    fn bypassed(&self) -> bool {
        self.bypass
    }
}

// ===========================================================================
// Phaser
// ===========================================================================

/// Phaser effect: allpass filter chain with modulated frequency.
///
/// Creates a sweeping "whoosh" sound by passing the signal through a
/// chain of allpass filters whose center frequency is modulated by an LFO.
/// The allpass-filtered signal is mixed with the dry signal, creating
/// notches that sweep through the frequency spectrum.
#[derive(Debug, Clone)]
pub struct Phaser {
    /// Number of allpass stages (more stages = deeper effect).
    pub stages: usize,
    /// Modulation LFO.
    lfo: Lfo,
    /// Minimum frequency of the sweep in Hz.
    pub min_frequency: f32,
    /// Maximum frequency of the sweep in Hz.
    pub max_frequency: f32,
    /// Feedback amount [-1, 1].
    pub feedback: f32,
    /// Wet/dry mix [0, 1].
    pub mix: f32,
    /// Allpass filter states (y[n-1] for each stage).
    allpass_states: Vec<f32>,
    /// Previous output for feedback.
    prev_output: f32,
    /// Whether bypassed.
    pub bypass: bool,
}

impl Phaser {
    /// Create a new phaser effect.
    pub fn new(rate: f32, stages: usize) -> Self {
        let stages = stages.clamp(2, 12);
        Self {
            stages,
            lfo: Lfo::new(rate, LfoWaveform::Sine),
            min_frequency: 200.0,
            max_frequency: 4000.0,
            feedback: 0.5,
            mix: 0.5,
            allpass_states: vec![0.0; stages],
            prev_output: 0.0,
            bypass: false,
        }
    }
}

impl DspEffect for Phaser {
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        for sample in samples.iter_mut() {
            let dry = *sample;

            // Compute the modulated allpass coefficient.
            let mod_val = self.lfo.tick(sample_rate);
            let frequency = self.min_frequency
                + (self.max_frequency - self.min_frequency) * 0.5 * (1.0 + mod_val);
            let w = TAU * frequency / sample_rate as f32;
            let coeff = (1.0 - w.sin()) / w.cos().max(0.0001);
            let coeff = coeff.clamp(-0.999, 0.999);

            // Process through allpass chain.
            let mut input = dry + self.prev_output * self.feedback;
            for i in 0..self.stages {
                let y = coeff * (input - self.allpass_states[i]) + input;
                // Simplified first-order allpass: use state as previous output.
                let old_state = self.allpass_states[i];
                self.allpass_states[i] = input;
                input = coeff * input + old_state * (1.0 - coeff * coeff);
                let _ = y; // We use the simplified form above.
            }

            self.prev_output = input;

            // Mix wet and dry.
            *sample = dry * (1.0 - self.mix) + input * self.mix;
        }
    }

    fn name(&self) -> &str {
        "Phaser"
    }

    fn reset(&mut self) {
        self.allpass_states.fill(0.0);
        self.prev_output = 0.0;
        self.lfo.reset();
    }

    fn bypassed(&self) -> bool {
        self.bypass
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a sine wave test signal.
    fn sine_signal(frequency: f32, sample_rate: u32, duration: f32) -> Vec<f32> {
        let n = (duration * sample_rate as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (TAU * frequency * t).sin()
            })
            .collect()
    }

    #[test]
    fn low_pass_attenuates_high_freq() {
        let mut filter = LowPassFilter::new(200.0);
        // Generate a high-frequency signal (5000 Hz).
        let mut signal = sine_signal(5000.0, 44100, 0.1);
        let original_energy: f32 = signal.iter().map(|s| s * s).sum();

        filter.process(&mut signal, 44100);
        let filtered_energy: f32 = signal.iter().map(|s| s * s).sum();

        assert!(
            filtered_energy < original_energy * 0.3,
            "Low-pass should attenuate 5000Hz with 200Hz cutoff: {filtered_energy} vs {original_energy}"
        );
    }

    #[test]
    fn low_pass_passes_low_freq() {
        let mut filter = LowPassFilter::new(5000.0);
        let mut signal = sine_signal(100.0, 44100, 0.1);
        let original_energy: f32 = signal.iter().map(|s| s * s).sum();

        filter.process(&mut signal, 44100);
        let filtered_energy: f32 = signal.iter().map(|s| s * s).sum();

        assert!(
            filtered_energy > original_energy * 0.7,
            "Low-pass should pass 100Hz with 5000Hz cutoff"
        );
    }

    #[test]
    fn high_pass_attenuates_low_freq() {
        let mut filter = HighPassFilter::new(5000.0);
        let mut signal = sine_signal(100.0, 44100, 0.1);
        let original_energy: f32 = signal.iter().map(|s| s * s).sum();

        filter.process(&mut signal, 44100);
        let filtered_energy: f32 = signal.iter().map(|s| s * s).sum();

        assert!(
            filtered_energy < original_energy * 0.3,
            "High-pass should attenuate 100Hz with 5000Hz cutoff"
        );
    }

    #[test]
    fn delay_produces_echo() {
        let mut delay = Delay::new(0.01, 0.0);
        delay.mix = 1.0; // Fully wet.
        let mut signal = vec![0.0f32; 2000];
        signal[0] = 1.0; // Impulse.

        delay.process(&mut signal, 44100);

        // The impulse should appear delayed by ~441 samples (0.01s * 44100).
        let delay_pos = 441;
        assert!(
            signal[delay_pos].abs() > 0.5,
            "Delayed impulse should appear at sample {delay_pos}: got {}",
            signal[delay_pos]
        );
    }

    #[test]
    fn delay_feedback() {
        let mut delay = Delay::new(0.01, 0.5);
        delay.mix = 1.0;
        let mut signal = vec![0.0f32; 4000];
        signal[0] = 1.0;

        delay.process(&mut signal, 44100);

        // First echo at ~441.
        let pos1 = 441;
        // Second echo (from feedback) at ~882.
        let pos2 = 882;
        assert!(signal[pos1].abs() > 0.5);
        assert!(signal[pos2].abs() > 0.1, "Feedback should produce a second echo");
    }

    #[test]
    fn reverb_produces_tail() {
        let mut reverb = Reverb::small_room();
        let mut signal = vec![0.0f32; 44100]; // 1 second.
        signal[0] = 1.0; // Impulse.

        reverb.process(&mut signal, 44100);

        // The reverb tail should have energy after the initial impulse.
        let tail_energy: f32 = signal[1000..10000].iter().map(|s| s * s).sum();
        assert!(
            tail_energy > 0.0001,
            "Reverb should produce a decaying tail"
        );
    }

    #[test]
    fn distortion_tanh() {
        let mut dist = Distortion::new(10.0, DistortionMode::Tanh);
        dist.mix = 1.0;
        let mut signal = sine_signal(440.0, 44100, 0.01);

        dist.process(&mut signal, 44100);

        // Tanh clipping should keep values in [-1, 1].
        for s in &signal {
            assert!(s.abs() <= 1.01, "Tanh distortion should stay in [-1, 1]");
        }
    }

    #[test]
    fn distortion_hard_clip() {
        let mut dist = Distortion::new(5.0, DistortionMode::HardClip);
        dist.mix = 1.0;
        dist.output_level = 1.0;
        let mut signal = sine_signal(440.0, 44100, 0.01);

        dist.process(&mut signal, 44100);

        for s in &signal {
            assert!(s.abs() <= 1.01, "Hard clip should stay in [-1, 1]");
        }
    }

    #[test]
    fn tremolo_modulates_amplitude() {
        let mut tremolo = Tremolo::new(10.0, 1.0);
        let mut signal = vec![1.0f32; 44100]; // Constant signal.

        tremolo.process(&mut signal, 44100);

        let min = signal.iter().cloned().fold(f32::MAX, f32::min);
        let max = signal.iter().cloned().fold(f32::MIN, f32::max);

        // With depth=1.0, the signal should vary between ~0 and ~1.
        assert!(min < 0.2, "Tremolo should reduce amplitude, min={min}");
        assert!(max > 0.8, "Tremolo should preserve peaks, max={max}");
    }

    #[test]
    fn dsp_chain_sequential() {
        let mut chain = DspChain::new();
        chain.add(Box::new(LowPassFilter::new(1000.0)));
        chain.add(Box::new(HighPassFilter::new(100.0)));

        let mut signal = sine_signal(500.0, 44100, 0.1);
        let original_energy: f32 = signal.iter().map(|s| s * s).sum();

        chain.process(&mut signal, 44100);
        let filtered_energy: f32 = signal.iter().map(|s| s * s).sum();

        // 500 Hz should pass through a 100-1000 Hz band-pass.
        assert!(
            filtered_energy > original_energy * 0.5,
            "500Hz should pass through 100-1000 Hz band"
        );
    }

    #[test]
    fn dsp_chain_bypass() {
        let mut chain = DspChain::new();
        chain.add(Box::new(LowPassFilter::new(10.0))); // Very aggressive filter.
        chain.bypassed = true;

        let mut signal = sine_signal(5000.0, 44100, 0.1);
        let original: Vec<f32> = signal.clone();

        chain.process(&mut signal, 44100);

        // When bypassed, signal should be unchanged.
        assert_eq!(signal, original);
    }

    #[test]
    fn dsp_chain_wet_dry_mix() {
        let mut chain = DspChain::new();
        chain.add(Box::new(Distortion::new(100.0, DistortionMode::HardClip)));
        chain.wet_mix = 0.0; // Fully dry.

        let mut signal = sine_signal(440.0, 44100, 0.01);
        let original: Vec<f32> = signal.clone();

        chain.process(&mut signal, 44100);

        // Fully dry should be unchanged.
        for (a, b) in signal.iter().zip(original.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn lfo_sine_range() {
        let mut lfo = Lfo::new(1.0, LfoWaveform::Sine);
        for _ in 0..44100 {
            let val = lfo.tick(44100);
            assert!(val >= -1.01 && val <= 1.01, "LFO value out of range: {val}");
        }
    }

    #[test]
    fn lfo_square() {
        let mut lfo = Lfo::new(1.0, LfoWaveform::Square);
        let val1 = lfo.tick(44100);
        // Square at phase 0 should be 1 or -1.
        assert!(val1.abs() > 0.9);
    }

    #[test]
    fn lfo_triangle() {
        let mut lfo = Lfo::new(1.0, LfoWaveform::Triangle);
        for _ in 0..44100 {
            let val = lfo.tick(44100);
            assert!(val >= -1.01 && val <= 1.01);
        }
    }

    #[test]
    fn compressor_reduces_loud_signal() {
        let mut comp = Compressor::new(-20.0, 4.0, 0.001, 0.01);
        comp.detection = DetectionMode::Peak;
        comp.makeup_gain = 0.0;

        let mut signal = sine_signal(440.0, 44100, 0.1);
        let original_peak: f32 = signal.iter().map(|s| s.abs()).fold(0.0, f32::max);

        comp.process(&mut signal, 44100);
        let compressed_peak: f32 = signal.iter().map(|s| s.abs()).fold(0.0, f32::max);

        assert!(
            compressed_peak < original_peak,
            "Compressor should reduce peak level: {compressed_peak} >= {original_peak}"
        );
    }

    #[test]
    fn equalizer_boost() {
        let mut eq = Equalizer::three_band(6.0, 0.0, 0.0);
        let mut signal = sine_signal(200.0, 44100, 0.1);
        let original_energy: f32 = signal.iter().map(|s| s * s).sum();

        eq.process(&mut signal, 44100);
        let boosted_energy: f32 = signal.iter().map(|s| s * s).sum();

        assert!(
            boosted_energy > original_energy,
            "EQ boost at 200Hz should increase energy of 200Hz signal"
        );
    }

    #[test]
    fn chorus_basic() {
        let mut chorus = Chorus::new(1.0, 0.002, 0.5);
        let mut signal = sine_signal(440.0, 44100, 0.1);

        chorus.process(&mut signal, 44100);

        // Should still have audio content.
        let energy: f32 = signal.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn phaser_basic() {
        let mut phaser = Phaser::new(0.5, 4);
        let mut signal = sine_signal(440.0, 44100, 0.1);

        phaser.process(&mut signal, 44100);

        let energy: f32 = signal.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn band_pass_filter() {
        let mut filter = BandPassFilter::new(400.0, 600.0);
        let mut signal = sine_signal(500.0, 44100, 0.1);
        let original_energy: f32 = signal.iter().map(|s| s * s).sum();

        filter.process(&mut signal, 44100);
        let filtered_energy: f32 = signal.iter().map(|s| s * s).sum();

        assert!(
            filtered_energy > original_energy * 0.3,
            "500Hz should partially pass through 400-600Hz band-pass"
        );
    }

    #[test]
    fn effect_reset() {
        let mut delay = Delay::new(0.01, 0.5);
        let mut signal = sine_signal(440.0, 44100, 0.01);
        delay.process(&mut signal, 44100);

        delay.reset();

        // After reset, processing silence should produce silence.
        let mut silence = vec![0.0f32; 1000];
        delay.process(&mut silence, 44100);
        let energy: f32 = silence.iter().map(|s| s * s).sum();
        assert!(energy < 0.0001, "After reset, delay should produce silence");
    }

    #[test]
    fn reverb_presets() {
        let _small = Reverb::small_room();
        let _large = Reverb::large_hall();
        let _cathedral = Reverb::cathedral();
        // Just verify they construct without panic.
    }
}
