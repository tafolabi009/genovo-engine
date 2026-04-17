// engine/audio/src/reverb_processor.rs
//
// Real reverb processor implementing a Schroeder reverb with:
//   - 4 parallel comb filters for late reverb
//   - 2 series allpass filters for diffusion
//   - Early reflections via a tapped delay line
//   - Room size, damping, pre-delay, and wet/dry mix controls
//   - Stereo output with decorrelation

// ---------------------------------------------------------------------------
// Delay line
// ---------------------------------------------------------------------------

/// A circular buffer delay line.
#[derive(Debug, Clone)]
pub struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
    length: usize,
}

impl DelayLine {
    pub fn new(max_length: usize) -> Self {
        Self {
            buffer: vec![0.0; max_length.max(1)],
            write_pos: 0,
            length: max_length.max(1),
        }
    }

    /// Write a sample and advance the write pointer.
    #[inline]
    pub fn write(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.length;
    }

    /// Read a sample at `delay` samples behind the write position.
    #[inline]
    pub fn read(&self, delay: usize) -> f32 {
        let idx = (self.write_pos + self.length - delay) % self.length;
        self.buffer[idx]
    }

    /// Read with fractional delay (linear interpolation).
    #[inline]
    pub fn read_fractional(&self, delay: f32) -> f32 {
        let delay_int = delay as usize;
        let frac = delay - delay_int as f32;
        let a = self.read(delay_int);
        let b = self.read(delay_int + 1);
        a + (b - a) * frac
    }

    /// Write and read in one step (for feedback loops).
    #[inline]
    pub fn process(&mut self, input: f32, delay: usize) -> f32 {
        let output = self.read(delay);
        self.write(input);
        output
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }

    /// Resize the delay line.
    pub fn resize(&mut self, new_length: usize) {
        let new_length = new_length.max(1);
        self.buffer.resize(new_length, 0.0);
        self.length = new_length;
        if self.write_pos >= self.length {
            self.write_pos = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Comb filter
// ---------------------------------------------------------------------------

/// A feedback comb filter with low-pass damping.
/// y[n] = x[n] + feedback * lpf(y[n - delay])
#[derive(Debug, Clone)]
pub struct CombFilter {
    delay_line: DelayLine,
    delay_samples: usize,
    feedback: f32,
    damping: f32,
    /// Low-pass filter state for damping.
    lp_state: f32,
}

impl CombFilter {
    pub fn new(delay_samples: usize, feedback: f32, damping: f32) -> Self {
        Self {
            delay_line: DelayLine::new(delay_samples + 1),
            delay_samples,
            feedback,
            damping,
            lp_state: 0.0,
        }
    }

    /// Process one sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let delayed = self.delay_line.read(self.delay_samples);

        // One-pole low-pass filter for damping.
        self.lp_state = delayed * (1.0 - self.damping) + self.lp_state * self.damping;

        let output = input + self.lp_state * self.feedback;
        self.delay_line.write(output);
        delayed
    }

    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback;
    }

    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping;
    }

    pub fn clear(&mut self) {
        self.delay_line.clear();
        self.lp_state = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Allpass filter
// ---------------------------------------------------------------------------

/// A Schroeder allpass filter.
/// y[n] = -g * x[n] + x[n - delay] + g * y[n - delay]
#[derive(Debug, Clone)]
pub struct AllpassFilter {
    delay_line: DelayLine,
    delay_samples: usize,
    gain: f32,
}

impl AllpassFilter {
    pub fn new(delay_samples: usize, gain: f32) -> Self {
        Self {
            delay_line: DelayLine::new(delay_samples + 1),
            delay_samples,
            gain,
        }
    }

    /// Process one sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let delayed = self.delay_line.read(self.delay_samples);
        let output = -self.gain * input + delayed + self.gain * delayed;
        self.delay_line.write(input + self.gain * delayed);
        output
    }

    pub fn clear(&mut self) {
        self.delay_line.clear();
    }
}

// ---------------------------------------------------------------------------
// Early reflections
// ---------------------------------------------------------------------------

/// Early reflections using a tapped delay line.
#[derive(Debug, Clone)]
pub struct EarlyReflections {
    delay_line: DelayLine,
    taps: Vec<ReflectionTap>,
    gain: f32,
}

/// A single reflection tap.
#[derive(Debug, Clone, Copy)]
pub struct ReflectionTap {
    pub delay_samples: usize,
    pub gain: f32,
}

impl EarlyReflections {
    pub fn new(max_delay: usize, sample_rate: u32) -> Self {
        // Default tap pattern simulating a small room.
        let ms_to_samples = |ms: f32| -> usize {
            (ms * sample_rate as f32 / 1000.0) as usize
        };

        let taps = vec![
            ReflectionTap { delay_samples: ms_to_samples(3.0), gain: 0.841 },
            ReflectionTap { delay_samples: ms_to_samples(5.5), gain: 0.504 },
            ReflectionTap { delay_samples: ms_to_samples(8.2), gain: -0.491 },
            ReflectionTap { delay_samples: ms_to_samples(11.7), gain: 0.379 },
            ReflectionTap { delay_samples: ms_to_samples(15.3), gain: -0.380 },
            ReflectionTap { delay_samples: ms_to_samples(19.8), gain: 0.346 },
            ReflectionTap { delay_samples: ms_to_samples(24.1), gain: -0.289 },
            ReflectionTap { delay_samples: ms_to_samples(29.0), gain: 0.272 },
            ReflectionTap { delay_samples: ms_to_samples(35.4), gain: -0.193 },
            ReflectionTap { delay_samples: ms_to_samples(42.6), gain: 0.174 },
            ReflectionTap { delay_samples: ms_to_samples(51.3), gain: -0.145 },
            ReflectionTap { delay_samples: ms_to_samples(63.0), gain: 0.111 },
        ];

        Self {
            delay_line: DelayLine::new(max_delay),
            taps,
            gain: 0.5,
        }
    }

    /// Process one sample.
    pub fn process(&mut self, input: f32) -> f32 {
        self.delay_line.write(input);
        let mut output = 0.0;
        for tap in &self.taps {
            if tap.delay_samples < self.delay_line.length {
                output += self.delay_line.read(tap.delay_samples) * tap.gain;
            }
        }
        output * self.gain
    }

    pub fn set_gain(&mut self, gain: f32) { self.gain = gain; }

    pub fn clear(&mut self) { self.delay_line.clear(); }
}

// ---------------------------------------------------------------------------
// Reverb parameters
// ---------------------------------------------------------------------------

/// Parameters for the reverb effect.
#[derive(Debug, Clone)]
pub struct ReverbParams {
    /// Room size (0.0 = small, 1.0 = large). Affects comb filter delay lengths.
    pub room_size: f32,
    /// Damping (0.0 = bright, 1.0 = dark). Controls HF absorption.
    pub damping: f32,
    /// Wet/dry mix (0.0 = fully dry, 1.0 = fully wet).
    pub mix: f32,
    /// Pre-delay in milliseconds.
    pub pre_delay_ms: f32,
    /// Early reflection gain (0.0 to 1.0).
    pub early_gain: f32,
    /// Late reverb gain (0.0 to 1.0).
    pub late_gain: f32,
    /// Stereo width (0.0 = mono, 1.0 = full stereo).
    pub width: f32,
    /// Decay time in seconds (approximate RT60).
    pub decay_time: f32,
    /// High-frequency damping frequency.
    pub hf_damping: f32,
}

impl Default for ReverbParams {
    fn default() -> Self {
        Self {
            room_size: 0.5,
            damping: 0.5,
            mix: 0.3,
            pre_delay_ms: 20.0,
            early_gain: 0.5,
            late_gain: 0.7,
            width: 1.0,
            decay_time: 1.5,
            hf_damping: 0.5,
        }
    }
}

/// Reverb presets.
impl ReverbParams {
    pub fn small_room() -> Self {
        Self { room_size: 0.2, damping: 0.7, mix: 0.25, pre_delay_ms: 5.0, decay_time: 0.5, ..Default::default() }
    }
    pub fn large_hall() -> Self {
        Self { room_size: 0.9, damping: 0.3, mix: 0.4, pre_delay_ms: 40.0, decay_time: 3.0, ..Default::default() }
    }
    pub fn cathedral() -> Self {
        Self { room_size: 1.0, damping: 0.15, mix: 0.5, pre_delay_ms: 60.0, decay_time: 5.0, width: 1.0, ..Default::default() }
    }
    pub fn bathroom() -> Self {
        Self { room_size: 0.15, damping: 0.3, mix: 0.4, pre_delay_ms: 3.0, decay_time: 0.8, ..Default::default() }
    }
    pub fn cave() -> Self {
        Self { room_size: 0.8, damping: 0.6, mix: 0.45, pre_delay_ms: 30.0, decay_time: 4.0, ..Default::default() }
    }
    pub fn plate() -> Self {
        Self { room_size: 0.6, damping: 0.2, mix: 0.35, pre_delay_ms: 10.0, decay_time: 2.0, width: 0.8, ..Default::default() }
    }
}

// ---------------------------------------------------------------------------
// ReverbProcessor
// ---------------------------------------------------------------------------

/// Base comb filter delay lengths (in samples at 44100 Hz).
const COMB_DELAYS: [usize; 4] = [1557, 1617, 1491, 1422];

/// Base allpass delay lengths.
const ALLPASS_DELAYS: [usize; 2] = [225, 556];

/// Stereo spread (offset for the second channel).
const STEREO_SPREAD: usize = 23;

/// Full Schroeder reverb processor with stereo output.
pub struct ReverbProcessor {
    sample_rate: u32,
    params: ReverbParams,
    // Left channel.
    combs_l: [CombFilter; 4],
    allpasses_l: [AllpassFilter; 2],
    // Right channel (offset delays for stereo).
    combs_r: [CombFilter; 4],
    allpasses_r: [AllpassFilter; 2],
    // Early reflections.
    early_l: EarlyReflections,
    early_r: EarlyReflections,
    // Pre-delay.
    pre_delay: DelayLine,
    pre_delay_samples: usize,
    // Output high-pass to remove DC.
    dc_block_l: f32,
    dc_block_r: f32,
    dc_prev_in_l: f32,
    dc_prev_in_r: f32,
}

impl ReverbProcessor {
    pub fn new(sample_rate: u32) -> Self {
        let params = ReverbParams::default();
        let scale = sample_rate as f32 / 44100.0;

        let make_comb = |base_delay: usize, offset: usize, p: &ReverbParams| -> CombFilter {
            let delay = ((base_delay + offset) as f32 * scale * (0.5 + p.room_size * 0.5)) as usize;
            let feedback = 0.84;
            CombFilter::new(delay.max(1), feedback, p.damping)
        };

        let make_allpass = |base_delay: usize, offset: usize| -> AllpassFilter {
            let delay = ((base_delay + offset) as f32 * scale) as usize;
            AllpassFilter::new(delay.max(1), 0.5)
        };

        let max_pre = (200.0 * sample_rate as f32 / 1000.0) as usize;
        let pre_delay_samples = (params.pre_delay_ms * sample_rate as f32 / 1000.0) as usize;

        let max_er = (100.0 * sample_rate as f32 / 1000.0) as usize;

        Self {
            sample_rate,
            params: params.clone(),
            combs_l: [
                make_comb(COMB_DELAYS[0], 0, &params),
                make_comb(COMB_DELAYS[1], 0, &params),
                make_comb(COMB_DELAYS[2], 0, &params),
                make_comb(COMB_DELAYS[3], 0, &params),
            ],
            allpasses_l: [
                make_allpass(ALLPASS_DELAYS[0], 0),
                make_allpass(ALLPASS_DELAYS[1], 0),
            ],
            combs_r: [
                make_comb(COMB_DELAYS[0], STEREO_SPREAD, &params),
                make_comb(COMB_DELAYS[1], STEREO_SPREAD, &params),
                make_comb(COMB_DELAYS[2], STEREO_SPREAD, &params),
                make_comb(COMB_DELAYS[3], STEREO_SPREAD, &params),
            ],
            allpasses_r: [
                make_allpass(ALLPASS_DELAYS[0], STEREO_SPREAD),
                make_allpass(ALLPASS_DELAYS[1], STEREO_SPREAD),
            ],
            early_l: EarlyReflections::new(max_er, sample_rate),
            early_r: EarlyReflections::new(max_er, sample_rate),
            pre_delay: DelayLine::new(max_pre.max(1)),
            pre_delay_samples: pre_delay_samples.min(max_pre),
            dc_block_l: 0.0,
            dc_block_r: 0.0,
            dc_prev_in_l: 0.0,
            dc_prev_in_r: 0.0,
        }
    }

    /// Set reverb parameters and update internal state.
    pub fn set_params(&mut self, params: ReverbParams) {
        let scale = self.sample_rate as f32 / 44100.0;

        // Update comb filter feedback based on decay time.
        // RT60 = -60dB / (20 * log10(feedback)) * delay_time
        for i in 0..4 {
            let delay_l = ((COMB_DELAYS[i]) as f32 * scale * (0.5 + params.room_size * 0.5)) as usize;
            let delay_r = ((COMB_DELAYS[i] + STEREO_SPREAD) as f32 * scale * (0.5 + params.room_size * 0.5)) as usize;

            let dt_l = delay_l as f32 / self.sample_rate as f32;
            let feedback = if params.decay_time > 0.0 && dt_l > 0.0 {
                10.0f32.powf(-3.0 * dt_l / params.decay_time)
            } else {
                0.84
            };

            self.combs_l[i].set_feedback(feedback.min(0.99));
            self.combs_l[i].set_damping(params.damping);
            self.combs_r[i].set_feedback(feedback.min(0.99));
            self.combs_r[i].set_damping(params.damping);
        }

        self.pre_delay_samples = (params.pre_delay_ms * self.sample_rate as f32 / 1000.0) as usize;
        self.pre_delay_samples = self.pre_delay_samples.min(self.pre_delay.length - 1);

        self.early_l.set_gain(params.early_gain);
        self.early_r.set_gain(params.early_gain);

        self.params = params;
    }

    /// Get current parameters.
    pub fn params(&self) -> &ReverbParams { &self.params }

    /// Process a mono input sample, returning (left, right) output.
    pub fn process_mono(&mut self, input: f32) -> (f32, f32) {
        // Pre-delay.
        let pre_delayed = self.pre_delay.process(input, self.pre_delay_samples);

        // Early reflections.
        let early_l = self.early_l.process(pre_delayed);
        let early_r = self.early_r.process(pre_delayed);

        // Late reverb: parallel comb filters.
        let mut late_l = 0.0f32;
        let mut late_r = 0.0f32;

        for comb in &mut self.combs_l {
            late_l += comb.process(pre_delayed);
        }
        for comb in &mut self.combs_r {
            late_r += comb.process(pre_delayed);
        }

        // Normalize comb output.
        late_l *= 0.25;
        late_r *= 0.25;

        // Series allpass filters for diffusion.
        for ap in &mut self.allpasses_l {
            late_l = ap.process(late_l);
        }
        for ap in &mut self.allpasses_r {
            late_r = ap.process(late_r);
        }

        // Combine early and late with gains.
        let wet_l = early_l * self.params.early_gain + late_l * self.params.late_gain;
        let wet_r = early_r * self.params.early_gain + late_r * self.params.late_gain;

        // Stereo width.
        let width = self.params.width;
        let mid = (wet_l + wet_r) * 0.5;
        let side = (wet_l - wet_r) * 0.5;
        let out_l = mid + side * width;
        let out_r = mid - side * width;

        // DC blocking filter.
        let dc_l = out_l - self.dc_prev_in_l + 0.995 * self.dc_block_l;
        self.dc_prev_in_l = out_l;
        self.dc_block_l = dc_l;

        let dc_r = out_r - self.dc_prev_in_r + 0.995 * self.dc_block_r;
        self.dc_prev_in_r = out_r;
        self.dc_block_r = dc_r;

        // Mix.
        let mix = self.params.mix;
        let final_l = input * (1.0 - mix) + dc_l * mix;
        let final_r = input * (1.0 - mix) + dc_r * mix;

        (final_l, final_r)
    }

    /// Process a stereo input, returning (left, right).
    pub fn process_stereo(&mut self, left: f32, right: f32) -> (f32, f32) {
        let mono = (left + right) * 0.5;
        let (rev_l, rev_r) = self.process_mono(mono);
        let mix = self.params.mix;
        (
            left * (1.0 - mix) + rev_l * mix,
            right * (1.0 - mix) + rev_r * mix,
        )
    }

    /// Process a buffer of mono samples in-place, outputting interleaved stereo.
    pub fn process_buffer_mono_to_stereo(&mut self, input: &[f32], output: &mut [f32]) {
        let samples = input.len();
        assert!(output.len() >= samples * 2);
        for i in 0..samples {
            let (l, r) = self.process_mono(input[i]);
            output[i * 2] = l;
            output[i * 2 + 1] = r;
        }
    }

    /// Process an interleaved stereo buffer in-place.
    pub fn process_buffer_stereo(&mut self, buffer: &mut [f32]) {
        let frames = buffer.len() / 2;
        for i in 0..frames {
            let l = buffer[i * 2];
            let r = buffer[i * 2 + 1];
            let (out_l, out_r) = self.process_stereo(l, r);
            buffer[i * 2] = out_l;
            buffer[i * 2 + 1] = out_r;
        }
    }

    /// Clear all internal state (for seamless preset changes).
    pub fn clear(&mut self) {
        for c in &mut self.combs_l { c.clear(); }
        for c in &mut self.combs_r { c.clear(); }
        for a in &mut self.allpasses_l { a.clear(); }
        for a in &mut self.allpasses_r { a.clear(); }
        self.early_l.clear();
        self.early_r.clear();
        self.pre_delay.clear();
        self.dc_block_l = 0.0;
        self.dc_block_r = 0.0;
        self.dc_prev_in_l = 0.0;
        self.dc_prev_in_r = 0.0;
    }

    /// Apply a preset.
    pub fn apply_preset(&mut self, preset: ReverbParams) {
        self.clear();
        self.set_params(preset);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_line() {
        let mut dl = DelayLine::new(10);
        dl.write(1.0);
        for _ in 0..9 { dl.write(0.0); }
        assert!((dl.read(9) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_comb_filter() {
        let mut comb = CombFilter::new(100, 0.7, 0.3);
        let mut output = 0.0f32;
        // Feed an impulse.
        output = comb.process(1.0);
        for _ in 0..200 {
            output = comb.process(0.0);
        }
        // After enough time, output should decay toward zero.
        assert!(output.abs() < 0.5);
    }

    #[test]
    fn test_allpass_filter() {
        let mut ap = AllpassFilter::new(50, 0.5);
        let _ = ap.process(1.0);
        for _ in 0..100 {
            let _ = ap.process(0.0);
        }
    }

    #[test]
    fn test_reverb_processor() {
        let mut reverb = ReverbProcessor::new(44100);
        reverb.set_params(ReverbParams::default());

        // Process an impulse.
        let (l, r) = reverb.process_mono(1.0);
        assert!(l.is_finite());
        assert!(r.is_finite());

        // Process silence and check that tail decays.
        let mut max_val = 0.0f32;
        for _ in 0..44100 {
            let (l, r) = reverb.process_mono(0.0);
            max_val = max_val.max(l.abs()).max(r.abs());
        }
        // After 1 second of silence, reverb tail should be quiet.
        let (l, r) = reverb.process_mono(0.0);
        assert!(l.abs() < 0.1, "reverb tail not decaying: {l}");
    }

    #[test]
    fn test_preset() {
        let mut reverb = ReverbProcessor::new(44100);
        reverb.apply_preset(ReverbParams::cathedral());
        let (l, r) = reverb.process_mono(1.0);
        assert!(l.is_finite() && r.is_finite());
    }

    #[test]
    fn test_buffer_processing() {
        let mut reverb = ReverbProcessor::new(44100);
        let input = vec![0.5f32; 256];
        let mut output = vec![0.0f32; 512];
        reverb.process_buffer_mono_to_stereo(&input, &mut output);
        assert!(output.iter().any(|&v| v != 0.0));
    }
}
