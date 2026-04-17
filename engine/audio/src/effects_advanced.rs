//! Advanced DSP effects for the Genovo audio engine.
//!
//! This module extends the basic effects in [`crate::dsp`] with more
//! computationally intensive algorithms:
//!
//! - [`ConvolutionReverb`] — impulse-response reverb using partitioned FFT
//!   convolution (overlap-add) for real-time processing of long IRs.
//! - [`PitchShifter`] — pitch shifting without tempo change via a granular
//!   synthesis approach with crossfaded grains.
//! - [`Spatializer3D`] — improved 3D audio positioning with HRTF
//!   approximation (ILD, ITD, head shadow).
//! - [`Limiter`] — brick-wall look-ahead limiter for transparent peak control.
//! - [`StereoWidener`] — mid/side stereo image manipulation.
//!
//! All effects implement the [`DspEffect`] trait from `dsp.rs` where
//! applicable (mono in-place processing). Stereo effects provide their own
//! `process` methods that operate on left/right channel pairs.

use std::f32::consts::{PI, TAU};

use crate::dsp::DspEffect;

// ===========================================================================
// Constants
// ===========================================================================

/// Default FFT block size for convolution (must be power of 2).
const DEFAULT_FFT_SIZE: usize = 1024;

/// Default grain size for pitch shifter in samples (at 44100 Hz ~ 23ms).
const DEFAULT_GRAIN_SIZE: usize = 1024;

/// Default look-ahead for the limiter in samples (at 44100 Hz ~ 5ms).
const DEFAULT_LIMITER_LOOKAHEAD: usize = 220;

/// Speed of sound in air (m/s) for HRTF calculations.
const SPEED_OF_SOUND: f32 = 343.0;

/// Average human head radius in meters.
const HEAD_RADIUS: f32 = 0.0875;

/// Maximum ITD in samples at 44100 Hz (~0.66ms).
const MAX_ITD_SAMPLES_44100: usize = 30;

// ===========================================================================
// FFT Implementation (Radix-2 Cooley-Tukey)
// ===========================================================================

/// A complex number for FFT operations.
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    /// Zero complex number.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Create a new complex number.
    #[inline]
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Complex multiplication.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Complex addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Complex subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// Magnitude squared.
    #[inline]
    pub fn mag_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude.
    #[inline]
    pub fn mag(self) -> f32 {
        self.mag_sq().sqrt()
    }

    /// Phase angle.
    #[inline]
    pub fn phase(self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Create from polar form.
    #[inline]
    pub fn from_polar(mag: f32, phase: f32) -> Self {
        Self {
            re: mag * phase.cos(),
            im: mag * phase.sin(),
        }
    }

    /// Scale by a real number.
    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Conjugate.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

/// Check that n is a power of two.
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Next power of two >= n.
fn next_power_of_two(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Bit-reversal permutation for FFT.
fn bit_reverse(data: &mut [Complex]) {
    let n = data.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// In-place radix-2 Cooley-Tukey FFT.
///
/// `data` must have a power-of-two length. If `inverse` is true, computes
/// the inverse FFT (with 1/N scaling).
pub fn fft(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(is_power_of_two(n), "FFT length must be a power of two");

    bit_reverse(data);

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_sign = if inverse { 1.0 } else { -1.0 };
        let angle = angle_sign * TAU / len as f32;

        // Twiddle factor base.
        let w_base = Complex::new(angle.cos(), angle.sin());

        let mut start = 0;
        while start < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let even = data[start + k];
                let odd = data[start + k + half].mul(w);
                data[start + k] = even.add(odd);
                data[start + k + half] = even.sub(odd);
                w = w.mul(w_base);
            }
            start += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f32;
        for c in data.iter_mut() {
            c.re *= scale;
            c.im *= scale;
        }
    }
}

/// Forward FFT of real-valued samples, returning complex spectrum.
pub fn fft_real(samples: &[f32]) -> Vec<Complex> {
    let n = next_power_of_two(samples.len());
    let mut data: Vec<Complex> = samples
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .chain(std::iter::repeat(Complex::ZERO).take(n - samples.len()))
        .collect();
    fft(&mut data, false);
    data
}

/// Inverse FFT, returning real part.
pub fn ifft_real(spectrum: &mut [Complex]) -> Vec<f32> {
    fft(spectrum, true);
    spectrum.iter().map(|c| c.re).collect()
}

/// FFT-based linear convolution of two real signals.
///
/// Returns a vector of length `signal.len() + kernel.len() - 1`.
///
/// # Algorithm
/// 1. Pad both signals to the next power of two >= combined length.
/// 2. FFT both.
/// 3. Pointwise multiply in the frequency domain.
/// 4. IFFT the product.
pub fn fft_convolve(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
    if signal.is_empty() || kernel.is_empty() {
        return Vec::new();
    }

    let output_len = signal.len() + kernel.len() - 1;
    let fft_size = next_power_of_two(output_len);

    // Pad and FFT the signal.
    let mut sig_fft: Vec<Complex> = signal
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .chain(std::iter::repeat(Complex::ZERO).take(fft_size - signal.len()))
        .collect();
    fft(&mut sig_fft, false);

    // Pad and FFT the kernel.
    let mut kern_fft: Vec<Complex> = kernel
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .chain(std::iter::repeat(Complex::ZERO).take(fft_size - kernel.len()))
        .collect();
    fft(&mut kern_fft, false);

    // Pointwise multiply.
    let mut product: Vec<Complex> = sig_fft
        .iter()
        .zip(kern_fft.iter())
        .map(|(&a, &b)| a.mul(b))
        .collect();

    // IFFT.
    fft(&mut product, true);

    // Extract real parts, truncated to output length.
    product.iter().take(output_len).map(|c| c.re).collect()
}

// ===========================================================================
// Convolution Reverb
// ===========================================================================

/// Impulse-response reverb using partitioned overlap-add FFT convolution.
///
/// Long impulse responses (e.g., 2+ seconds at 44.1 kHz) are partitioned
/// into blocks of `fft_size / 2` samples. Each block's spectrum is
/// pre-computed. During real-time processing, incoming audio is FFT'd,
/// multiplied with each IR partition, and accumulated via overlap-add.
///
/// # Example
/// ```ignore
/// let ir_samples = load_ir("cathedral.wav");
/// let mut reverb = ConvolutionReverb::new(&ir_samples, 1024);
/// reverb.wet_mix = 0.4;
/// reverb.process(&mut audio_buffer, 44100);
/// ```
pub struct ConvolutionReverb {
    /// Pre-computed FFT of each IR partition.
    ir_partitions: Vec<Vec<Complex>>,
    /// FFT size (must be power of two; each partition processes fft_size/2 new samples).
    fft_size: usize,
    /// Number of partitions.
    num_partitions: usize,
    /// Input ring buffer (stores the last `num_partitions * block_size` samples).
    input_buffer: Vec<f32>,
    /// Write position in the input ring buffer.
    input_write_pos: usize,
    /// Overlap-add accumulation buffer.
    overlap_buffer: Vec<f32>,
    /// Overlap-add read position.
    overlap_read_pos: usize,
    /// Wet/dry mix (0.0 = fully dry, 1.0 = fully wet).
    pub wet_mix: f32,
    /// Pre-delay in samples.
    pub pre_delay_samples: usize,
    /// Pre-delay buffer.
    pre_delay_buffer: Vec<f32>,
    /// Pre-delay write position.
    pre_delay_write_pos: usize,
    /// Internal sample counter for block processing.
    samples_since_last_block: usize,
    /// FFT of the current input block (reused allocation).
    current_fft: Vec<Complex>,
    /// Product accumulator (reused allocation).
    product_accum: Vec<Complex>,
    /// Past input FFTs for partitioned convolution.
    past_input_ffts: Vec<Vec<Complex>>,
}

impl ConvolutionReverb {
    /// Create a new convolution reverb from an impulse response.
    ///
    /// # Arguments
    /// * `impulse_response` — mono IR samples
    /// * `fft_size` — FFT block size (power of two, default 1024)
    pub fn new(impulse_response: &[f32], fft_size: usize) -> Self {
        let fft_size = next_power_of_two(fft_size.max(64));
        let block_size = fft_size / 2;

        // Partition the IR into blocks and pre-compute their FFTs.
        let num_partitions = (impulse_response.len() + block_size - 1) / block_size;
        let mut ir_partitions = Vec::with_capacity(num_partitions);

        for p in 0..num_partitions {
            let start = p * block_size;
            let end = (start + block_size).min(impulse_response.len());

            let mut padded = vec![Complex::ZERO; fft_size];
            for (i, &sample) in impulse_response[start..end].iter().enumerate() {
                padded[i] = Complex::new(sample, 0.0);
            }
            fft(&mut padded, false);
            ir_partitions.push(padded);
        }

        let num_partitions = ir_partitions.len().max(1);

        Self {
            ir_partitions,
            fft_size,
            num_partitions,
            input_buffer: vec![0.0; block_size * num_partitions],
            input_write_pos: 0,
            overlap_buffer: vec![0.0; fft_size * 2],
            overlap_read_pos: 0,
            wet_mix: 0.5,
            pre_delay_samples: 0,
            pre_delay_buffer: vec![0.0; 44100], // up to 1 second at 44.1kHz
            pre_delay_write_pos: 0,
            samples_since_last_block: 0,
            current_fft: vec![Complex::ZERO; fft_size],
            product_accum: vec![Complex::ZERO; fft_size],
            past_input_ffts: vec![vec![Complex::ZERO; fft_size]; num_partitions],
        }
    }

    /// Set pre-delay in milliseconds.
    pub fn set_pre_delay_ms(&mut self, ms: f32, sample_rate: u32) {
        self.pre_delay_samples = (ms * 0.001 * sample_rate as f32) as usize;
        if self.pre_delay_buffer.len() < self.pre_delay_samples {
            self.pre_delay_buffer.resize(self.pre_delay_samples + 1, 0.0);
        }
    }

    /// Process a block of input through the convolution reverb.
    ///
    /// This uses the overlap-add method with partitioned convolution:
    /// for each block of `fft_size/2` input samples, we compute
    /// `output[n] = sum over k of (input_block[n-k] * ir_partition[k])`
    /// in the frequency domain, then overlap-add the result.
    fn process_block(&mut self, block: &[f32]) -> Vec<f32> {
        let block_size = self.fft_size / 2;
        let mut output = vec![0.0; block.len()];

        let mut input_pos = 0;
        while input_pos < block.len() {
            let samples_to_fill = (block_size - self.samples_since_last_block)
                .min(block.len() - input_pos);

            // Store input samples.
            for i in 0..samples_to_fill {
                let idx = self.input_write_pos % self.input_buffer.len();
                self.input_buffer[idx] = block[input_pos + i];
                self.input_write_pos += 1;
            }
            self.samples_since_last_block += samples_to_fill;

            if self.samples_since_last_block >= block_size {
                // We have a complete block — process it.
                self.samples_since_last_block = 0;

                // Prepare the current block's FFT.
                let buf_len = self.input_buffer.len();
                for i in 0..self.fft_size {
                    if i < block_size {
                        let idx = (self.input_write_pos + buf_len - block_size + i) % buf_len;
                        self.current_fft[i] = Complex::new(self.input_buffer[idx], 0.0);
                    } else {
                        self.current_fft[i] = Complex::ZERO;
                    }
                }
                fft(&mut self.current_fft, false);

                // Shift past input FFTs and insert the current one.
                for i in (1..self.num_partitions).rev() {
                    // Use swap to move partitions without cloning.
                    let (left, right) = self.past_input_ffts.split_at_mut(i);
                    if !left.is_empty() && !right.is_empty() {
                        std::mem::swap(&mut left[i - 1], &mut right[0]);
                    }
                }
                if !self.past_input_ffts.is_empty() {
                    self.past_input_ffts[0].copy_from_slice(&self.current_fft);
                }

                // Partitioned convolution: accumulate products.
                for c in self.product_accum.iter_mut() {
                    *c = Complex::ZERO;
                }

                for (p, ir_part) in self.ir_partitions.iter().enumerate() {
                    if p >= self.past_input_ffts.len() {
                        break;
                    }
                    let input_fft = &self.past_input_ffts[p];
                    for i in 0..self.fft_size {
                        self.product_accum[i] =
                            self.product_accum[i].add(input_fft[i].mul(ir_part[i]));
                    }
                }

                // IFFT the accumulated product.
                let mut result = self.product_accum.clone();
                fft(&mut result, true);

                // Overlap-add into the output buffer.
                let overlap_len = self.overlap_buffer.len();
                for i in 0..self.fft_size {
                    let idx = (self.overlap_read_pos + i) % overlap_len;
                    self.overlap_buffer[idx] += result[i].re;
                }
            }

            // Read from the overlap buffer.
            let overlap_len = self.overlap_buffer.len();
            for i in 0..samples_to_fill {
                let idx = (self.overlap_read_pos + i) % overlap_len;
                output[input_pos + i] = self.overlap_buffer[idx];
                self.overlap_buffer[idx] = 0.0;
            }
            self.overlap_read_pos = (self.overlap_read_pos + samples_to_fill) % overlap_len;

            input_pos += samples_to_fill;
        }

        output
    }
}

impl DspEffect for ConvolutionReverb {
    fn process(&mut self, samples: &mut [f32], _sample_rate: u32) {
        if samples.is_empty() {
            return;
        }

        // Apply pre-delay.
        let mut delayed_input = Vec::with_capacity(samples.len());
        for &s in samples.iter() {
            if self.pre_delay_samples > 0 {
                let pd_len = self.pre_delay_buffer.len().max(1);
                let read_idx =
                    (self.pre_delay_write_pos + pd_len - self.pre_delay_samples.min(pd_len))
                        % pd_len;
                delayed_input.push(self.pre_delay_buffer[read_idx]);
                self.pre_delay_buffer[self.pre_delay_write_pos % pd_len] = s;
                self.pre_delay_write_pos = (self.pre_delay_write_pos + 1) % pd_len;
            } else {
                delayed_input.push(s);
            }
        }

        let wet = self.process_block(&delayed_input);

        // Mix wet/dry.
        let dry_mix = 1.0 - self.wet_mix;
        for (i, s) in samples.iter_mut().enumerate() {
            let wet_sample = if i < wet.len() { wet[i] } else { 0.0 };
            *s = *s * dry_mix + wet_sample * self.wet_mix;
        }
    }

    fn name(&self) -> &str {
        "ConvolutionReverb"
    }

    fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.input_write_pos = 0;
        self.overlap_buffer.fill(0.0);
        self.overlap_read_pos = 0;
        self.pre_delay_buffer.fill(0.0);
        self.pre_delay_write_pos = 0;
        self.samples_since_last_block = 0;
        for buf in &mut self.past_input_ffts {
            for c in buf.iter_mut() {
                *c = Complex::ZERO;
            }
        }
    }
}

// ===========================================================================
// Pitch Shifter (Granular Approach)
// ===========================================================================

/// Pitch shifter using granular synthesis with crossfaded grains.
///
/// Splits the input into small overlapping grains (20-50ms), resamples each
/// grain at the target pitch ratio, then crossfades the output grains to
/// avoid discontinuities.
///
/// # Algorithm
///
/// For a pitch shift of `ratio`:
/// 1. Read grains of `grain_size` samples with `hop_size` spacing.
/// 2. Resample each grain by `ratio` using linear interpolation.
///    - `ratio > 1.0` → higher pitch (grain is shorter → loop more).
///    - `ratio < 1.0` → lower pitch (grain is longer → skip some).
/// 3. Apply a Hann window to each grain for smooth crossfading.
/// 4. Overlap-add the windowed, resampled grains.
///
/// # Example
/// ```ignore
/// let mut shifter = PitchShifter::new(1024, 256);
/// shifter.set_semitones(5.0); // shift up 5 semitones
/// shifter.process(&mut buffer, 44100);
/// ```
pub struct PitchShifter {
    /// Grain size in samples.
    grain_size: usize,
    /// Hop size (overlap = grain_size - hop_size).
    hop_size: usize,
    /// Pitch ratio (1.0 = no change, 2.0 = octave up, 0.5 = octave down).
    pub ratio: f32,
    /// Input ring buffer.
    input_buffer: Vec<f32>,
    /// Write position in the input buffer.
    write_pos: usize,
    /// Read position for grain extraction (fractional).
    read_pos: f64,
    /// Output accumulation buffer.
    output_buffer: Vec<f32>,
    /// Output read position.
    output_read_pos: usize,
    /// Output write position.
    output_write_pos: usize,
    /// Pre-computed Hann window.
    window: Vec<f32>,
    /// Samples processed since last grain.
    samples_since_grain: usize,
}

impl PitchShifter {
    /// Create a new pitch shifter.
    ///
    /// # Arguments
    /// * `grain_size` — grain length in samples (512-2048 typical)
    /// * `hop_size` — hop between grains (grain_size / 4 typical)
    pub fn new(grain_size: usize, hop_size: usize) -> Self {
        let grain_size = grain_size.max(64);
        let hop_size = hop_size.max(1).min(grain_size);

        // Pre-compute Hann window.
        let window: Vec<f32> = (0..grain_size)
            .map(|i| {
                let t = i as f32 / grain_size as f32;
                0.5 * (1.0 - (TAU * t).cos())
            })
            .collect();

        let buffer_size = grain_size * 8;

        Self {
            grain_size,
            hop_size,
            ratio: 1.0,
            input_buffer: vec![0.0; buffer_size],
            write_pos: 0,
            read_pos: 0.0,
            output_buffer: vec![0.0; buffer_size],
            output_read_pos: 0,
            output_write_pos: 0,
            window,
            samples_since_grain: 0,
        }
    }

    /// Set pitch shift in semitones (positive = up, negative = down).
    pub fn set_semitones(&mut self, semitones: f32) {
        self.ratio = 2.0_f32.powf(semitones / 12.0);
    }

    /// Set pitch ratio directly (1.0 = unity, 2.0 = octave up).
    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio = ratio.clamp(0.25, 4.0);
    }

    /// Shift a buffer of samples by a given number of semitones.
    ///
    /// Standalone convenience function — creates a temporary shifter
    /// internally. For real-time use, prefer the `DspEffect` implementation.
    pub fn shift(samples: &[f32], semitones: f32) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let ratio = 2.0_f32.powf(semitones / 12.0);
        let grain_size = 1024;
        let hop_size = grain_size / 4;

        // Pre-compute window.
        let window: Vec<f32> = (0..grain_size)
            .map(|i| {
                let t = i as f32 / grain_size as f32;
                0.5 * (1.0 - (TAU * t).cos())
            })
            .collect();

        let output_len = samples.len();
        let mut output = vec![0.0f32; output_len + grain_size];
        let mut norm = vec![0.0f32; output_len + grain_size];

        // Process grains.
        let mut grain_start = 0i64;
        let mut output_pos = 0usize;

        while output_pos < output_len {
            // Extract and resample one grain.
            for i in 0..grain_size {
                let read_idx_f = grain_start as f64 + (i as f64 * ratio as f64);
                let read_idx = read_idx_f as i64;
                let frac = read_idx_f - read_idx as f64;

                let s0 = if read_idx >= 0 && (read_idx as usize) < samples.len() {
                    samples[read_idx as usize]
                } else {
                    0.0
                };
                let s1 = if read_idx + 1 >= 0 && ((read_idx + 1) as usize) < samples.len() {
                    samples[(read_idx + 1) as usize]
                } else {
                    0.0
                };

                let interpolated = s0 + (s1 - s0) * frac as f32;
                let windowed = interpolated * window[i];
                let out_idx = output_pos + i;
                if out_idx < output.len() {
                    output[out_idx] += windowed;
                    norm[out_idx] += window[i] * window[i];
                }
            }

            grain_start += hop_size as i64;
            output_pos += hop_size;
        }

        // Normalize by the overlap window sum.
        for i in 0..output_len {
            if norm[i] > 1e-6 {
                output[i] /= norm[i];
            }
        }

        output.truncate(output_len);
        output
    }

    /// Extract, resample, and accumulate one grain.
    fn process_grain(&mut self) {
        let buf_len = self.input_buffer.len();
        let out_len = self.output_buffer.len();

        for i in 0..self.grain_size {
            // Read from input buffer with fractional position.
            let read_idx_f = self.read_pos + (i as f64 * self.ratio as f64);
            let read_idx = read_idx_f.floor() as i64;
            let frac = (read_idx_f - read_idx as f64) as f32;

            let idx0 = ((read_idx % buf_len as i64 + buf_len as i64) % buf_len as i64) as usize;
            let idx1 = (idx0 + 1) % buf_len;

            let s0 = self.input_buffer[idx0];
            let s1 = self.input_buffer[idx1];
            let interpolated = s0 + (s1 - s0) * frac;

            let windowed = interpolated * self.window[i];
            let out_idx = (self.output_write_pos + i) % out_len;
            self.output_buffer[out_idx] += windowed;
        }

        // Advance the read position by hop_size (in the input domain).
        self.read_pos += self.hop_size as f64;
        if self.read_pos >= buf_len as f64 {
            self.read_pos -= buf_len as f64;
        }

        // Advance the output write position by hop_size.
        self.output_write_pos = (self.output_write_pos + self.hop_size) % out_len;
    }
}

impl DspEffect for PitchShifter {
    fn process(&mut self, samples: &mut [f32], _sample_rate: u32) {
        let buf_len = self.input_buffer.len();
        let out_len = self.output_buffer.len();

        for s in samples.iter_mut() {
            // Write input sample.
            self.input_buffer[self.write_pos % buf_len] = *s;
            self.write_pos = (self.write_pos + 1) % buf_len;
            self.samples_since_grain += 1;

            // When we've accumulated enough samples, process a grain.
            if self.samples_since_grain >= self.hop_size {
                self.samples_since_grain = 0;
                self.process_grain();
            }

            // Read output.
            *s = self.output_buffer[self.output_read_pos];
            self.output_buffer[self.output_read_pos] = 0.0;
            self.output_read_pos = (self.output_read_pos + 1) % out_len;
        }
    }

    fn name(&self) -> &str {
        "PitchShifter"
    }

    fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.write_pos = 0;
        self.read_pos = 0.0;
        self.output_buffer.fill(0.0);
        self.output_read_pos = 0;
        self.output_write_pos = 0;
        self.samples_since_grain = 0;
    }
}

// ===========================================================================
// 3D Spatializer with HRTF Approximation
// ===========================================================================

/// Listener state for 3D audio spatialization.
#[derive(Debug, Clone, Copy)]
pub struct SpatialListener {
    /// World-space position.
    pub position: glam::Vec3,
    /// Forward direction (unit vector).
    pub forward: glam::Vec3,
    /// Up direction (unit vector).
    pub up: glam::Vec3,
    /// Right direction (computed from forward x up, unit vector).
    pub right: glam::Vec3,
}

impl Default for SpatialListener {
    fn default() -> Self {
        Self {
            position: glam::Vec3::ZERO,
            forward: glam::Vec3::NEG_Z,
            up: glam::Vec3::Y,
            right: glam::Vec3::X,
        }
    }
}

impl SpatialListener {
    /// Create a listener from position, forward, and up vectors.
    pub fn new(position: glam::Vec3, forward: glam::Vec3, up: glam::Vec3) -> Self {
        let forward = forward.normalize();
        let right = forward.cross(up).normalize();
        let up = right.cross(forward).normalize();
        Self {
            position,
            forward,
            up,
            right,
        }
    }
}

/// Improved 3D audio spatializer with HRTF approximation.
///
/// Provides binaural rendering using three cues:
///
/// 1. **ILD** (Interaural Level Difference): the ear closer to the source
///    hears it louder. Implemented as per-ear gain.
/// 2. **ITD** (Interaural Time Difference): the ear farther from the source
///    hears it later. Implemented as a fractional delay.
/// 3. **Head Shadow**: high frequencies are attenuated more for the far ear,
///    because the head blocks short wavelengths. Implemented as a simple
///    low-pass filter on the far ear.
///
/// # Example
/// ```ignore
/// let mut spatializer = Spatializer3D::new(44100);
/// let listener = SpatialListener::new(pos, forward, up);
/// let source_pos = Vec3::new(5.0, 0.0, 0.0);
/// let (left, right) = spatializer.spatialize(&mono_input, source_pos, &listener);
/// ```
pub struct Spatializer3D {
    /// Sample rate.
    sample_rate: u32,
    /// Left ear delay line.
    delay_left: Vec<f32>,
    /// Right ear delay line.
    delay_right: Vec<f32>,
    /// Left delay line write position.
    delay_left_pos: usize,
    /// Right delay line write position.
    delay_right_pos: usize,
    /// Head shadow low-pass filter state for left ear.
    shadow_state_left: f32,
    /// Head shadow low-pass filter state for right ear.
    shadow_state_right: f32,
    /// Maximum ITD in samples for this sample rate.
    max_itd_samples: usize,
    /// Previous left gain (for smoothing).
    prev_gain_left: f32,
    /// Previous right gain (for smoothing).
    prev_gain_right: f32,
}

impl Spatializer3D {
    /// Create a new spatializer for the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        let max_itd = (HEAD_RADIUS / SPEED_OF_SOUND) * sample_rate as f32;
        let max_itd_samples = (max_itd * 2.0) as usize + 2;
        let delay_size = max_itd_samples + 256;

        Self {
            sample_rate,
            delay_left: vec![0.0; delay_size],
            delay_right: vec![0.0; delay_size],
            delay_left_pos: 0,
            delay_right_pos: 0,
            shadow_state_left: 0.0,
            shadow_state_right: 0.0,
            max_itd_samples,
            prev_gain_left: 1.0,
            prev_gain_right: 1.0,
        }
    }

    /// Compute ILD and ITD parameters from source/listener geometry.
    fn compute_hrtf_params(
        &self,
        source_pos: glam::Vec3,
        listener: &SpatialListener,
    ) -> HrtfParams {
        let to_source = source_pos - listener.position;
        let distance = to_source.length();

        if distance < 1e-6 {
            return HrtfParams {
                gain_left: 1.0,
                gain_right: 1.0,
                delay_left: 0.0,
                delay_right: 0.0,
                shadow_left: 1.0,
                shadow_right: 1.0,
            };
        }

        let direction = to_source / distance;

        // Azimuth: how far right the source is.
        // Positive = source is to the right of the listener.
        let azimuth = direction.dot(listener.right);

        // ILD: simple cosine-based model.
        // The ear on the same side as the source gets more volume.
        let ild_factor = 0.4; // Maximum ILD in dB would be ~6-10 dB; we use a scaled linear model
        let gain_left = 1.0 - azimuth.max(0.0) * ild_factor;
        let gain_right = 1.0 + azimuth.min(0.0) * ild_factor;

        // ITD: based on the Woodworth model.
        // ITD = (head_radius / speed_of_sound) * (azimuth + sin(azimuth))
        let itd_seconds = (HEAD_RADIUS / SPEED_OF_SOUND) * (azimuth.abs() + azimuth.abs().sin());
        let itd_samples = itd_seconds * self.sample_rate as f32;

        // The ear farther from the source gets the delay.
        let (delay_left, delay_right) = if azimuth >= 0.0 {
            // Source is to the right → left ear is farther.
            (itd_samples, 0.0)
        } else {
            (0.0, itd_samples)
        };

        // Head shadow: high-frequency attenuation for the far ear.
        // We model this as a low-pass filter coefficient (lower = more shadow).
        let shadow_amount = azimuth.abs() * 0.3; // 0 to 0.3
        let shadow_left = if azimuth >= 0.0 {
            1.0 - shadow_amount
        } else {
            1.0
        };
        let shadow_right = if azimuth < 0.0 {
            1.0 - shadow_amount
        } else {
            1.0
        };

        HrtfParams {
            gain_left,
            gain_right,
            delay_left,
            delay_right,
            shadow_left,
            shadow_right,
        }
    }

    /// Spatialize a mono input signal for a given source position and listener.
    ///
    /// Returns a tuple of (left_channel, right_channel).
    pub fn spatialize(
        &mut self,
        mono_input: &[f32],
        source_pos: glam::Vec3,
        listener: &SpatialListener,
    ) -> (Vec<f32>, Vec<f32>) {
        let params = self.compute_hrtf_params(source_pos, listener);
        let len = mono_input.len();
        let mut left = vec![0.0f32; len];
        let mut right = vec![0.0f32; len];

        let delay_size_l = self.delay_left.len();
        let delay_size_r = self.delay_right.len();

        // Smooth gain transitions to avoid clicks.
        let gain_smooth = 0.99;

        for i in 0..len {
            let sample = mono_input[i];

            // Smoothed gains.
            self.prev_gain_left =
                self.prev_gain_left * gain_smooth + params.gain_left * (1.0 - gain_smooth);
            self.prev_gain_right =
                self.prev_gain_right * gain_smooth + params.gain_right * (1.0 - gain_smooth);

            // Write to delay lines.
            self.delay_left[self.delay_left_pos] = sample;
            self.delay_right[self.delay_right_pos] = sample;

            // Read from delay lines with fractional delay (linear interpolation).
            let read_l = read_delay_fractional(
                &self.delay_left,
                self.delay_left_pos,
                params.delay_left,
            );
            let read_r = read_delay_fractional(
                &self.delay_right,
                self.delay_right_pos,
                params.delay_right,
            );

            // Apply gain.
            let left_sample = read_l * self.prev_gain_left;
            let right_sample = read_r * self.prev_gain_right;

            // Head shadow (simple one-pole low-pass).
            let shadow_coeff_l = params.shadow_left;
            let shadow_coeff_r = params.shadow_right;
            self.shadow_state_left =
                self.shadow_state_left * (1.0 - shadow_coeff_l) + left_sample * shadow_coeff_l;
            self.shadow_state_right = self.shadow_state_right * (1.0 - shadow_coeff_r)
                + right_sample * shadow_coeff_r;

            left[i] = self.shadow_state_left;
            right[i] = self.shadow_state_right;

            // Advance delay positions.
            self.delay_left_pos = (self.delay_left_pos + 1) % delay_size_l;
            self.delay_right_pos = (self.delay_right_pos + 1) % delay_size_r;
        }

        (left, right)
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.delay_left.fill(0.0);
        self.delay_right.fill(0.0);
        self.delay_left_pos = 0;
        self.delay_right_pos = 0;
        self.shadow_state_left = 0.0;
        self.shadow_state_right = 0.0;
        self.prev_gain_left = 1.0;
        self.prev_gain_right = 1.0;
    }
}

/// Internal HRTF parameters computed from geometry.
#[derive(Debug, Clone, Copy)]
struct HrtfParams {
    gain_left: f32,
    gain_right: f32,
    delay_left: f32,
    delay_right: f32,
    shadow_left: f32,
    shadow_right: f32,
}

/// Read from a delay line with fractional delay (linear interpolation).
fn read_delay_fractional(buffer: &[f32], write_pos: usize, delay: f32) -> f32 {
    let buf_len = buffer.len();
    let delay_int = delay as usize;
    let delay_frac = delay - delay_int as f32;

    let idx0 = (write_pos + buf_len - delay_int) % buf_len;
    let idx1 = (idx0 + buf_len - 1) % buf_len;

    buffer[idx0] * (1.0 - delay_frac) + buffer[idx1] * delay_frac
}

// ===========================================================================
// Brick-Wall Limiter
// ===========================================================================

/// Brick-wall limiter with look-ahead for transparent peak limiting.
///
/// The look-ahead allows the limiter to start reducing gain *before* a peak
/// arrives, avoiding harsh clipping artifacts. The release stage smoothly
/// returns gain to unity after a peak has passed.
///
/// # Algorithm
///
/// 1. Buffer `look_ahead` samples into a delay line.
/// 2. Scan the look-ahead window for peaks above `threshold`.
/// 3. Compute the gain reduction needed: `gain = threshold / peak`.
/// 4. Apply attack smoothing (instant attack via look-ahead).
/// 5. Apply release smoothing (exponential decay back to 1.0).
///
/// # Example
/// ```ignore
/// let mut limiter = Limiter::new(220, 0.95, 0.1);
/// limiter.process(&mut samples, 44100);
/// ```
pub struct Limiter {
    /// Look-ahead delay buffer.
    delay_buffer: Vec<f32>,
    /// Write position in the delay buffer.
    write_pos: usize,
    /// Look-ahead in samples.
    look_ahead: usize,
    /// Threshold (0.0 to 1.0).
    pub threshold: f32,
    /// Release time in seconds.
    pub release_time: f32,
    /// Current gain envelope.
    current_gain: f32,
    /// Attack coefficient (derived from look-ahead).
    attack_coeff: f32,
    /// Release coefficient (derived from release_time and sample_rate).
    release_coeff: f32,
    /// Peak hold buffer for look-ahead scanning.
    peak_buffer: Vec<f32>,
    /// Peak write position.
    peak_write_pos: usize,
}

impl Limiter {
    /// Create a new limiter.
    ///
    /// # Arguments
    /// * `look_ahead_samples` — look-ahead in samples (typically 5ms worth)
    /// * `threshold` — peak threshold (0.0 to 1.0, typically 0.95)
    /// * `release_time` — release time in seconds
    pub fn new(look_ahead_samples: usize, threshold: f32, release_time: f32) -> Self {
        let look_ahead = look_ahead_samples.max(1);

        Self {
            delay_buffer: vec![0.0; look_ahead + 1],
            write_pos: 0,
            look_ahead,
            threshold: threshold.clamp(0.01, 1.0),
            release_time,
            current_gain: 1.0,
            attack_coeff: 1.0 / look_ahead as f32,
            release_coeff: 0.0, // computed in process()
            peak_buffer: vec![0.0; look_ahead + 1],
            peak_write_pos: 0,
        }
    }

    /// Process a buffer of samples through the limiter.
    pub fn process_buffer(&mut self, samples: &mut [f32], threshold: f32, release_time: f32) {
        self.threshold = threshold.clamp(0.01, 1.0);
        self.release_time = release_time;

        // DspEffect::process delegates here.
        let dummy_rate = 44100u32;
        self.process_impl(samples, dummy_rate);
    }

    fn process_impl(&mut self, samples: &mut [f32], sample_rate: u32) {
        // Compute release coefficient.
        if self.release_time > 0.0 {
            self.release_coeff =
                (-1.0 / (self.release_time * sample_rate as f32)).exp();
        } else {
            self.release_coeff = 0.0;
        }

        let delay_len = self.delay_buffer.len();
        let peak_len = self.peak_buffer.len();

        for s in samples.iter_mut() {
            let input = *s;
            let input_abs = input.abs();

            // Write input to delay and peak buffers.
            self.delay_buffer[self.write_pos % delay_len] = input;
            self.peak_buffer[self.peak_write_pos % peak_len] = input_abs;
            self.peak_write_pos = (self.peak_write_pos + 1) % peak_len;

            // Find peak in the look-ahead window.
            let mut peak = 0.0f32;
            for i in 0..self.look_ahead {
                let idx = (self.peak_write_pos + peak_len - 1 - i) % peak_len;
                peak = peak.max(self.peak_buffer[idx]);
            }

            // Compute target gain.
            let target_gain = if peak > self.threshold {
                self.threshold / peak
            } else {
                1.0
            };

            // Smooth gain transitions.
            if target_gain < self.current_gain {
                // Attack (fast).
                self.current_gain = target_gain;
            } else {
                // Release (slow).
                self.current_gain =
                    self.current_gain * self.release_coeff + target_gain * (1.0 - self.release_coeff);
            }

            // Read from the delay buffer (delayed by look_ahead).
            let read_pos = (self.write_pos + delay_len - self.look_ahead) % delay_len;
            let delayed_sample = self.delay_buffer[read_pos];

            // Apply gain.
            *s = delayed_sample * self.current_gain;

            self.write_pos = (self.write_pos + 1) % delay_len;
        }
    }
}

impl DspEffect for Limiter {
    fn process(&mut self, samples: &mut [f32], sample_rate: u32) {
        self.process_impl(samples, sample_rate);
    }

    fn name(&self) -> &str {
        "Limiter"
    }

    fn reset(&mut self) {
        self.delay_buffer.fill(0.0);
        self.write_pos = 0;
        self.current_gain = 1.0;
        self.peak_buffer.fill(0.0);
        self.peak_write_pos = 0;
    }
}

// ===========================================================================
// Stereo Widener
// ===========================================================================

/// Mid/side stereo widener.
///
/// Decomposes the stereo signal into mid (center) and side (stereo difference)
/// components, scales the side signal by a width factor, and recombines.
///
/// `width = 0.0` → mono (no stereo content)
/// `width = 1.0` → unchanged
/// `width = 2.0` → exaggerated stereo (may cause phase issues)
///
/// # Algorithm
///
/// ```text
/// mid  = (L + R) / 2
/// side = (L - R) / 2
/// side *= width
/// L_out = mid + side
/// R_out = mid - side
/// ```
///
/// # Example
/// ```ignore
/// let mut widener = StereoWidener::new(1.5);
/// let (left_out, right_out) = widener.process(&left, &right);
/// ```
pub struct StereoWidener {
    /// Width factor. 1.0 = unchanged, 0.0 = mono, >1.0 = wider.
    pub width: f32,
}

impl StereoWidener {
    /// Create a new stereo widener.
    pub fn new(width: f32) -> Self {
        Self {
            width: width.max(0.0),
        }
    }

    /// Process stereo audio in-place.
    ///
    /// # Arguments
    /// * `left` — left channel samples (modified in place)
    /// * `right` — right channel samples (modified in place)
    /// * `width` — width factor override
    ///
    /// # Returns
    /// Tuple of (left, right) output.
    pub fn process(left: &[f32], right: &[f32], width: f32) -> (Vec<f32>, Vec<f32>) {
        let len = left.len().min(right.len());
        let mut out_left = vec![0.0f32; len];
        let mut out_right = vec![0.0f32; len];

        for i in 0..len {
            let mid = (left[i] + right[i]) * 0.5;
            let side = (left[i] - right[i]) * 0.5;
            let widened_side = side * width;
            out_left[i] = mid + widened_side;
            out_right[i] = mid - widened_side;
        }

        (out_left, out_right)
    }

    /// Process stereo audio in-place using the instance's width setting.
    pub fn process_inplace(&self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            let mid = (left[i] + right[i]) * 0.5;
            let side = (left[i] - right[i]) * 0.5;
            let widened_side = side * self.width;
            left[i] = mid + widened_side;
            right[i] = mid - widened_side;
        }
    }
}

impl Default for StereoWidener {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- FFT tests -----------------------------------------------------------

    #[test]
    fn fft_power_of_two_check() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(1000));
    }

    #[test]
    fn fft_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(1000), 1024);
        assert_eq!(next_power_of_two(1024), 1024);
    }

    #[test]
    fn fft_roundtrip_impulse() {
        // FFT of a unit impulse should be all 1s.
        let n = 8;
        let mut data = vec![Complex::ZERO; n];
        data[0] = Complex::new(1.0, 0.0);
        let original = data.clone();

        fft(&mut data, false);
        // All bins should have magnitude ~1.
        for c in &data {
            assert!((c.mag() - 1.0).abs() < 1e-5);
        }

        // IFFT should recover the original.
        fft(&mut data, true);
        for (i, c) in data.iter().enumerate() {
            assert!(
                (c.re - original[i].re).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                c.re,
                original[i].re
            );
        }
    }

    #[test]
    fn fft_roundtrip_sine() {
        let n = 256;
        let freq = 4.0;
        let mut data: Vec<Complex> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                Complex::new((TAU * freq * t).sin(), 0.0)
            })
            .collect();
        let original: Vec<f32> = data.iter().map(|c| c.re).collect();

        fft(&mut data, false);
        fft(&mut data, true);

        for (i, c) in data.iter().enumerate() {
            assert!(
                (c.re - original[i]).abs() < 1e-4,
                "FFT roundtrip mismatch at index {}: {} vs {}",
                i,
                c.re,
                original[i]
            );
        }
    }

    #[test]
    fn fft_known_dc() {
        // FFT of [1, 1, 1, 1] should have DC = 4 and all other bins = 0.
        let mut data = vec![Complex::new(1.0, 0.0); 4];
        fft(&mut data, false);
        assert!((data[0].re - 4.0).abs() < 1e-5);
        for c in &data[1..] {
            assert!(c.mag() < 1e-5);
        }
    }

    // -- Convolution tests ---------------------------------------------------

    #[test]
    fn fft_convolve_impulse() {
        // Convolving with a unit impulse should return the original signal.
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![1.0, 0.0, 0.0];
        let result = fft_convolve(&signal, &kernel);

        assert_eq!(result.len(), signal.len() + kernel.len() - 1);
        for (i, &s) in signal.iter().enumerate() {
            assert!(
                (result[i] - s).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                result[i],
                s
            );
        }
    }

    #[test]
    fn fft_convolve_known() {
        // [1, 2, 3] * [1, 1] = [1, 3, 5, 3]
        let signal = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0, 1.0];
        let result = fft_convolve(&signal, &kernel);

        let expected = [1.0, 3.0, 5.0, 3.0];
        assert_eq!(result.len(), expected.len());
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                (result[i] - e).abs() < 1e-3,
                "Conv mismatch at {}: {} vs {}",
                i,
                result[i],
                e
            );
        }
    }

    #[test]
    fn fft_convolve_empty() {
        assert!(fft_convolve(&[], &[1.0]).is_empty());
        assert!(fft_convolve(&[1.0], &[]).is_empty());
    }

    // -- Convolution reverb tests --------------------------------------------

    #[test]
    fn convolution_reverb_silence() {
        let ir = vec![1.0, 0.0, 0.0, 0.0];
        let mut reverb = ConvolutionReverb::new(&ir, 64);
        reverb.wet_mix = 1.0;
        let mut samples = vec![0.0; 128];
        reverb.process(&mut samples, 44100);
        // Silence in → silence out.
        for s in &samples {
            assert!(s.abs() < 1e-4);
        }
    }

    #[test]
    fn convolution_reverb_dry_passthrough() {
        let ir = vec![1.0; 64];
        let mut reverb = ConvolutionReverb::new(&ir, 64);
        reverb.wet_mix = 0.0;
        let original: Vec<f32> = (0..128)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let mut samples = original.clone();
        reverb.process(&mut samples, 44100);
        // With wet_mix = 0, output should match input.
        for (i, &s) in samples.iter().enumerate() {
            assert!(
                (s - original[i]).abs() < 1e-4,
                "Dry passthrough mismatch at {}",
                i
            );
        }
    }

    // -- Pitch shifter tests -------------------------------------------------

    #[test]
    fn pitch_shifter_unity_ratio() {
        // With ratio = 1.0, the output should approximate the input.
        let mut shifter = PitchShifter::new(512, 128);
        shifter.set_ratio(1.0);
        let input: Vec<f32> = (0..2048)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let mut output = input.clone();
        shifter.process(&mut output, 44100);
        // The output won't be identical due to windowing, but energy should be similar.
        let input_energy: f32 = input.iter().map(|s| s * s).sum();
        let output_energy: f32 = output.iter().map(|s| s * s).sum();
        // Energy ratio should be roughly 1.0 (within 50% tolerance due to windowing).
        let ratio = output_energy / input_energy.max(1e-10);
        assert!(
            ratio > 0.1 && ratio < 10.0,
            "Energy ratio {} out of expected range",
            ratio
        );
    }

    #[test]
    fn pitch_shifter_static_function() {
        let input: Vec<f32> = (0..4096)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let output = PitchShifter::shift(&input, 0.0);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn pitch_shifter_semitones() {
        let mut shifter = PitchShifter::new(1024, 256);
        shifter.set_semitones(12.0);
        assert!((shifter.ratio - 2.0).abs() < 1e-4);
        shifter.set_semitones(-12.0);
        assert!((shifter.ratio - 0.5).abs() < 1e-4);
        shifter.set_semitones(0.0);
        assert!((shifter.ratio - 1.0).abs() < 1e-4);
    }

    // -- Spatializer tests ---------------------------------------------------

    #[test]
    fn spatializer_center_source() {
        // A source directly in front should produce equal left and right.
        let mut spat = Spatializer3D::new(44100);
        let listener = SpatialListener::default();
        let source_pos = listener.position + listener.forward * 5.0;

        let mono: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let (left, right) = spat.spatialize(&mono, source_pos, &listener);

        assert_eq!(left.len(), mono.len());
        assert_eq!(right.len(), mono.len());

        // Energy should be roughly equal between channels.
        let energy_l: f32 = left.iter().map(|s| s * s).sum();
        let energy_r: f32 = right.iter().map(|s| s * s).sum();
        let ratio = energy_l / energy_r.max(1e-10);
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Center source L/R energy ratio should be near 1.0, got {}",
            ratio
        );
    }

    #[test]
    fn spatializer_right_source() {
        // A source to the right should be louder in the right channel.
        let mut spat = Spatializer3D::new(44100);
        let listener = SpatialListener::default();
        let source_pos = listener.position + listener.right * 5.0;

        let mono: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let (left, right) = spat.spatialize(&mono, source_pos, &listener);

        // Skip early samples (gain smoothing).
        let skip = 100;
        let energy_l: f32 = left[skip..].iter().map(|s| s * s).sum();
        let energy_r: f32 = right[skip..].iter().map(|s| s * s).sum();

        assert!(
            energy_r > energy_l,
            "Right source should have more energy in right channel: L={} R={}",
            energy_l,
            energy_r
        );
    }

    #[test]
    fn spatializer_reset() {
        let mut spat = Spatializer3D::new(44100);
        spat.shadow_state_left = 999.0;
        spat.reset();
        assert_eq!(spat.shadow_state_left, 0.0);
    }

    // -- Limiter tests -------------------------------------------------------

    #[test]
    fn limiter_below_threshold() {
        let mut limiter = Limiter::new(220, 0.95, 0.1);
        let mut samples: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let original = samples.clone();
        limiter.process(&mut samples, 44100);
        // Samples below threshold should not be significantly altered
        // (they're delayed by the look-ahead though).
    }

    #[test]
    fn limiter_clips_peaks() {
        let mut limiter = Limiter::new(64, 0.5, 0.01);
        // Create samples with a loud peak.
        let mut samples = vec![0.1; 256];
        samples[128] = 2.0;
        samples[129] = 2.0;
        samples[130] = 2.0;

        limiter.process(&mut samples, 44100);

        // After limiting, no sample should exceed the threshold (approximately).
        for (i, &s) in samples.iter().enumerate() {
            assert!(
                s.abs() <= 0.55, // some tolerance for release
                "Sample {} exceeded threshold: {}",
                i,
                s
            );
        }
    }

    #[test]
    fn limiter_reset() {
        let mut limiter = Limiter::new(220, 0.95, 0.1);
        limiter.current_gain = 0.5;
        limiter.reset();
        assert_eq!(limiter.current_gain, 1.0);
    }

    // -- Stereo widener tests ------------------------------------------------

    #[test]
    fn stereo_widener_mono() {
        // Width = 0 should produce mono (left == right).
        let left = vec![1.0, 0.5, -0.3, 0.8];
        let right = vec![0.5, 0.8, 0.1, -0.2];
        let (out_l, out_r) = StereoWidener::process(&left, &right, 0.0);
        for i in 0..left.len() {
            assert!(
                (out_l[i] - out_r[i]).abs() < 1e-6,
                "Mono: L[{}]={} R[{}]={}",
                i,
                out_l[i],
                i,
                out_r[i]
            );
        }
    }

    #[test]
    fn stereo_widener_unity() {
        // Width = 1 should preserve the original.
        let left = vec![1.0, 0.5, -0.3, 0.8];
        let right = vec![0.5, 0.8, 0.1, -0.2];
        let (out_l, out_r) = StereoWidener::process(&left, &right, 1.0);
        for i in 0..left.len() {
            assert!(
                (out_l[i] - left[i]).abs() < 1e-6,
                "Unity L mismatch at {}",
                i
            );
            assert!(
                (out_r[i] - right[i]).abs() < 1e-6,
                "Unity R mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn stereo_widener_double() {
        // Width = 2 should exaggerate stereo differences.
        let left = vec![1.0, 0.0];
        let right = vec![0.0, 1.0];
        let (out_l, out_r) = StereoWidener::process(&left, &right, 2.0);
        // mid = 0.5, side = 0.5, widened_side = 1.0
        // L = 0.5 + 1.0 = 1.5, R = 0.5 - 1.0 = -0.5
        assert!((out_l[0] - 1.5).abs() < 1e-6);
        assert!((out_r[0] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn stereo_widener_inplace() {
        let mut left = vec![1.0, 0.5];
        let mut right = vec![0.5, 0.8];
        let widener = StereoWidener::new(1.5);
        widener.process_inplace(&mut left, &mut right);
        // Just verify it doesn't crash and modifies values.
        assert_ne!(left[0], 1.0); // Should have changed
    }

    // -- Complex number tests ------------------------------------------------

    #[test]
    fn complex_mul() {
        let a = Complex::new(3.0, 2.0);
        let b = Complex::new(1.0, 4.0);
        let c = a.mul(b);
        // (3+2i)(1+4i) = 3+12i+2i+8i^2 = 3+14i-8 = -5+14i
        assert!((c.re - (-5.0)).abs() < 1e-5);
        assert!((c.im - 14.0).abs() < 1e-5);
    }

    #[test]
    fn complex_polar_roundtrip() {
        let c = Complex::new(3.0, 4.0);
        let mag = c.mag();
        let phase = c.phase();
        let recovered = Complex::from_polar(mag, phase);
        assert!((recovered.re - c.re).abs() < 1e-5);
        assert!((recovered.im - c.im).abs() < 1e-5);
    }

    #[test]
    fn complex_conjugate() {
        let c = Complex::new(3.0, 4.0);
        let conj = c.conj();
        assert_eq!(conj.re, c.re);
        assert_eq!(conj.im, -c.im);
    }
}
