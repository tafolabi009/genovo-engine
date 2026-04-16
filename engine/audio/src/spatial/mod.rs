//! Spatial (3D) audio processing.
//!
//! Provides components and algorithms for positioning sounds in 3D space,
//! distance attenuation with multiple models, stereo panning from
//! listener-relative direction, Doppler pitch shifting, cone-based
//! directional emitters, and occlusion parameters.

use glam::Vec3;

// ---------------------------------------------------------------------------
// Attenuation models
// ---------------------------------------------------------------------------

/// Distance attenuation model for spatial audio sources.
#[derive(Debug, Clone)]
pub enum AttenuationModel {
    /// Volume decreases linearly between `min_distance` and `max_distance`.
    ///
    /// `gain = 1.0 - clamp((dist - inner) / (outer - inner), 0, 1)`
    Linear,

    /// Volume decreases logarithmically (realistic for open environments).
    ///
    /// `gain = 1.0 / (1.0 + rolloff * (dist - inner))`
    Logarithmic {
        /// Rolloff factor controlling attenuation steepness.
        rolloff_factor: f32,
    },

    /// Volume follows inverse-square law (physically accurate).
    ///
    /// `gain = inner^2 / max(dist^2, inner^2)`
    InverseSquare,

    /// User-defined attenuation curve.
    Custom {
        /// Lookup table: (distance, volume) pairs sorted by distance.
        /// Values between entries are linearly interpolated.
        curve: Vec<(f32, f32)>,
    },
}

impl Default for AttenuationModel {
    fn default() -> Self {
        Self::InverseSquare
    }
}

// ---------------------------------------------------------------------------
// Spatial audio source
// ---------------------------------------------------------------------------

/// Spatial audio source component.
///
/// Attached to entities that emit 3D-positioned sound.  Works in conjunction
/// with [`AudioListener`] to compute attenuation, panning, and Doppler.
#[derive(Debug, Clone)]
pub struct SpatialAudioSource {
    /// World-space position of the source (updated by AudioSystem from Transform).
    pub position: Vec3,

    /// World-space velocity of the source (for Doppler effect).
    pub velocity: Vec3,

    /// Distance at which attenuation begins (inner radius).
    pub min_distance: f32,

    /// Distance beyond which the source is inaudible (outer radius).
    pub max_distance: f32,

    /// Inner cone angle in radians (full volume inside this cone).
    pub inner_cone_angle: f32,

    /// Outer cone angle in radians (attenuated between inner and outer cone).
    pub outer_cone_angle: f32,

    /// Volume multiplier applied outside the outer cone.
    pub outer_cone_volume: f32,

    /// Direction the source is facing (for directional/cone emitters).
    pub direction: Vec3,

    /// Attenuation model.
    pub attenuation: AttenuationModel,

    /// Whether Doppler effect is enabled for this source.
    pub doppler_enabled: bool,

    /// Doppler factor multiplier (1.0 = realistic, 0.0 = disabled).
    pub doppler_factor: f32,
}

impl Default for SpatialAudioSource {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            min_distance: 1.0,
            max_distance: 100.0,
            inner_cone_angle: std::f32::consts::TAU, // omnidirectional
            outer_cone_angle: std::f32::consts::TAU,
            outer_cone_volume: 0.0,
            direction: Vec3::NEG_Z,
            attenuation: AttenuationModel::default(),
            doppler_enabled: true,
            doppler_factor: 1.0,
        }
    }
}

impl SpatialAudioSource {
    // -----------------------------------------------------------------------
    // Distance attenuation
    // -----------------------------------------------------------------------

    /// Compute the distance attenuation factor `[0.0, 1.0]` for a given
    /// listener position.
    pub fn compute_attenuation(&self, listener_pos: Vec3) -> f32 {
        let distance = (self.position - listener_pos).length();
        self.compute_attenuation_distance(distance)
    }

    /// Compute attenuation from a pre-computed distance value.
    pub fn compute_attenuation_distance(&self, distance: f32) -> f32 {
        if distance <= self.min_distance {
            return 1.0;
        }
        if distance >= self.max_distance {
            return 0.0;
        }

        let raw = match &self.attenuation {
            AttenuationModel::Linear => {
                // Linear: 1.0 - clamp((dist - inner) / (outer - inner), 0, 1)
                let range = self.max_distance - self.min_distance;
                if range <= 0.0 {
                    1.0
                } else {
                    1.0 - ((distance - self.min_distance) / range).clamp(0.0, 1.0)
                }
            }
            AttenuationModel::Logarithmic { rolloff_factor } => {
                // Logarithmic: 1.0 / (1.0 + rolloff * (dist - inner))
                let d = distance.max(self.min_distance);
                1.0 / (1.0 + rolloff_factor * (d - self.min_distance))
            }
            AttenuationModel::InverseSquare => {
                // InverseSquare: inner^2 / max(dist^2, inner^2)
                let inner_sq = self.min_distance * self.min_distance;
                let dist_sq = distance * distance;
                inner_sq / dist_sq.max(inner_sq)
            }
            AttenuationModel::Custom { curve } => Self::sample_curve(curve, distance),
        };

        raw.clamp(0.0, 1.0)
    }

    /// Sample a user-defined attenuation curve via linear interpolation.
    fn sample_curve(curve: &[(f32, f32)], distance: f32) -> f32 {
        if curve.is_empty() {
            return 1.0;
        }
        if distance <= curve[0].0 {
            return curve[0].1;
        }
        let last = curve.len() - 1;
        if distance >= curve[last].0 {
            return curve[last].1;
        }
        for window in curve.windows(2) {
            let (d0, v0) = window[0];
            let (d1, v1) = window[1];
            if distance >= d0 && distance <= d1 {
                let t = if (d1 - d0).abs() < 1e-9 {
                    0.0
                } else {
                    (distance - d0) / (d1 - d0)
                };
                return v0 + t * (v1 - v0);
            }
        }
        1.0
    }

    // -----------------------------------------------------------------------
    // Cone attenuation
    // -----------------------------------------------------------------------

    /// Compute cone-based directional attenuation for this source relative
    /// to the listener position.
    ///
    /// Returns a gain factor in `[outer_cone_volume, 1.0]`.
    pub fn compute_cone_attenuation(&self, listener_pos: Vec3) -> f32 {
        let to_listener = listener_pos - self.position;
        let dist = to_listener.length();
        if dist < 1e-9 {
            return 1.0; // listener is on top of source
        }
        let to_listener_norm = to_listener / dist;

        let dir_norm = if self.direction.length_squared() < 1e-9 {
            return 1.0; // no direction = omnidirectional
        } else {
            self.direction.normalize()
        };

        let cos_angle = dir_norm.dot(to_listener_norm).clamp(-1.0, 1.0);
        let angle = cos_angle.acos(); // [0, PI]

        let half_inner = self.inner_cone_angle * 0.5;
        let half_outer = self.outer_cone_angle * 0.5;

        if angle <= half_inner {
            1.0
        } else if angle >= half_outer {
            self.outer_cone_volume
        } else {
            // Interpolate between inner and outer cone.
            let range = half_outer - half_inner;
            if range <= 0.0 {
                1.0
            } else {
                let t = (angle - half_inner) / range;
                1.0 + t * (self.outer_cone_volume - 1.0)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stereo panning
    // -----------------------------------------------------------------------

    /// Compute a stereo pan value `[-1.0 left, +1.0 right]` from the
    /// listener's perspective.
    ///
    /// * `listener_pos`     - world position of the listener
    /// * `listener_forward` - normalised forward direction of the listener
    /// * `listener_right`   - normalised right direction of the listener
    ///
    /// The pan is computed by projecting the source direction onto the
    /// listener's right axis:
    /// ```text
    /// pan = dot(normalize(source_pos - listener_pos), listener_right)
    /// ```
    pub fn compute_pan(
        &self,
        listener_pos: Vec3,
        listener_forward: Vec3,
        listener_right: Vec3,
    ) -> f32 {
        let to_source = self.position - listener_pos;
        let dist = to_source.length();
        if dist < 1e-9 {
            return 0.0; // source is at listener position
        }
        let dir = to_source / dist;

        // Project onto listener's right axis.
        let pan = dir.dot(listener_right).clamp(-1.0, 1.0);

        // Optionally attenuate panning at very close range (within min_distance)
        // so sounds near the listener don't hard-pan to one side.
        let proximity_scale = if dist < self.min_distance && self.min_distance > 0.0 {
            dist / self.min_distance
        } else {
            1.0
        };

        // We also fade pan toward centre for sources behind the listener.
        let behind_factor = dir.dot(listener_forward);
        // When behind_factor < 0 (behind), narrow the pan slightly.
        let behind_scale = if behind_factor < 0.0 {
            // Reduce pan width by 30% when directly behind.
            1.0 + behind_factor * 0.3
        } else {
            1.0
        };

        (pan * proximity_scale * behind_scale).clamp(-1.0, 1.0)
    }

    // -----------------------------------------------------------------------
    // Doppler effect
    // -----------------------------------------------------------------------

    /// Compute the Doppler pitch multiplier.
    ///
    /// Uses the classical Doppler formula:
    /// ```text
    /// ratio = (speed_of_sound - listener_vel_toward) / (speed_of_sound - source_vel_toward)
    /// ```
    /// where `vel_toward` is the velocity component along the line from
    /// listener to source (positive = approaching).
    ///
    /// Returns a multiplier to apply to the playback rate (pitch).
    pub fn compute_doppler(
        &self,
        listener_pos: Vec3,
        listener_velocity: Vec3,
        speed_of_sound: f32,
    ) -> f32 {
        if !self.doppler_enabled || self.doppler_factor <= 0.0 {
            return 1.0;
        }

        let to_source = self.position - listener_pos;
        let dist = to_source.length();
        if dist < 1e-9 {
            return 1.0;
        }
        let dir = to_source / dist; // unit vector from listener toward source

        // Component of velocity along the listener-source axis.
        // Positive = approaching the other party.
        let listener_vel_toward = listener_velocity.dot(dir);
        let source_vel_toward = self.velocity.dot(-dir); // negative dir because source moving toward listener is approaching

        // Apply doppler factor as an exaggeration multiplier.
        let scaled_listener = listener_vel_toward * self.doppler_factor;
        let scaled_source = source_vel_toward * self.doppler_factor;

        // Guard against supersonic / zero-denominator.
        let denominator = (speed_of_sound - scaled_source).max(1.0);
        let numerator = (speed_of_sound - scaled_listener).max(1.0);

        let ratio = numerator / denominator;

        // Clamp to a sane range [0.25, 4.0] to prevent extreme pitch shifts.
        ratio.clamp(0.25, 4.0)
    }

    // -----------------------------------------------------------------------
    // Combined spatial parameters
    // -----------------------------------------------------------------------

    /// Compute all spatial audio parameters at once: attenuation, pan, and
    /// Doppler pitch ratio.
    ///
    /// Returns `(volume_attenuation, pan, doppler_pitch_ratio)`.
    pub fn compute_spatial_params(
        &self,
        listener: &AudioListener,
        speed_of_sound: f32,
    ) -> SpatialResult {
        let distance = (self.position - listener.position).length();

        // Distance attenuation.
        let distance_atten = self.compute_attenuation_distance(distance);

        // Cone attenuation.
        let cone_atten = self.compute_cone_attenuation(listener.position);

        // Combined volume.
        let volume = distance_atten * cone_atten;

        // Stereo pan.
        let right = listener.right();
        let pan = self.compute_pan(listener.position, listener.forward, right);

        // Doppler.
        let doppler = self.compute_doppler(listener.position, listener.velocity, speed_of_sound);

        SpatialResult {
            volume,
            distance_attenuation: distance_atten,
            cone_attenuation: cone_atten,
            pan,
            doppler_pitch_ratio: doppler,
            distance,
        }
    }
}

/// Result of computing spatial audio parameters for a source relative to a
/// listener.
#[derive(Debug, Clone, Copy)]
pub struct SpatialResult {
    /// Combined volume attenuation (distance * cone).
    pub volume: f32,
    /// Distance-based attenuation alone.
    pub distance_attenuation: f32,
    /// Cone-based attenuation alone.
    pub cone_attenuation: f32,
    /// Stereo pan [-1 left, +1 right].
    pub pan: f32,
    /// Doppler pitch multiplier.
    pub doppler_pitch_ratio: f32,
    /// Euclidean distance from listener to source.
    pub distance: f32,
}

// ---------------------------------------------------------------------------
// Audio listener
// ---------------------------------------------------------------------------

/// The audio listener component.
///
/// Typically attached to the active camera entity.  There should be at most
/// one active listener in a scene at any time.
#[derive(Debug, Clone)]
pub struct AudioListener {
    /// World-space position.
    pub position: Vec3,
    /// World-space velocity (for Doppler effect on the listener side).
    pub velocity: Vec3,
    /// Forward direction vector (normalised).
    pub forward: Vec3,
    /// Up direction vector (normalised).
    pub up: Vec3,
    /// Master listener volume multiplier.
    pub volume: f32,
}

impl Default for AudioListener {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            volume: 1.0,
        }
    }
}

impl AudioListener {
    /// Compute the right direction from forward and up.
    pub fn right(&self) -> Vec3 {
        self.forward.cross(self.up).normalize()
    }

    /// Set orientation from forward and up vectors (will be normalised).
    pub fn set_orientation(&mut self, forward: Vec3, up: Vec3) {
        self.forward = forward.normalize_or_zero();
        self.up = up.normalize_or_zero();
    }

    /// Build a listener from position and look-at target.
    pub fn look_at(position: Vec3, target: Vec3, up: Vec3) -> Self {
        let forward = (target - position).normalize_or_zero();
        Self {
            position,
            velocity: Vec3::ZERO,
            forward,
            up: up.normalize_or_zero(),
            volume: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Occlusion / Obstruction
// ---------------------------------------------------------------------------

/// Parameters for audio occlusion and obstruction.
///
/// Occlusion: source is behind a wall (muffles high frequencies).
/// Obstruction: direct path is blocked but indirect paths exist.
#[derive(Debug, Clone)]
pub struct OcclusionParams {
    /// Occlusion factor [0.0, 1.0].  0 = no occlusion, 1 = fully occluded.
    pub occlusion: f32,
    /// Obstruction factor [0.0, 1.0].  0 = no obstruction, 1 = fully obstructed.
    pub obstruction: f32,
    /// Low-pass filter cutoff frequency applied when occluded (Hz).
    pub lpf_cutoff: f32,
    /// Wet mix for reverb when occluded (simulates sound travelling around
    /// obstacles).
    pub wet_mix: f32,
}

impl Default for OcclusionParams {
    fn default() -> Self {
        Self {
            occlusion: 0.0,
            obstruction: 0.0,
            lpf_cutoff: 22000.0,
            wet_mix: 0.0,
        }
    }
}

impl OcclusionParams {
    /// Compute an approximate volume multiplier from occlusion/obstruction.
    ///
    /// A simple heuristic: each point of occlusion reduces volume by up to 80%
    /// and obstruction by up to 50%.
    pub fn volume_multiplier(&self) -> f32 {
        let occ = 1.0 - self.occlusion * 0.8;
        let obs = 1.0 - self.obstruction * 0.5;
        (occ * obs).clamp(0.0, 1.0)
    }

    /// Compute a low-pass filter cutoff taking occlusion into account.
    /// Fully occluded -> 500 Hz.  No occlusion -> `lpf_cutoff` (default 22 kHz).
    pub fn effective_lpf_cutoff(&self) -> f32 {
        let min_cutoff = 500.0;
        let t = self.occlusion.clamp(0.0, 1.0);
        self.lpf_cutoff + t * (min_cutoff - self.lpf_cutoff)
    }
}

// ---------------------------------------------------------------------------
// HRTF (Head-Related Transfer Function) placeholder
// ---------------------------------------------------------------------------

/// Head-Related Transfer Function data for binaural audio rendering.
///
/// HRTF enables realistic 3D audio over headphones by simulating how sound
/// is filtered by the listener's head, ears, and torso.
pub struct HrtfData {
    /// Number of azimuth/elevation measurement positions.
    pub measurement_count: u32,
    /// Sample rate the HRTF was measured at.
    pub sample_rate: u32,
    /// Length of each impulse response in samples.
    pub ir_length: u32,
}

/// HRTF processor for binaural rendering.
pub struct HrtfProcessor {
    /// The HRTF dataset to use.
    pub hrtf_data: Option<HrtfData>,
    /// Whether HRTF processing is enabled.
    pub enabled: bool,
}

impl Default for HrtfProcessor {
    fn default() -> Self {
        Self {
            hrtf_data: None,
            enabled: false,
        }
    }
}

impl HrtfProcessor {
    /// Process a mono audio buffer through the HRTF to produce stereo binaural
    /// output.
    ///
    /// Note: full HRTF convolution requires impulse response data which is
    /// loaded externally.  This placeholder does a simple stereo widening
    /// approximation when no HRTF data is loaded.
    pub fn process(
        &self,
        mono_input: &[f32],
        azimuth: f32,
        _elevation: f32,
        left_output: &mut [f32],
        right_output: &mut [f32],
    ) {
        if self.hrtf_data.is_some() {
            // With real HRTF data we would do overlap-save FFT convolution.
            // For now, fall through to the simple path.
        }

        // Simple approximation: pan the mono signal based on azimuth and apply
        // a small inter-aural time delay (ITD) as a sample offset.
        let pan = azimuth.sin().clamp(-1.0, 1.0);
        let angle = ((pan + 1.0) * 0.5) * std::f32::consts::FRAC_PI_2;
        let gain_l = angle.cos();
        let gain_r = angle.sin();

        let len = mono_input.len().min(left_output.len()).min(right_output.len());
        for i in 0..len {
            left_output[i] = mono_input[i] * gain_l;
            right_output[i] = mono_input[i] * gain_r;
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_source() -> SpatialAudioSource {
        SpatialAudioSource::default()
    }

    fn default_listener() -> AudioListener {
        AudioListener::default()
    }

    // -- Attenuation tests --

    #[test]
    fn linear_attenuation_at_min() {
        let mut src = default_source();
        src.attenuation = AttenuationModel::Linear;
        src.min_distance = 1.0;
        src.max_distance = 10.0;
        src.position = Vec3::new(1.0, 0.0, 0.0);
        let atten = src.compute_attenuation(Vec3::ZERO);
        assert!((atten - 1.0).abs() < 1e-4, "at min distance, atten should be 1.0, got {atten}");
    }

    #[test]
    fn linear_attenuation_at_max() {
        let mut src = default_source();
        src.attenuation = AttenuationModel::Linear;
        src.min_distance = 1.0;
        src.max_distance = 10.0;
        src.position = Vec3::new(10.0, 0.0, 0.0);
        let atten = src.compute_attenuation(Vec3::ZERO);
        assert!((atten - 0.0).abs() < 1e-4, "at max distance, atten should be 0.0, got {atten}");
    }

    #[test]
    fn linear_attenuation_midpoint() {
        let mut src = default_source();
        src.attenuation = AttenuationModel::Linear;
        src.min_distance = 0.0;
        src.max_distance = 10.0;
        src.position = Vec3::new(5.0, 0.0, 0.0);
        let atten = src.compute_attenuation(Vec3::ZERO);
        assert!((atten - 0.5).abs() < 1e-4, "at midpoint, atten should be 0.5, got {atten}");
    }

    #[test]
    fn inverse_square_attenuation() {
        let mut src = default_source();
        src.attenuation = AttenuationModel::InverseSquare;
        src.min_distance = 1.0;
        src.max_distance = 100.0;
        // At distance 2: gain = 1^2 / 2^2 = 0.25
        src.position = Vec3::new(2.0, 0.0, 0.0);
        let atten = src.compute_attenuation(Vec3::ZERO);
        assert!((atten - 0.25).abs() < 1e-4, "inverse square at d=2 should be 0.25, got {atten}");
    }

    #[test]
    fn inverse_square_at_min() {
        let mut src = default_source();
        src.attenuation = AttenuationModel::InverseSquare;
        src.min_distance = 2.0;
        src.max_distance = 100.0;
        src.position = Vec3::new(1.0, 0.0, 0.0); // within min distance
        let atten = src.compute_attenuation(Vec3::ZERO);
        assert!((atten - 1.0).abs() < 1e-4, "within min distance should be 1.0");
    }

    #[test]
    fn logarithmic_attenuation() {
        let mut src = default_source();
        src.attenuation = AttenuationModel::Logarithmic { rolloff_factor: 1.0 };
        src.min_distance = 1.0;
        src.max_distance = 100.0;
        // At distance 2: gain = 1 / (1 + 1 * (2 - 1)) = 1/2 = 0.5
        src.position = Vec3::new(2.0, 0.0, 0.0);
        let atten = src.compute_attenuation(Vec3::ZERO);
        assert!((atten - 0.5).abs() < 1e-4, "log atten at d=2 should be 0.5, got {atten}");
    }

    #[test]
    fn custom_curve_attenuation() {
        let mut src = default_source();
        src.attenuation = AttenuationModel::Custom {
            curve: vec![(0.0, 1.0), (5.0, 0.5), (10.0, 0.0)],
        };
        src.min_distance = 0.0;
        src.max_distance = 20.0;

        src.position = Vec3::new(5.0, 0.0, 0.0);
        let a1 = src.compute_attenuation(Vec3::ZERO);
        assert!((a1 - 0.5).abs() < 1e-4, "at curve point d=5, expected 0.5, got {a1}");

        src.position = Vec3::new(2.5, 0.0, 0.0);
        let a2 = src.compute_attenuation(Vec3::ZERO);
        assert!((a2 - 0.75).abs() < 1e-4, "interpolated d=2.5, expected 0.75, got {a2}");
    }

    #[test]
    fn attenuation_beyond_max_is_zero() {
        let mut src = default_source();
        src.min_distance = 1.0;
        src.max_distance = 10.0;
        src.position = Vec3::new(50.0, 0.0, 0.0);
        let atten = src.compute_attenuation(Vec3::ZERO);
        assert!((atten).abs() < 1e-4, "beyond max should be 0");
    }

    // -- Panning tests --

    #[test]
    fn pan_source_on_right() {
        let mut src = default_source();
        src.position = Vec3::new(5.0, 0.0, 0.0); // to the right of origin

        let listener = AudioListener {
            position: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            ..default_listener()
        };
        let right = listener.right();
        let pan = src.compute_pan(listener.position, listener.forward, right);
        assert!(pan > 0.5, "source on the right should produce positive pan, got {pan}");
    }

    #[test]
    fn pan_source_on_left() {
        let mut src = default_source();
        src.position = Vec3::new(-5.0, 0.0, 0.0);

        let listener = AudioListener {
            position: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            ..default_listener()
        };
        let right = listener.right();
        let pan = src.compute_pan(listener.position, listener.forward, right);
        assert!(pan < -0.5, "source on the left should produce negative pan, got {pan}");
    }

    #[test]
    fn pan_source_directly_ahead() {
        let mut src = default_source();
        src.position = Vec3::new(0.0, 0.0, -5.0); // ahead (forward = -Z)

        let listener = default_listener();
        let right = listener.right();
        let pan = src.compute_pan(listener.position, listener.forward, right);
        assert!(pan.abs() < 0.1, "source ahead should be near-centre pan, got {pan}");
    }

    #[test]
    fn pan_source_at_listener_is_centre() {
        let mut src = default_source();
        src.position = Vec3::ZERO;
        let listener = default_listener();
        let right = listener.right();
        let pan = src.compute_pan(listener.position, listener.forward, right);
        assert!((pan).abs() < 1e-6, "source at listener should be pan=0");
    }

    // -- Doppler tests --

    #[test]
    fn doppler_stationary() {
        let src = default_source();
        let listener = default_listener();
        let d = src.compute_doppler(listener.position, listener.velocity, 343.0);
        assert!((d - 1.0).abs() < 1e-4, "stationary objects => doppler 1.0, got {d}");
    }

    #[test]
    fn doppler_approaching() {
        let mut src = default_source();
        src.position = Vec3::new(10.0, 0.0, 0.0);
        src.velocity = Vec3::new(-50.0, 0.0, 0.0); // moving toward listener

        let listener = default_listener();
        let d = src.compute_doppler(listener.position, listener.velocity, 343.0);
        assert!(d > 1.0, "approaching source should have doppler > 1.0, got {d}");
    }

    #[test]
    fn doppler_receding() {
        let mut src = default_source();
        src.position = Vec3::new(10.0, 0.0, 0.0);
        src.velocity = Vec3::new(50.0, 0.0, 0.0); // moving away

        let listener = default_listener();
        let d = src.compute_doppler(listener.position, listener.velocity, 343.0);
        assert!(d < 1.0, "receding source should have doppler < 1.0, got {d}");
    }

    #[test]
    fn doppler_disabled() {
        let mut src = default_source();
        src.doppler_enabled = false;
        src.position = Vec3::new(10.0, 0.0, 0.0);
        src.velocity = Vec3::new(-100.0, 0.0, 0.0);
        let d = src.compute_doppler(Vec3::ZERO, Vec3::ZERO, 343.0);
        assert!((d - 1.0).abs() < 1e-6, "disabled doppler should return 1.0");
    }

    #[test]
    fn doppler_clamped_to_range() {
        let mut src = default_source();
        src.position = Vec3::new(10.0, 0.0, 0.0);
        src.velocity = Vec3::new(-340.0, 0.0, 0.0); // nearly speed of sound
        src.doppler_factor = 1.0;
        let d = src.compute_doppler(Vec3::ZERO, Vec3::ZERO, 343.0);
        assert!(d <= 4.0 && d >= 0.25, "doppler should be clamped, got {d}");
    }

    // -- Cone tests --

    #[test]
    fn cone_omnidirectional() {
        let src = default_source(); // TAU cone = omnidirectional
        let atten = src.compute_cone_attenuation(Vec3::new(0.0, 0.0, 5.0));
        assert!((atten - 1.0).abs() < 1e-4, "omnidirectional cone should be 1.0, got {atten}");
    }

    #[test]
    fn cone_narrow_in_front() {
        let mut src = default_source();
        src.direction = Vec3::NEG_Z;
        src.inner_cone_angle = std::f32::consts::FRAC_PI_4; // 45 degrees total
        src.outer_cone_angle = std::f32::consts::FRAC_PI_2; // 90 degrees total
        src.outer_cone_volume = 0.0;

        // Listener directly in front.
        let atten = src.compute_cone_attenuation(Vec3::new(0.0, 0.0, -5.0));
        assert!((atten - 1.0).abs() < 1e-4, "in front of narrow cone should be 1.0");
    }

    #[test]
    fn cone_narrow_behind() {
        let mut src = default_source();
        src.direction = Vec3::NEG_Z;
        src.inner_cone_angle = std::f32::consts::FRAC_PI_4;
        src.outer_cone_angle = std::f32::consts::FRAC_PI_2;
        src.outer_cone_volume = 0.0;

        // Listener behind.
        let atten = src.compute_cone_attenuation(Vec3::new(0.0, 0.0, 5.0));
        assert!((atten).abs() < 0.01, "behind narrow cone should be near 0, got {atten}");
    }

    // -- Listener tests --

    #[test]
    fn listener_right_vector() {
        let listener = AudioListener {
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            ..default_listener()
        };
        let right = listener.right();
        // forward = -Z, up = Y => right = (-Z) x Y = X
        assert!((right - Vec3::X).length() < 1e-4, "right should be +X, got {right}");
    }

    #[test]
    fn listener_look_at() {
        let listener = AudioListener::look_at(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -10.0),
            Vec3::Y,
        );
        assert!((listener.forward - Vec3::NEG_Z).length() < 1e-4);
    }

    // -- Combined spatial test --

    #[test]
    fn compute_spatial_params_smoke() {
        let mut src = default_source();
        src.position = Vec3::new(5.0, 0.0, -3.0);
        src.velocity = Vec3::new(-1.0, 0.0, 0.0);

        let listener = default_listener();

        let result = src.compute_spatial_params(&listener, 343.0);
        assert!(result.volume >= 0.0 && result.volume <= 1.0);
        assert!(result.pan >= -1.0 && result.pan <= 1.0);
        assert!(result.doppler_pitch_ratio > 0.0);
        assert!(result.distance > 0.0);
    }

    // -- Occlusion tests --

    #[test]
    fn occlusion_volume_multiplier() {
        let params = OcclusionParams {
            occlusion: 1.0,
            obstruction: 0.0,
            ..Default::default()
        };
        let vol = params.volume_multiplier();
        assert!((vol - 0.2).abs() < 1e-4, "full occlusion should give 0.2, got {vol}");
    }

    #[test]
    fn occlusion_none() {
        let params = OcclusionParams::default();
        let vol = params.volume_multiplier();
        assert!((vol - 1.0).abs() < 1e-4, "no occlusion should give 1.0");
    }

    #[test]
    fn occlusion_effective_lpf() {
        let params = OcclusionParams {
            occlusion: 1.0,
            lpf_cutoff: 22000.0,
            ..Default::default()
        };
        let cutoff = params.effective_lpf_cutoff();
        assert!((cutoff - 500.0).abs() < 1.0, "full occlusion lpf should be 500 Hz, got {cutoff}");

        let params2 = OcclusionParams::default();
        let cutoff2 = params2.effective_lpf_cutoff();
        assert!((cutoff2 - 22000.0).abs() < 1.0, "no occlusion lpf should be 22000 Hz");
    }
}
