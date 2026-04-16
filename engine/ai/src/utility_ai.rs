//! Utility AI system.
//!
//! Provides a score-based action selection framework where each potential action
//! is scored by evaluating a set of considerations through response curves. The
//! action with the highest composite score is selected for execution.
//!
//! Features:
//! - Multiple response curve types (Linear, Quadratic, Logistic, etc.)
//! - Action momentum to prevent flip-flopping
//! - Cooldown tracking to prevent recently used actions
//! - Debug score breakdowns
//! - Both deterministic and probabilistic action selection

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Response Curves
// ---------------------------------------------------------------------------

/// Describes how an input value [0,1] maps to a score [0,1].
///
/// Response curves shape the AI's preferences by transforming raw input
/// signals into utility scores with different characteristics.
#[derive(Debug, Clone)]
pub enum ResponseCurve {
    /// Linear mapping: `score = slope * input + offset`, clamped to [0,1].
    Linear {
        /// Slope of the linear function.
        slope: f32,
        /// Y-intercept offset.
        offset: f32,
    },
    /// Quadratic mapping: `score = exponent * (input - x_shift)^2 + y_shift`.
    Quadratic {
        /// Exponent (controls curvature direction and steepness).
        exponent: f32,
        /// Horizontal shift.
        x_shift: f32,
        /// Vertical shift.
        y_shift: f32,
    },
    /// Logistic (sigmoid) curve: `score = 1 / (1 + e^(-steepness * (input - midpoint)))`.
    Logistic {
        /// Controls steepness of the S-curve.
        steepness: f32,
        /// The input value at the curve's midpoint (0.5 output).
        midpoint: f32,
    },
    /// Exponential mapping: `score = base^(exponent * input) - 1`, normalized.
    Exponential {
        /// Base of the exponential.
        base: f32,
        /// Exponent multiplier.
        exponent: f32,
    },
    /// Sine-based curve: `score = 0.5 * sin(2*PI*frequency*input + phase) + 0.5`.
    Sine {
        /// Frequency multiplier.
        frequency: f32,
        /// Phase offset in radians.
        phase: f32,
    },
    /// Step function: below `threshold` returns `low`, above returns `high`.
    Step {
        /// Input threshold.
        threshold: f32,
        /// Output value below threshold.
        low: f32,
        /// Output value at or above threshold.
        high: f32,
    },
    /// Custom curve defined by control points with linear interpolation.
    ///
    /// Points should be sorted by x-coordinate. Values between points are
    /// linearly interpolated; values outside the range use the nearest endpoint.
    Custom(Vec<(f32, f32)>),
}

impl ResponseCurve {
    /// Evaluate the response curve for the given input, clamping to [0,1].
    pub fn evaluate(&self, input: f32) -> f32 {
        let raw = match self {
            ResponseCurve::Linear { slope, offset } => {
                slope * input + offset
            }
            ResponseCurve::Quadratic {
                exponent,
                x_shift,
                y_shift,
            } => {
                let shifted = input - x_shift;
                exponent * shifted * shifted + y_shift
            }
            ResponseCurve::Logistic {
                steepness,
                midpoint,
            } => {
                let e = std::f32::consts::E;
                1.0 / (1.0 + e.powf(-steepness * (input - midpoint)))
            }
            ResponseCurve::Exponential { base, exponent } => {
                let max_val = base.powf(*exponent) - 1.0;
                if max_val.abs() < 1e-8 {
                    input
                } else {
                    (base.powf(exponent * input) - 1.0) / max_val
                }
            }
            ResponseCurve::Sine { frequency, phase } => {
                let pi2 = std::f32::consts::PI * 2.0;
                0.5 * (pi2 * frequency * input + phase).sin() + 0.5
            }
            ResponseCurve::Step {
                threshold,
                low,
                high,
            } => {
                if input < *threshold {
                    *low
                } else {
                    *high
                }
            }
            ResponseCurve::Custom(points) => {
                evaluate_custom_curve(points, input)
            }
        };
        raw.clamp(0.0, 1.0)
    }

    /// Create a simple linear curve from 0 to 1.
    pub fn linear() -> Self {
        ResponseCurve::Linear {
            slope: 1.0,
            offset: 0.0,
        }
    }

    /// Create an inverted linear curve from 1 to 0.
    pub fn linear_inverted() -> Self {
        ResponseCurve::Linear {
            slope: -1.0,
            offset: 1.0,
        }
    }

    /// Create a quadratic curve (parabola opening upward).
    pub fn quadratic() -> Self {
        ResponseCurve::Quadratic {
            exponent: 1.0,
            x_shift: 0.0,
            y_shift: 0.0,
        }
    }

    /// Create a logistic (sigmoid) curve with default parameters.
    pub fn logistic() -> Self {
        ResponseCurve::Logistic {
            steepness: 10.0,
            midpoint: 0.5,
        }
    }

    /// Create a step function at the given threshold.
    pub fn step(threshold: f32) -> Self {
        ResponseCurve::Step {
            threshold,
            low: 0.0,
            high: 1.0,
        }
    }
}

/// Linearly interpolate between control points of a custom curve.
fn evaluate_custom_curve(points: &[(f32, f32)], input: f32) -> f32 {
    if points.is_empty() {
        return 0.0;
    }
    if points.len() == 1 {
        return points[0].1;
    }

    // Clamp to endpoints.
    if input <= points[0].0 {
        return points[0].1;
    }
    let last = points.len() - 1;
    if input >= points[last].0 {
        return points[last].1;
    }

    // Find the two surrounding points and interpolate.
    for i in 0..last {
        let (x0, y0) = points[i];
        let (x1, y1) = points[i + 1];
        if input >= x0 && input <= x1 {
            let range = x1 - x0;
            if range.abs() < 1e-8 {
                return y0;
            }
            let t = (input - x0) / range;
            return y0 + t * (y1 - y0);
        }
    }
    points[last].1
}

// ---------------------------------------------------------------------------
// Consideration
// ---------------------------------------------------------------------------

/// A single factor that contributes to an action's score.
///
/// Each consideration samples an input value from the game world (via a
/// closure), transforms it through a response curve, and weights the result.
pub struct Consideration {
    /// Human-readable name for debugging.
    pub name: String,
    /// Closure that reads a value from the game context and returns it
    /// normalized to [0, 1].
    pub input_source: Box<dyn Fn(&UtilityContext) -> f32 + Send + Sync>,
    /// The response curve that maps input to score.
    pub response_curve: ResponseCurve,
    /// Weight multiplier for this consideration (default 1.0).
    pub weight: f32,
}

impl Consideration {
    /// Creates a new consideration.
    pub fn new(
        name: impl Into<String>,
        input_source: impl Fn(&UtilityContext) -> f32 + Send + Sync + 'static,
        response_curve: ResponseCurve,
    ) -> Self {
        Self {
            name: name.into(),
            input_source: Box::new(input_source),
            response_curve,
            weight: 1.0,
        }
    }

    /// Sets the weight for this consideration.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Evaluate this consideration: sample input, apply response curve, apply weight.
    pub fn evaluate(&self, context: &UtilityContext) -> f32 {
        let input = (self.input_source)(context);
        let input_clamped = input.clamp(0.0, 1.0);
        let curved = self.response_curve.evaluate(input_clamped);
        curved * self.weight
    }
}

impl std::fmt::Debug for Consideration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Consideration")
            .field("name", &self.name)
            .field("response_curve", &self.response_curve)
            .field("weight", &self.weight)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// An action that the AI can choose to perform.
///
/// Each action has a set of considerations that determine its desirability.
/// The final score is the product of all consideration scores, multiplied
/// by the action's weight.
pub struct Action {
    /// Human-readable name.
    pub name: String,
    /// List of considerations that determine this action's score.
    pub considerations: Vec<Consideration>,
    /// Weight multiplier applied to the final score.
    pub weight: f32,
    /// Cooldown duration in seconds. While on cooldown, the action scores 0.
    pub cooldown: f32,
    /// Optional callback to execute when this action is selected.
    pub on_execute: Option<Box<dyn Fn(&mut UtilityContext) + Send + Sync>>,
}

impl Action {
    /// Creates a new action with no considerations.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            considerations: Vec::new(),
            weight: 1.0,
            cooldown: 0.0,
            on_execute: None,
        }
    }

    /// Adds a consideration to this action.
    pub fn with_consideration(mut self, consideration: Consideration) -> Self {
        self.considerations.push(consideration);
        self
    }

    /// Sets the weight for this action.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Sets the cooldown duration.
    pub fn with_cooldown(mut self, cooldown_secs: f32) -> Self {
        self.cooldown = cooldown_secs;
        self
    }

    /// Sets the execution callback.
    pub fn with_execute(
        mut self,
        callback: impl Fn(&mut UtilityContext) + Send + Sync + 'static,
    ) -> Self {
        self.on_execute = Some(Box::new(callback));
        self
    }
}

impl std::fmt::Debug for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Action")
            .field("name", &self.name)
            .field("considerations_count", &self.considerations.len())
            .field("weight", &self.weight)
            .field("cooldown", &self.cooldown)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// UtilityContext
// ---------------------------------------------------------------------------

/// Context passed to consideration input sources and action callbacks.
///
/// Contains arbitrary key-value data that the game world populates before
/// utility evaluation.
pub struct UtilityContext {
    /// Float values (health %, distance, ammo %, etc.).
    pub floats: HashMap<String, f32>,
    /// Boolean flags.
    pub bools: HashMap<String, bool>,
    /// Integer values.
    pub ints: HashMap<String, i64>,
    /// String values.
    pub strings: HashMap<String, String>,
    /// Delta time for the current frame.
    pub dt: f32,
    /// Total elapsed time.
    pub elapsed: f32,
}

impl UtilityContext {
    /// Creates a new empty context.
    pub fn new() -> Self {
        Self {
            floats: HashMap::new(),
            bools: HashMap::new(),
            ints: HashMap::new(),
            strings: HashMap::new(),
            dt: 0.0,
            elapsed: 0.0,
        }
    }

    /// Sets a float value.
    pub fn set_float(&mut self, key: impl Into<String>, value: f32) {
        self.floats.insert(key.into(), value);
    }

    /// Gets a float value, defaulting to 0.0.
    pub fn get_float(&self, key: &str) -> f32 {
        self.floats.get(key).copied().unwrap_or(0.0)
    }

    /// Sets a boolean value.
    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) {
        self.bools.insert(key.into(), value);
    }

    /// Gets a boolean value, defaulting to false.
    pub fn get_bool(&self, key: &str) -> bool {
        self.bools.get(key).copied().unwrap_or(false)
    }

    /// Sets an integer value.
    pub fn set_int(&mut self, key: impl Into<String>, value: i64) {
        self.ints.insert(key.into(), value);
    }

    /// Gets an integer value, defaulting to 0.
    pub fn get_int(&self, key: &str) -> i64 {
        self.ints.get(key).copied().unwrap_or(0)
    }

    /// Sets a string value.
    pub fn set_string(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.strings.insert(key.into(), value.into());
    }

    /// Gets a string reference.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.strings.get(key).map(|s| s.as_str())
    }
}

impl Default for UtilityContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ActionScoreDebug
// ---------------------------------------------------------------------------

/// Debug information for a single action's scoring breakdown.
#[derive(Debug, Clone)]
pub struct ActionScoreDebug {
    /// Name of the action.
    pub action_name: String,
    /// Final composite score.
    pub final_score: f32,
    /// Individual consideration scores: (name, input, curved_score, weighted_score).
    pub consideration_scores: Vec<ConsiderationScoreDebug>,
    /// Whether the action was on cooldown.
    pub on_cooldown: bool,
    /// Momentum bonus applied.
    pub momentum_bonus: f32,
}

/// Debug information for a single consideration's scoring.
#[derive(Debug, Clone)]
pub struct ConsiderationScoreDebug {
    /// Consideration name.
    pub name: String,
    /// Raw input value [0,1].
    pub input: f32,
    /// Score after response curve.
    pub curved_score: f32,
    /// Score after weight multiplication.
    pub weighted_score: f32,
}

// ---------------------------------------------------------------------------
// UtilityAI
// ---------------------------------------------------------------------------

/// Score-based action selection system.
///
/// The Utility AI evaluates all available actions by scoring their
/// considerations and selects the best one. It supports:
///
/// - **Multiplicative scoring**: Each action's score is the product of its
///   consideration scores, preventing any single bad factor from being ignored.
/// - **Compensation factor**: Applies Rescorla-Wagner-like compensation so
///   actions with more considerations aren't penalized.
/// - **Momentum**: Bonus for continuing the current action to prevent
///   rapid switching (flip-flopping).
/// - **Cooldowns**: Prevent recently used actions from being selected.
/// - **Debug logging**: Full score breakdowns for tuning.
pub struct UtilityAI {
    /// All available actions.
    pub actions: Vec<Action>,
    /// Remaining cooldown time for each action (by name).
    cooldowns: HashMap<String, f32>,
    /// The name of the currently active action (for momentum).
    current_action: Option<String>,
    /// Momentum bonus added to the current action's score.
    pub momentum_bonus: f32,
    /// Whether to collect debug information during evaluation.
    pub debug_enabled: bool,
    /// Last evaluation's debug info (populated when debug_enabled is true).
    last_debug: Vec<ActionScoreDebug>,
    /// Minimum score threshold; actions below this are not considered.
    pub score_threshold: f32,
    /// Counter for deterministic pseudo-random selection.
    rng_counter: u32,
}

impl UtilityAI {
    /// Creates a new Utility AI system with no actions.
    pub fn new() -> Self {
        Self {
            actions: Vec::new(),
            cooldowns: HashMap::new(),
            current_action: None,
            momentum_bonus: 0.1,
            debug_enabled: false,
            last_debug: Vec::new(),
            score_threshold: 0.0,
            rng_counter: 12345,
        }
    }

    /// Adds an action to the system.
    pub fn add_action(&mut self, action: Action) {
        self.actions.push(action);
    }

    /// Set the currently active action name (e.g., when loading a saved state).
    pub fn set_current_action(&mut self, name: impl Into<String>) {
        self.current_action = Some(name.into());
    }

    /// Clear the current action.
    pub fn clear_current_action(&mut self) {
        self.current_action = None;
    }

    /// Update cooldowns. Call once per frame with delta time.
    pub fn update_cooldowns(&mut self, dt: f32) {
        let mut to_remove = Vec::new();
        for (name, remaining) in &mut self.cooldowns {
            *remaining -= dt;
            if *remaining <= 0.0 {
                to_remove.push(name.clone());
            }
        }
        for name in to_remove {
            self.cooldowns.remove(&name);
        }
    }

    /// Evaluate all actions and return their scores.
    ///
    /// Each action's score is the product of all its consideration scores,
    /// modified by the compensation factor, action weight, and momentum.
    pub fn evaluate_all(&mut self, context: &UtilityContext) -> Vec<(usize, f32)> {
        let mut scores = Vec::with_capacity(self.actions.len());
        let mut debug_entries = if self.debug_enabled {
            Vec::with_capacity(self.actions.len())
        } else {
            Vec::new()
        };

        for (idx, action) in self.actions.iter().enumerate() {
            // Check cooldown.
            let on_cooldown = self.cooldowns.contains_key(&action.name);
            if on_cooldown {
                if self.debug_enabled {
                    debug_entries.push(ActionScoreDebug {
                        action_name: action.name.clone(),
                        final_score: 0.0,
                        consideration_scores: Vec::new(),
                        on_cooldown: true,
                        momentum_bonus: 0.0,
                    });
                }
                scores.push((idx, 0.0));
                continue;
            }

            let (score, consideration_debug) =
                self.evaluate_action(action, context);

            // Apply momentum bonus.
            let momentum = if let Some(ref current) = self.current_action {
                if *current == action.name {
                    self.momentum_bonus
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let final_score = (score + momentum).clamp(0.0, 1.0);

            if self.debug_enabled {
                debug_entries.push(ActionScoreDebug {
                    action_name: action.name.clone(),
                    final_score,
                    consideration_scores: consideration_debug,
                    on_cooldown: false,
                    momentum_bonus: momentum,
                });
            }

            scores.push((idx, final_score));
        }

        if self.debug_enabled {
            self.last_debug = debug_entries;
        }

        scores
    }

    /// Evaluate a single action and return its score, plus debug info.
    fn evaluate_action(
        &self,
        action: &Action,
        context: &UtilityContext,
    ) -> (f32, Vec<ConsiderationScoreDebug>) {
        if action.considerations.is_empty() {
            return (action.weight, Vec::new());
        }

        let num_considerations = action.considerations.len();
        let mut score = 1.0f32;
        let mut debug_list = Vec::with_capacity(num_considerations);

        for consideration in &action.considerations {
            let input = (consideration.input_source)(context).clamp(0.0, 1.0);
            let curved = consideration.response_curve.evaluate(input);
            let weighted = curved * consideration.weight;

            debug_list.push(ConsiderationScoreDebug {
                name: consideration.name.clone(),
                input,
                curved_score: curved,
                weighted_score: weighted,
            });

            score *= weighted;

            // Early out: if score is 0, no need to continue.
            if score <= 0.0 {
                break;
            }
        }

        // Apply Rescorla-Wagner compensation factor to prevent actions with
        // more considerations from being penalized:
        //   compensation = 1 - (1/num_considerations)
        //   modified_score = score + (1 - score) * compensation
        // This lifts the geometric mean toward the arithmetic mean.
        if num_considerations > 1 && score > 0.0 {
            let modification = 1.0 - (1.0 / num_considerations as f32);
            score += (1.0 - score) * modification;
            // Re-raise to preserve ordering intent.
            score = score.powf(1.0 / num_considerations as f32);
        }

        // Apply action weight.
        score *= action.weight;

        (score.clamp(0.0, 1.0), debug_list)
    }

    /// Select the action with the highest score.
    ///
    /// Returns the index of the best action, or `None` if no action scores
    /// above the threshold.
    pub fn select_best_action(&mut self, context: &UtilityContext) -> Option<usize> {
        let scores = self.evaluate_all(context);

        let mut best_idx = None;
        let mut best_score = self.score_threshold;

        for (idx, score) in &scores {
            if *score > best_score {
                best_score = *score;
                best_idx = Some(*idx);
            }
        }

        if let Some(idx) = best_idx {
            let action_name = self.actions[idx].name.clone();
            let cooldown = self.actions[idx].cooldown;

            // Start cooldown for the selected action.
            if cooldown > 0.0 {
                self.cooldowns.insert(action_name.clone(), cooldown);
            }

            self.current_action = Some(action_name);

            if self.debug_enabled {
                self.log_debug_info();
            }
        }

        best_idx
    }

    /// Select an action probabilistically, weighted by score.
    ///
    /// Actions with higher scores have a proportionally higher chance of
    /// being selected. Uses a deterministic pseudo-random scheme.
    pub fn select_weighted_action(&mut self, context: &UtilityContext) -> Option<usize> {
        let scores = self.evaluate_all(context);

        let total: f32 = scores.iter().map(|(_, s)| *s).sum();
        if total <= 0.0 {
            return None;
        }

        // Deterministic pseudo-random selection.
        self.rng_counter = self.rng_counter.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = (self.rng_counter as f32 / u32::MAX as f32) * total;

        let mut accumulated = 0.0;
        for (idx, score) in &scores {
            accumulated += score;
            if accumulated >= r {
                let action_name = self.actions[*idx].name.clone();
                let cooldown = self.actions[*idx].cooldown;

                if cooldown > 0.0 {
                    self.cooldowns.insert(action_name.clone(), cooldown);
                }

                self.current_action = Some(action_name);
                return Some(*idx);
            }
        }

        // Fallback: return the last non-zero action.
        scores.iter().rev().find(|(_, s)| *s > 0.0).map(|(i, _)| *i)
    }

    /// Execute the given action's callback, if it has one.
    pub fn execute_action(&self, action_idx: usize, context: &mut UtilityContext) {
        if action_idx < self.actions.len() {
            if let Some(ref callback) = self.actions[action_idx].on_execute {
                callback(context);
            }
        }
    }

    /// Select the best action and immediately execute its callback.
    pub fn select_and_execute(&mut self, context: &mut UtilityContext) -> Option<usize> {
        let best = self.select_best_action(context);
        if let Some(idx) = best {
            self.execute_action(idx, context);
        }
        best
    }

    /// Returns the debug info from the last evaluation.
    pub fn last_debug_info(&self) -> &[ActionScoreDebug] {
        &self.last_debug
    }

    /// Log debug info for the last evaluation.
    fn log_debug_info(&self) {
        for entry in &self.last_debug {
            if entry.on_cooldown {
                log::debug!(
                    "  [UtilityAI] {} = 0.00 (on cooldown)",
                    entry.action_name
                );
                continue;
            }
            log::debug!(
                "  [UtilityAI] {} = {:.4} (momentum: {:.2})",
                entry.action_name,
                entry.final_score,
                entry.momentum_bonus
            );
            for c in &entry.consideration_scores {
                log::debug!(
                    "    - {} | input={:.3} curved={:.3} weighted={:.3}",
                    c.name,
                    c.input,
                    c.curved_score,
                    c.weighted_score
                );
            }
        }
    }

    /// Returns the name of the currently active action.
    pub fn current_action_name(&self) -> Option<&str> {
        self.current_action.as_deref()
    }

    /// Check if a specific action is on cooldown.
    pub fn is_on_cooldown(&self, action_name: &str) -> bool {
        self.cooldowns.contains_key(action_name)
    }

    /// Get remaining cooldown time for an action.
    pub fn cooldown_remaining(&self, action_name: &str) -> f32 {
        self.cooldowns.get(action_name).copied().unwrap_or(0.0)
    }

    /// Force an action off cooldown.
    pub fn clear_cooldown(&mut self, action_name: &str) {
        self.cooldowns.remove(action_name);
    }

    /// Clear all cooldowns.
    pub fn clear_all_cooldowns(&mut self) {
        self.cooldowns.clear();
    }

    /// Returns the number of registered actions.
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }
}

impl Default for UtilityAI {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Builder for creating utility AI configurations fluently.
pub struct UtilityAIBuilder {
    ai: UtilityAI,
}

impl UtilityAIBuilder {
    /// Start building a new utility AI system.
    pub fn new() -> Self {
        Self {
            ai: UtilityAI::new(),
        }
    }

    /// Set the momentum bonus.
    pub fn momentum(mut self, bonus: f32) -> Self {
        self.ai.momentum_bonus = bonus;
        self
    }

    /// Enable or disable debug mode.
    pub fn debug(mut self, enabled: bool) -> Self {
        self.ai.debug_enabled = enabled;
        self
    }

    /// Set the minimum score threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.ai.score_threshold = threshold;
        self
    }

    /// Add an action.
    pub fn action(mut self, action: Action) -> Self {
        self.ai.add_action(action);
        self
    }

    /// Build the utility AI system.
    pub fn build(self) -> UtilityAI {
        self.ai
    }
}

impl Default for UtilityAIBuilder {
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
    fn test_response_curve_linear() {
        let curve = ResponseCurve::linear();
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.001);
        assert!((curve.evaluate(0.5) - 0.5).abs() < 0.001);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_response_curve_linear_inverted() {
        let curve = ResponseCurve::linear_inverted();
        assert!((curve.evaluate(0.0) - 1.0).abs() < 0.001);
        assert!((curve.evaluate(1.0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_response_curve_linear_clamping() {
        let curve = ResponseCurve::Linear {
            slope: 2.0,
            offset: 0.0,
        };
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.001);
        // 2.0 * 0.6 = 1.2 -> clamped to 1.0
        assert!((curve.evaluate(0.6) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_response_curve_quadratic() {
        let curve = ResponseCurve::quadratic();
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.001);
        assert!((curve.evaluate(0.5) - 0.25).abs() < 0.001);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_response_curve_logistic() {
        let curve = ResponseCurve::logistic();
        // At midpoint, should be ~0.5.
        assert!((curve.evaluate(0.5) - 0.5).abs() < 0.01);
        // At 0, should be close to 0.
        assert!(curve.evaluate(0.0) < 0.02);
        // At 1, should be close to 1.
        assert!(curve.evaluate(1.0) > 0.98);
    }

    #[test]
    fn test_response_curve_step() {
        let curve = ResponseCurve::step(0.5);
        assert!((curve.evaluate(0.3) - 0.0).abs() < 0.001);
        assert!((curve.evaluate(0.5) - 1.0).abs() < 0.001);
        assert!((curve.evaluate(0.7) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_response_curve_sine() {
        let curve = ResponseCurve::Sine {
            frequency: 1.0,
            phase: 0.0,
        };
        // sin(0) = 0, so score = 0.5*0 + 0.5 = 0.5
        assert!((curve.evaluate(0.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_response_curve_custom() {
        let curve = ResponseCurve::Custom(vec![
            (0.0, 0.0),
            (0.5, 1.0),
            (1.0, 0.5),
        ]);
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.001);
        assert!((curve.evaluate(0.25) - 0.5).abs() < 0.001);
        assert!((curve.evaluate(0.5) - 1.0).abs() < 0.001);
        assert!((curve.evaluate(0.75) - 0.75).abs() < 0.001);
        assert!((curve.evaluate(1.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_response_curve_exponential() {
        let curve = ResponseCurve::Exponential {
            base: 2.0,
            exponent: 1.0,
        };
        // At 0: (2^0 - 1)/(2^1 - 1) = 0/1 = 0.
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.01);
        // At 1: (2^1 - 1)/(2^1 - 1) = 1.0.
        assert!((curve.evaluate(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_consideration_evaluation() {
        let ctx = {
            let mut c = UtilityContext::new();
            c.set_float("health", 0.7);
            c
        };

        let consideration = Consideration::new(
            "health_check",
            |ctx| ctx.get_float("health"),
            ResponseCurve::linear(),
        );

        let score = consideration.evaluate(&ctx);
        assert!((score - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_consideration_with_weight() {
        let ctx = {
            let mut c = UtilityContext::new();
            c.set_float("health", 0.5);
            c
        };

        let consideration = Consideration::new(
            "health",
            |ctx| ctx.get_float("health"),
            ResponseCurve::linear(),
        )
        .with_weight(2.0);

        let score = consideration.evaluate(&ctx);
        // 0.5 * 2.0 = 1.0 (clamped by weight in evaluate, but weight>1 can push above 1)
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_utility_ai_select_best() {
        let mut ai = UtilityAI::new();

        // Action 1: high health -> attack
        ai.add_action(
            Action::new("attack")
                .with_consideration(Consideration::new(
                    "health",
                    |ctx| ctx.get_float("health"),
                    ResponseCurve::linear(),
                ))
                .with_consideration(Consideration::new(
                    "enemy_near",
                    |ctx| ctx.get_float("enemy_distance_inv"),
                    ResponseCurve::linear(),
                )),
        );

        // Action 2: low health -> flee
        ai.add_action(
            Action::new("flee")
                .with_consideration(Consideration::new(
                    "low_health",
                    |ctx| ctx.get_float("health"),
                    ResponseCurve::linear_inverted(),
                )),
        );

        // Context: high health, enemy is near.
        let mut ctx = UtilityContext::new();
        ctx.set_float("health", 0.9);
        ctx.set_float("enemy_distance_inv", 0.8);

        let best = ai.select_best_action(&ctx);
        assert!(best.is_some());
        let best_idx = best.unwrap();
        assert_eq!(ai.actions[best_idx].name, "attack");
    }

    #[test]
    fn test_utility_ai_select_flee_when_low_health() {
        let mut ai = UtilityAI::new();

        ai.add_action(
            Action::new("attack")
                .with_consideration(Consideration::new(
                    "health",
                    |ctx| ctx.get_float("health"),
                    ResponseCurve::linear(),
                )),
        );

        ai.add_action(
            Action::new("flee")
                .with_consideration(Consideration::new(
                    "low_health",
                    |ctx| ctx.get_float("health"),
                    ResponseCurve::linear_inverted(),
                )),
        );

        let mut ctx = UtilityContext::new();
        ctx.set_float("health", 0.1);

        let best = ai.select_best_action(&ctx);
        assert!(best.is_some());
        let best_idx = best.unwrap();
        assert_eq!(ai.actions[best_idx].name, "flee");
    }

    #[test]
    fn test_utility_ai_cooldown() {
        let mut ai = UtilityAI::new();

        ai.add_action(
            Action::new("special_attack")
                .with_cooldown(5.0)
                .with_consideration(Consideration::new(
                    "always_high",
                    |_| 1.0,
                    ResponseCurve::linear(),
                )),
        );

        ai.add_action(
            Action::new("basic_attack")
                .with_consideration(Consideration::new(
                    "moderate",
                    |_| 0.5,
                    ResponseCurve::linear(),
                )),
        );

        let ctx = UtilityContext::new();

        // First selection: special_attack wins.
        let best = ai.select_best_action(&ctx).unwrap();
        assert_eq!(ai.actions[best].name, "special_attack");

        // Now special_attack is on cooldown.
        assert!(ai.is_on_cooldown("special_attack"));

        // Second selection: basic_attack should win.
        let best = ai.select_best_action(&ctx).unwrap();
        assert_eq!(ai.actions[best].name, "basic_attack");

        // Update cooldowns to expire.
        ai.update_cooldowns(6.0);
        assert!(!ai.is_on_cooldown("special_attack"));
    }

    #[test]
    fn test_utility_ai_momentum() {
        let mut ai = UtilityAI::new();
        ai.momentum_bonus = 0.3;

        ai.add_action(
            Action::new("action_a")
                .with_consideration(Consideration::new(
                    "score",
                    |_| 0.5,
                    ResponseCurve::linear(),
                )),
        );

        ai.add_action(
            Action::new("action_b")
                .with_consideration(Consideration::new(
                    "score",
                    |_| 0.55,
                    ResponseCurve::linear(),
                )),
        );

        let ctx = UtilityContext::new();

        // First selection: action_b wins (0.55 > 0.5).
        let best = ai.select_best_action(&ctx).unwrap();
        assert_eq!(ai.actions[best].name, "action_b");

        // Now action_b has momentum. Even though scores are similar,
        // action_b should still win.
        let best = ai.select_best_action(&ctx).unwrap();
        assert_eq!(ai.actions[best].name, "action_b");
    }

    #[test]
    fn test_utility_ai_weighted_selection() {
        let mut ai = UtilityAI::new();

        ai.add_action(
            Action::new("high_score")
                .with_consideration(Consideration::new(
                    "always",
                    |_| 0.9,
                    ResponseCurve::linear(),
                )),
        );

        ai.add_action(
            Action::new("low_score")
                .with_consideration(Consideration::new(
                    "always",
                    |_| 0.1,
                    ResponseCurve::linear(),
                )),
        );

        let ctx = UtilityContext::new();

        // Run weighted selection many times. The high-score action should
        // be selected most of the time.
        let mut high_count = 0;
        for _ in 0..100 {
            if let Some(idx) = ai.select_weighted_action(&ctx) {
                if ai.actions[idx].name == "high_score" {
                    high_count += 1;
                }
            }
            // Clear cooldowns since these actions have none, but let's be safe.
            ai.clear_all_cooldowns();
        }

        assert!(
            high_count > 50,
            "Expected high_score to be selected most often, got {high_count}/100"
        );
    }

    #[test]
    fn test_utility_ai_debug_mode() {
        let mut ai = UtilityAI::new();
        ai.debug_enabled = true;

        ai.add_action(
            Action::new("test_action")
                .with_consideration(Consideration::new(
                    "test_input",
                    |ctx| ctx.get_float("val"),
                    ResponseCurve::linear(),
                )),
        );

        let mut ctx = UtilityContext::new();
        ctx.set_float("val", 0.7);

        ai.select_best_action(&ctx);

        let debug = ai.last_debug_info();
        assert_eq!(debug.len(), 1);
        assert_eq!(debug[0].action_name, "test_action");
        assert!(!debug[0].on_cooldown);
        assert_eq!(debug[0].consideration_scores.len(), 1);
        assert!((debug[0].consideration_scores[0].input - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_utility_ai_no_actions() {
        let mut ai = UtilityAI::new();
        let ctx = UtilityContext::new();
        assert!(ai.select_best_action(&ctx).is_none());
    }

    #[test]
    fn test_utility_context() {
        let mut ctx = UtilityContext::new();
        ctx.set_float("health", 0.5);
        ctx.set_bool("is_visible", true);
        ctx.set_int("ammo", 30);
        ctx.set_string("state", "idle");

        assert_eq!(ctx.get_float("health"), 0.5);
        assert!(ctx.get_bool("is_visible"));
        assert_eq!(ctx.get_int("ammo"), 30);
        assert_eq!(ctx.get_string("state"), Some("idle"));

        // Defaults.
        assert_eq!(ctx.get_float("missing"), 0.0);
        assert!(!ctx.get_bool("missing"));
        assert_eq!(ctx.get_int("missing"), 0);
        assert_eq!(ctx.get_string("missing"), None);
    }

    #[test]
    fn test_utility_ai_builder() {
        let ai = UtilityAIBuilder::new()
            .momentum(0.2)
            .debug(true)
            .threshold(0.1)
            .action(
                Action::new("patrol")
                    .with_consideration(Consideration::new(
                        "default",
                        |_| 0.3,
                        ResponseCurve::linear(),
                    )),
            )
            .build();

        assert_eq!(ai.action_count(), 1);
        assert_eq!(ai.momentum_bonus, 0.2);
        assert!(ai.debug_enabled);
        assert!((ai.score_threshold - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_utility_ai_execute_callback() {
        let mut ai = UtilityAI::new();

        ai.add_action(
            Action::new("increment")
                .with_consideration(Consideration::new(
                    "always",
                    |_| 1.0,
                    ResponseCurve::linear(),
                ))
                .with_execute(|ctx| {
                    let current = ctx.get_int("counter");
                    ctx.set_int("counter", current + 1);
                }),
        );

        let mut ctx = UtilityContext::new();
        ctx.set_int("counter", 0);

        ai.select_and_execute(&mut ctx);
        assert_eq!(ctx.get_int("counter"), 1);

        ai.select_and_execute(&mut ctx);
        assert_eq!(ctx.get_int("counter"), 2);
    }

    #[test]
    fn test_utility_ai_clear_cooldown() {
        let mut ai = UtilityAI::new();

        ai.add_action(
            Action::new("ability")
                .with_cooldown(10.0)
                .with_consideration(Consideration::new(
                    "always",
                    |_| 1.0,
                    ResponseCurve::linear(),
                )),
        );

        let ctx = UtilityContext::new();
        ai.select_best_action(&ctx);
        assert!(ai.is_on_cooldown("ability"));

        ai.clear_cooldown("ability");
        assert!(!ai.is_on_cooldown("ability"));
    }

    #[test]
    fn test_utility_ai_score_threshold() {
        let mut ai = UtilityAI::new();
        ai.score_threshold = 0.5;

        ai.add_action(
            Action::new("weak_action")
                .with_consideration(Consideration::new(
                    "low",
                    |_| 0.2,
                    ResponseCurve::linear(),
                )),
        );

        let ctx = UtilityContext::new();
        // Score is below threshold so no action should be selected.
        let result = ai.select_best_action(&ctx);
        assert!(result.is_none());
    }
}
