//! # L-Systems (Lindenmayer Systems)
//!
//! A string rewriting system for procedural generation of branching structures,
//! fractals, and organic forms. An L-system consists of an axiom (initial string)
//! and a set of production rules that replace characters in parallel at each
//! iteration.
//!
//! ## Turtle interpretation
//!
//! The rewritten string is interpreted as a sequence of drawing commands:
//! - `F` — move forward (draw line)
//! - `f` — move forward (no draw)
//! - `+` — turn right by angle
//! - `-` — turn left by angle
//! - `[` — push state (start branch)
//! - `]` — pop state (end branch)
//! - `|` — turn 180 degrees
//!
//! 3D turtle adds:
//! - `^` — pitch up
//! - `&` — pitch down
//! - `\\` — roll clockwise
//! - `/` — roll counter-clockwise

use genovo_core::Rng;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===========================================================================
// Line segments (output geometry)
// ===========================================================================

/// A 2D line segment produced by turtle interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LineSegment2D {
    /// Start point.
    pub start: Vec2,
    /// End point.
    pub end: Vec2,
    /// Depth in the branching hierarchy (0 = trunk).
    pub depth: u32,
}

/// A 3D line segment produced by 3D turtle interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LineSegment3D {
    /// Start point.
    pub start: Vec3,
    /// End point.
    pub end: Vec3,
    /// Depth in the branching hierarchy.
    pub depth: u32,
    /// Thickness/radius at this segment (tapers with depth).
    pub thickness: f32,
}

// ===========================================================================
// Turtle commands
// ===========================================================================

/// Commands that can appear in an L-system string for turtle interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TurtleCommand {
    /// Move forward and draw a line (`F`).
    Forward,
    /// Move forward without drawing (`f`).
    ForwardNoDraw,
    /// Turn right by the configured angle (`+`).
    TurnRight,
    /// Turn left by the configured angle (`-`).
    TurnLeft,
    /// Turn 180 degrees (`|`).
    TurnAround,
    /// Push the current state onto the stack (`[`).
    PushState,
    /// Pop the state from the stack (`]`).
    PopState,
    /// Pitch up (3D, `^`).
    PitchUp,
    /// Pitch down (3D, `&`).
    PitchDown,
    /// Roll clockwise (3D, `\`).
    RollCW,
    /// Roll counter-clockwise (3D, `/`).
    RollCCW,
    /// Decrease step size (`<`).
    DecreaseStep,
    /// Increase step size (`>`).
    IncreaseStep,
    /// An unrecognized character (ignored by the turtle).
    Unknown(char),
}

impl TurtleCommand {
    /// Parse a character into a turtle command.
    pub fn from_char(ch: char) -> Self {
        match ch {
            'F' => TurtleCommand::Forward,
            'f' => TurtleCommand::ForwardNoDraw,
            'G' => TurtleCommand::Forward, // Alternative forward.
            '+' => TurtleCommand::TurnRight,
            '-' => TurtleCommand::TurnLeft,
            '|' => TurtleCommand::TurnAround,
            '[' => TurtleCommand::PushState,
            ']' => TurtleCommand::PopState,
            '^' => TurtleCommand::PitchUp,
            '&' => TurtleCommand::PitchDown,
            '\\' => TurtleCommand::RollCW,
            '/' => TurtleCommand::RollCCW,
            '<' => TurtleCommand::DecreaseStep,
            '>' => TurtleCommand::IncreaseStep,
            c => TurtleCommand::Unknown(c),
        }
    }
}

// ===========================================================================
// Production rules
// ===========================================================================

/// A single production rule: replace a character with a string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionRule {
    /// The character to match.
    pub predecessor: char,
    /// The replacement string.
    pub successor: String,
}

/// A stochastic production rule: multiple possible replacements with weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticRule {
    /// The character to match.
    pub predecessor: char,
    /// Possible replacements with their relative weights.
    pub successors: Vec<(f32, String)>,
}

impl StochasticRule {
    /// Create a new stochastic rule.
    pub fn new(predecessor: char) -> Self {
        Self {
            predecessor,
            successors: Vec::new(),
        }
    }

    /// Add a weighted successor.
    pub fn with_successor(mut self, weight: f32, successor: &str) -> Self {
        self.successors.push((weight, successor.to_string()));
        self
    }

    /// Choose a successor based on weighted random selection.
    pub fn choose(&self, rng: &mut Rng) -> &str {
        if self.successors.is_empty() {
            return "";
        }
        let weights: Vec<f32> = self.successors.iter().map(|(w, _)| *w).collect();
        let idx = rng.weighted_pick(&weights);
        &self.successors[idx].1
    }
}

// ===========================================================================
// L-System
// ===========================================================================

/// An L-system string rewriting system.
///
/// Consists of an axiom (initial string) and production rules. At each
/// iteration, all characters in the string are simultaneously replaced
/// according to the rules. Characters without rules are left unchanged
/// (identity rule).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSystem {
    /// Initial string.
    pub axiom: String,

    /// Deterministic production rules.
    pub rules: HashMap<char, String>,

    /// Stochastic rules (used when a random generator is provided).
    #[serde(skip)]
    pub stochastic_rules: HashMap<char, Vec<(f32, String)>>,

    /// Current string state (result of iterations).
    pub current: String,

    /// Number of iterations applied.
    pub iterations: usize,
}

impl LSystem {
    /// Create a new L-system with the given axiom.
    pub fn new(axiom: &str) -> Self {
        Self {
            axiom: axiom.to_string(),
            rules: HashMap::new(),
            stochastic_rules: HashMap::new(),
            current: axiom.to_string(),
            iterations: 0,
        }
    }

    /// Add a deterministic production rule.
    pub fn add_rule(mut self, predecessor: char, successor: &str) -> Self {
        self.rules.insert(predecessor, successor.to_string());
        self
    }

    /// Add a stochastic production rule (multiple weighted outcomes).
    pub fn add_stochastic_rule(mut self, predecessor: char, successors: Vec<(f32, &str)>) -> Self {
        let entries: Vec<(f32, String)> = successors
            .into_iter()
            .map(|(w, s)| (w, s.to_string()))
            .collect();
        self.stochastic_rules.insert(predecessor, entries);
        self
    }

    /// Apply deterministic rules N times, returning the resulting string.
    pub fn iterate(&mut self, n: usize) -> &str {
        for _ in 0..n {
            let mut next = String::with_capacity(self.current.len() * 2);
            for ch in self.current.chars() {
                if let Some(replacement) = self.rules.get(&ch) {
                    next.push_str(replacement);
                } else {
                    next.push(ch);
                }
            }
            self.current = next;
            self.iterations += 1;
        }
        &self.current
    }

    /// Apply rules N times with stochastic selection for rules that have
    /// multiple weighted outcomes.
    pub fn iterate_stochastic(&mut self, n: usize, rng: &mut Rng) -> &str {
        for _ in 0..n {
            let mut next = String::with_capacity(self.current.len() * 2);
            for ch in self.current.chars() {
                if let Some(successors) = self.stochastic_rules.get(&ch) {
                    let weights: Vec<f32> = successors.iter().map(|(w, _)| *w).collect();
                    let idx = rng.weighted_pick(&weights);
                    next.push_str(&successors[idx].1);
                } else if let Some(replacement) = self.rules.get(&ch) {
                    next.push_str(replacement);
                } else {
                    next.push(ch);
                }
            }
            self.current = next;
            self.iterations += 1;
        }
        &self.current
    }

    /// Reset to the axiom.
    pub fn reset(&mut self) {
        self.current = self.axiom.clone();
        self.iterations = 0;
    }

    /// Get the current string length.
    pub fn length(&self) -> usize {
        self.current.len()
    }

    /// Get the current string.
    pub fn current_string(&self) -> &str {
        &self.current
    }
}

// ===========================================================================
// 2D Turtle Interpreter
// ===========================================================================

/// State of the 2D turtle.
#[derive(Debug, Clone, Copy)]
struct TurtleState2D {
    x: f32,
    y: f32,
    angle: f32, // In radians.
    step_size: f32,
    depth: u32,
}

/// A 2D turtle interpreter that converts L-system strings into line geometry.
pub struct TurtleInterpreter {
    /// Step size for forward movement.
    pub step_size: f32,
    /// Turn angle in degrees.
    pub angle_degrees: f32,
    /// Step size multiplier for `<` and `>` commands.
    pub step_scale: f32,
    /// Starting X position.
    pub start_x: f32,
    /// Starting Y position.
    pub start_y: f32,
    /// Starting angle in degrees (0 = right, 90 = up).
    pub start_angle: f32,
}

impl TurtleInterpreter {
    /// Create a new 2D turtle interpreter.
    pub fn new(step_size: f32, angle_degrees: f32) -> Self {
        Self {
            step_size,
            angle_degrees,
            step_scale: 0.8,
            start_x: 0.0,
            start_y: 0.0,
            start_angle: 90.0, // Point upward by default.
        }
    }

    /// Set starting position and return self.
    pub fn with_start(mut self, x: f32, y: f32) -> Self {
        self.start_x = x;
        self.start_y = y;
        self
    }

    /// Set starting angle and return self.
    pub fn with_angle(mut self, angle: f32) -> Self {
        self.start_angle = angle;
        self
    }

    /// Set step size multiplier for `<` and `>` commands.
    pub fn with_step_scale(mut self, scale: f32) -> Self {
        self.step_scale = scale;
        self
    }

    /// Interpret an L-system string and produce 2D line segments.
    pub fn interpret(&self, input: &str) -> Vec<LineSegment2D> {
        let mut segments = Vec::new();
        let mut state = TurtleState2D {
            x: self.start_x,
            y: self.start_y,
            angle: self.start_angle.to_radians(),
            step_size: self.step_size,
            depth: 0,
        };
        let mut stack: Vec<TurtleState2D> = Vec::new();
        let angle_rad = self.angle_degrees.to_radians();

        for ch in input.chars() {
            let cmd = TurtleCommand::from_char(ch);
            match cmd {
                TurtleCommand::Forward | TurtleCommand::ForwardNoDraw => {
                    let dx = state.angle.cos() * state.step_size;
                    let dy = state.angle.sin() * state.step_size;
                    let new_x = state.x + dx;
                    let new_y = state.y + dy;

                    if cmd == TurtleCommand::Forward {
                        segments.push(LineSegment2D {
                            start: Vec2::new(state.x, state.y),
                            end: Vec2::new(new_x, new_y),
                            depth: state.depth,
                        });
                    }

                    state.x = new_x;
                    state.y = new_y;
                }
                TurtleCommand::TurnRight => {
                    state.angle -= angle_rad;
                }
                TurtleCommand::TurnLeft => {
                    state.angle += angle_rad;
                }
                TurtleCommand::TurnAround => {
                    state.angle += std::f32::consts::PI;
                }
                TurtleCommand::PushState => {
                    stack.push(state);
                    state.depth += 1;
                }
                TurtleCommand::PopState => {
                    if let Some(saved) = stack.pop() {
                        state = saved;
                    }
                }
                TurtleCommand::DecreaseStep => {
                    state.step_size *= self.step_scale;
                }
                TurtleCommand::IncreaseStep => {
                    state.step_size /= self.step_scale;
                }
                _ => {} // Ignore unknown/3D commands.
            }
        }

        segments
    }

    /// Compute the bounding box of the generated line segments.
    pub fn bounding_box(segments: &[LineSegment2D]) -> (Vec2, Vec2) {
        if segments.is_empty() {
            return (Vec2::ZERO, Vec2::ZERO);
        }

        let mut min = Vec2::new(f32::MAX, f32::MAX);
        let mut max = Vec2::new(f32::MIN, f32::MIN);

        for seg in segments {
            min.x = min.x.min(seg.start.x).min(seg.end.x);
            min.y = min.y.min(seg.start.y).min(seg.end.y);
            max.x = max.x.max(seg.start.x).max(seg.end.x);
            max.y = max.y.max(seg.start.y).max(seg.end.y);
        }

        (min, max)
    }
}

impl Default for TurtleInterpreter {
    fn default() -> Self {
        Self::new(1.0, 90.0)
    }
}

// ===========================================================================
// 3D Turtle Interpreter
// ===========================================================================

/// State of the 3D turtle.
#[derive(Debug, Clone, Copy)]
struct TurtleState3D {
    position: Vec3,
    heading: Vec3,   // Forward direction.
    left: Vec3,      // Left direction.
    up: Vec3,        // Up direction.
    step_size: f32,
    thickness: f32,
    depth: u32,
}

/// A 3D turtle interpreter that converts L-system strings into 3D line geometry.
///
/// Uses a local coordinate frame (heading, left, up) to support pitch, yaw,
/// and roll operations.
pub struct TurtleInterpreter3D {
    /// Step size for forward movement.
    pub step_size: f32,
    /// Turn angle in degrees.
    pub angle_degrees: f32,
    /// Step size multiplier for `<` and `>` commands.
    pub step_scale: f32,
    /// Thickness multiplier per depth level.
    pub thickness_decay: f32,
    /// Initial trunk thickness.
    pub initial_thickness: f32,
}

impl TurtleInterpreter3D {
    /// Create a new 3D turtle interpreter.
    pub fn new(step_size: f32, angle_degrees: f32) -> Self {
        Self {
            step_size,
            angle_degrees,
            step_scale: 0.8,
            thickness_decay: 0.7,
            initial_thickness: 1.0,
        }
    }

    /// Set thickness parameters.
    pub fn with_thickness(mut self, initial: f32, decay: f32) -> Self {
        self.initial_thickness = initial;
        self.thickness_decay = decay;
        self
    }

    /// Interpret an L-system string and produce 3D line segments.
    pub fn interpret(&self, input: &str) -> Vec<LineSegment3D> {
        let mut segments = Vec::new();
        let mut state = TurtleState3D {
            position: Vec3::ZERO,
            heading: Vec3::Y,  // Initially pointing up.
            left: Vec3::NEG_X, // Left is -X.
            up: Vec3::Z,       // Up is +Z (screen out).
            step_size: self.step_size,
            thickness: self.initial_thickness,
            depth: 0,
        };
        let mut stack: Vec<TurtleState3D> = Vec::new();
        let angle_rad = self.angle_degrees.to_radians();

        for ch in input.chars() {
            let cmd = TurtleCommand::from_char(ch);
            match cmd {
                TurtleCommand::Forward => {
                    let new_pos = state.position + state.heading * state.step_size;
                    segments.push(LineSegment3D {
                        start: state.position,
                        end: new_pos,
                        depth: state.depth,
                        thickness: state.thickness,
                    });
                    state.position = new_pos;
                }
                TurtleCommand::ForwardNoDraw => {
                    state.position += state.heading * state.step_size;
                }
                TurtleCommand::TurnRight => {
                    // Rotate heading around up axis.
                    self.rotate_around_up(&mut state, -angle_rad);
                }
                TurtleCommand::TurnLeft => {
                    self.rotate_around_up(&mut state, angle_rad);
                }
                TurtleCommand::TurnAround => {
                    self.rotate_around_up(&mut state, std::f32::consts::PI);
                }
                TurtleCommand::PitchUp => {
                    self.rotate_around_left(&mut state, angle_rad);
                }
                TurtleCommand::PitchDown => {
                    self.rotate_around_left(&mut state, -angle_rad);
                }
                TurtleCommand::RollCW => {
                    self.rotate_around_heading(&mut state, -angle_rad);
                }
                TurtleCommand::RollCCW => {
                    self.rotate_around_heading(&mut state, angle_rad);
                }
                TurtleCommand::PushState => {
                    stack.push(state);
                    state.depth += 1;
                    state.thickness *= self.thickness_decay;
                    state.step_size *= self.step_scale;
                }
                TurtleCommand::PopState => {
                    if let Some(saved) = stack.pop() {
                        state = saved;
                    }
                }
                TurtleCommand::DecreaseStep => {
                    state.step_size *= self.step_scale;
                }
                TurtleCommand::IncreaseStep => {
                    state.step_size /= self.step_scale;
                }
                TurtleCommand::Unknown(_) => {}
            }
        }

        segments
    }

    /// Rotate the turtle's frame around the up axis (yaw).
    fn rotate_around_up(&self, state: &mut TurtleState3D, angle: f32) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let new_heading = state.heading * cos_a + state.left * sin_a;
        let new_left = state.left * cos_a - state.heading * sin_a;
        state.heading = new_heading.normalize();
        state.left = new_left.normalize();
    }

    /// Rotate the turtle's frame around the left axis (pitch).
    fn rotate_around_left(&self, state: &mut TurtleState3D, angle: f32) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let new_heading = state.heading * cos_a + state.up * sin_a;
        let new_up = state.up * cos_a - state.heading * sin_a;
        state.heading = new_heading.normalize();
        state.up = new_up.normalize();
    }

    /// Rotate the turtle's frame around the heading axis (roll).
    fn rotate_around_heading(&self, state: &mut TurtleState3D, angle: f32) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let new_left = state.left * cos_a + state.up * sin_a;
        let new_up = state.up * cos_a - state.left * sin_a;
        state.left = new_left.normalize();
        state.up = new_up.normalize();
    }

    /// Compute the bounding box of 3D line segments.
    pub fn bounding_box(segments: &[LineSegment3D]) -> (Vec3, Vec3) {
        if segments.is_empty() {
            return (Vec3::ZERO, Vec3::ZERO);
        }

        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for seg in segments {
            min = min.min(seg.start).min(seg.end);
            max = max.max(seg.start).max(seg.end);
        }

        (min, max)
    }
}

impl Default for TurtleInterpreter3D {
    fn default() -> Self {
        Self::new(1.0, 25.0)
    }
}

// ===========================================================================
// Presets
// ===========================================================================

/// Pre-built L-system definitions for common patterns.
pub struct LSystemPresets;

impl LSystemPresets {
    /// Koch snowflake fractal.
    ///
    /// After 4 iterations, produces a classic snowflake outline.
    pub fn koch_snowflake() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("F--F--F")
            .add_rule('F', "F+F--F+F");
        let turtle = TurtleInterpreter::new(1.0, 60.0);
        (system, turtle)
    }

    /// Sierpinski triangle.
    pub fn sierpinski_triangle() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("F-G-G")
            .add_rule('F', "F-G+F+G-F")
            .add_rule('G', "GG");
        let turtle = TurtleInterpreter::new(1.0, 120.0);
        (system, turtle)
    }

    /// Dragon curve.
    pub fn dragon_curve() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("FX")
            .add_rule('X', "X+YF+")
            .add_rule('Y', "-FX-Y");
        let turtle = TurtleInterpreter::new(1.0, 90.0);
        (system, turtle)
    }

    /// Hilbert curve (space-filling).
    pub fn hilbert_curve() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("A")
            .add_rule('A', "-BF+AFA+FB-")
            .add_rule('B', "+AF-BFB-FA+");
        let turtle = TurtleInterpreter::new(1.0, 90.0);
        (system, turtle)
    }

    /// Simple branching tree.
    ///
    /// Produces a 2D tree with symmetric branching.
    pub fn simple_tree() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("F")
            .add_rule('F', "FF+[+F-F-F]-[-F+F+F]");
        let turtle = TurtleInterpreter::new(2.0, 22.5)
            .with_start(0.0, 0.0)
            .with_angle(90.0);
        (system, turtle)
    }

    /// Stochastic tree with variation in branching.
    pub fn stochastic_tree() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("F")
            .add_stochastic_rule('F', vec![
                (0.33, "F[+F]F[-F]F"),
                (0.33, "F[+F]F"),
                (0.34, "F[-F]F"),
            ]);
        let turtle = TurtleInterpreter::new(2.0, 25.7)
            .with_start(0.0, 0.0)
            .with_angle(90.0);
        (system, turtle)
    }

    /// Fern-like plant.
    pub fn fern() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("X")
            .add_rule('X', "F+[[X]-X]-F[-FX]+X")
            .add_rule('F', "FF");
        let turtle = TurtleInterpreter::new(2.0, 25.0)
            .with_start(0.0, 0.0)
            .with_angle(90.0);
        (system, turtle)
    }

    /// Bush-like branching structure.
    pub fn bush() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("F")
            .add_rule('F', "FF-[-F+F+F]+[+F-F-F]");
        let turtle = TurtleInterpreter::new(1.5, 22.5)
            .with_start(0.0, 0.0)
            .with_angle(90.0);
        (system, turtle)
    }

    /// Penrose tiling (P3, kites and darts).
    pub fn penrose_tiling() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("[7]++[7]++[7]++[7]++[7]")
            .add_rule('6', "81++91----71[-81----61]++")
            .add_rule('7', "+81--91[---61--71]+")
            .add_rule('8', "-61++71[+++81++91]-")
            .add_rule('9', "--81++++61[+91++++71]--71")
            .add_rule('1', ""); // 1 is just F drawn.
        let turtle = TurtleInterpreter::new(1.0, 36.0);
        (system, turtle)
    }

    /// Simple 3D tree (branches in 3D space).
    pub fn tree_3d() -> (LSystem, TurtleInterpreter3D) {
        let system = LSystem::new("A")
            .add_rule('A', "F[&+A]F[&-A]+A")
            .add_rule('F', "FF");
        let turtle = TurtleInterpreter3D::new(1.0, 25.7)
            .with_thickness(0.5, 0.65);
        (system, turtle)
    }

    /// City block layout generator.
    ///
    /// Produces a grid-like structure representing city blocks when interpreted
    /// with a 90-degree turtle.
    pub fn city_blocks() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("F+F+F+F")
            .add_rule('F', "FF+F-F+F+FF");
        let turtle = TurtleInterpreter::new(5.0, 90.0)
            .with_start(0.0, 0.0)
            .with_angle(0.0);
        (system, turtle)
    }

    /// Gosper curve (hexagonal space-filling curve).
    pub fn gosper_curve() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("A")
            .add_rule('A', "A-B--B+A++AA+B-")
            .add_rule('B', "+A-BB--B-A++A+B");
        let turtle = TurtleInterpreter::new(1.0, 60.0);
        (system, turtle)
    }

    /// Lévy C curve.
    pub fn levy_c_curve() -> (LSystem, TurtleInterpreter) {
        let system = LSystem::new("F")
            .add_rule('F', "+F--F+");
        let turtle = TurtleInterpreter::new(1.0, 45.0);
        (system, turtle)
    }

    /// Stochastic 3D plant/coral.
    pub fn coral_3d() -> (LSystem, TurtleInterpreter3D) {
        let system = LSystem::new("F")
            .add_stochastic_rule('F', vec![
                (0.4, "F[+F]F[&F]F"),
                (0.3, "F[-F][/F]F"),
                (0.3, "FF[^F][\\F]"),
            ]);
        let turtle = TurtleInterpreter3D::new(1.0, 30.0)
            .with_thickness(0.3, 0.7);
        (system, turtle)
    }
}

// ===========================================================================
// Parametric L-system support
// ===========================================================================

/// A parametric L-system symbol with associated parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricSymbol {
    /// The symbol character.
    pub symbol: char,
    /// Associated numeric parameters.
    pub params: Vec<f32>,
}

impl ParametricSymbol {
    /// Create a new parametric symbol.
    pub fn new(symbol: char, params: Vec<f32>) -> Self {
        Self { symbol, params }
    }
}

/// A parametric L-system that supports symbols with numeric parameters.
///
/// This allows rules like `A(s) -> F(s)[+A(s*0.7)][-A(s*0.7)]` where `s`
/// is a size parameter that shrinks with each iteration.
pub struct ParametricLSystem {
    /// Initial symbols.
    pub axiom: Vec<ParametricSymbol>,
    /// Production rules: maps a character to a function that produces new symbols.
    rules: HashMap<char, Box<dyn Fn(&[f32]) -> Vec<ParametricSymbol> + Send + Sync>>,
    /// Current state.
    pub current: Vec<ParametricSymbol>,
}

impl std::fmt::Debug for ParametricLSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParametricLSystem")
            .field("axiom", &self.axiom)
            .field("rules", &format!("<{} rules>", self.rules.len()))
            .field("current", &self.current)
            .finish()
    }
}

// We need a manual Clone since Box<dyn Fn> doesn't implement Clone.
// The struct already derives Clone above, but the rules field won't compile
// with derive(Clone), so we provide a stub. In practice, ParametricLSystem
// is typically not cloned; users create new instances instead.

impl ParametricLSystem {
    /// Create a new parametric L-system with the given axiom.
    pub fn new(axiom: Vec<ParametricSymbol>) -> Self {
        Self {
            current: axiom.clone(),
            axiom,
            rules: HashMap::new(),
        }
    }

    /// Add a production rule. The closure receives the predecessor's parameters
    /// and returns the successor symbols.
    pub fn add_rule<F>(&mut self, predecessor: char, rule: F)
    where
        F: Fn(&[f32]) -> Vec<ParametricSymbol> + Send + Sync + 'static,
    {
        self.rules.insert(predecessor, Box::new(rule));
    }

    /// Apply rules for N iterations.
    pub fn iterate(&mut self, n: usize) {
        for _ in 0..n {
            let mut next = Vec::new();
            for sym in &self.current {
                if let Some(rule) = self.rules.get(&sym.symbol) {
                    next.extend(rule(&sym.params));
                } else {
                    next.push(sym.clone());
                }
            }
            self.current = next;
        }
    }

    /// Convert the current state to a simple string (ignoring parameters).
    pub fn to_string_simple(&self) -> String {
        self.current.iter().map(|s| s.symbol).collect()
    }

    /// Reset to the axiom.
    pub fn reset(&mut self) {
        self.current = self.axiom.clone();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_lsystem() {
        let mut system = LSystem::new("A")
            .add_rule('A', "AB")
            .add_rule('B', "A");

        system.iterate(1);
        assert_eq!(system.current_string(), "AB");

        system.iterate(1);
        assert_eq!(system.current_string(), "ABA");

        system.iterate(1);
        assert_eq!(system.current_string(), "ABAAB");
    }

    #[test]
    fn test_lsystem_reset() {
        let mut system = LSystem::new("F")
            .add_rule('F', "F+F");

        system.iterate(3);
        assert!(system.length() > 1);

        system.reset();
        assert_eq!(system.current_string(), "F");
        assert_eq!(system.iterations, 0);
    }

    #[test]
    fn test_koch_snowflake() {
        let (mut system, _) = LSystemPresets::koch_snowflake();
        system.iterate(1);
        assert_eq!(system.current_string(), "F+F--F+F--F+F--F+F--F+F--F+F");
    }

    #[test]
    fn test_turtle_forward() {
        let turtle = TurtleInterpreter::new(1.0, 90.0)
            .with_start(0.0, 0.0)
            .with_angle(0.0);

        let segments = turtle.interpret("FF");
        assert_eq!(segments.len(), 2);

        // First segment: (0,0) -> (1,0).
        assert!((segments[0].start.x).abs() < 0.001);
        assert!((segments[0].end.x - 1.0).abs() < 0.001);

        // Second segment: (1,0) -> (2,0).
        assert!((segments[1].start.x - 1.0).abs() < 0.001);
        assert!((segments[1].end.x - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_turtle_turn() {
        let turtle = TurtleInterpreter::new(1.0, 90.0)
            .with_start(0.0, 0.0)
            .with_angle(0.0);

        let segments = turtle.interpret("F+F");
        assert_eq!(segments.len(), 2);

        // After turning right 90 degrees from angle 0 (east), we should go south.
        let end = segments[1].end;
        assert!((end.x - 1.0).abs() < 0.01);
        assert!((end.y - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_turtle_branching() {
        let turtle = TurtleInterpreter::new(1.0, 45.0)
            .with_start(0.0, 0.0)
            .with_angle(90.0);

        let segments = turtle.interpret("F[+F]F");
        assert_eq!(segments.len(), 3);

        // The branch segment has depth 1, others depth 0.
        assert_eq!(segments[0].depth, 0); // Main trunk.
        assert_eq!(segments[1].depth, 1); // Branch.
        assert_eq!(segments[2].depth, 0); // Continued trunk (after pop).
    }

    #[test]
    fn test_turtle_no_draw() {
        let turtle = TurtleInterpreter::new(1.0, 90.0)
            .with_start(0.0, 0.0)
            .with_angle(0.0);

        let segments = turtle.interpret("fF");
        // Only F draws, f just moves.
        assert_eq!(segments.len(), 1);
        // The drawn segment should start at (1,0) since f moved forward.
        assert!((segments[0].start.x - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bounding_box() {
        let turtle = TurtleInterpreter::new(1.0, 90.0)
            .with_start(0.0, 0.0)
            .with_angle(0.0);

        let segments = turtle.interpret("F+F+F+F");
        let (min, max) = TurtleInterpreter::bounding_box(&segments);

        // Should form a square from (0,-1) to (1,0).
        assert!(min.x >= -0.01);
        assert!(min.y >= -1.01);
        assert!(max.x <= 1.01);
        assert!(max.y <= 0.01);
    }

    #[test]
    fn test_stochastic_rules() {
        let mut system = LSystem::new("F")
            .add_stochastic_rule('F', vec![
                (0.5, "F+F"),
                (0.5, "F-F"),
            ]);
        let mut rng = Rng::new(42);
        system.iterate_stochastic(1, &mut rng);
        let result = system.current_string();
        assert!(result == "F+F" || result == "F-F");
    }

    #[test]
    fn test_3d_turtle() {
        let turtle = TurtleInterpreter3D::new(1.0, 90.0);
        let segments = turtle.interpret("FF");
        assert_eq!(segments.len(), 2);

        // Default heading is Y-up.
        assert!((segments[0].start - Vec3::ZERO).length() < 0.001);
        assert!((segments[0].end.y - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_3d_turtle_pitch() {
        let turtle = TurtleInterpreter3D::new(1.0, 90.0);
        let segments = turtle.interpret("^F");
        assert_eq!(segments.len(), 1);
        // After pitching up 90 degrees from heading=Y, new heading should be Z.
        assert!((segments[0].end.z - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_simple_tree_preset() {
        let (mut system, turtle) = LSystemPresets::simple_tree();
        system.iterate(2);
        let segments = turtle.interpret(system.current_string());
        assert!(!segments.is_empty(), "Tree should produce line segments");

        // Check that there are branches (segments with depth > 0).
        let has_branches = segments.iter().any(|s| s.depth > 0);
        assert!(has_branches, "Tree should have branching structure");
    }

    #[test]
    fn test_dragon_curve_length() {
        let (mut system, _) = LSystemPresets::dragon_curve();
        system.iterate(5);
        // Dragon curve doubles in complexity each iteration.
        assert!(system.length() > 32);
    }

    #[test]
    fn test_parametric_lsystem() {
        let axiom = vec![ParametricSymbol::new('A', vec![1.0])];
        let mut system = ParametricLSystem::new(axiom);

        system.add_rule('A', |params| {
            let s = params.first().copied().unwrap_or(1.0);
            vec![
                ParametricSymbol::new('F', vec![s]),
                ParametricSymbol::new('[', vec![]),
                ParametricSymbol::new('+', vec![]),
                ParametricSymbol::new('A', vec![s * 0.7]),
                ParametricSymbol::new(']', vec![]),
                ParametricSymbol::new('A', vec![s * 0.7]),
            ]
        });

        system.iterate(1);
        let s = system.to_string_simple();
        assert_eq!(s, "F[+A]A");

        system.iterate(1);
        let s2 = system.to_string_simple();
        assert!(s2.contains("[+"), "Should have branches after 2 iterations");
    }

    #[test]
    fn test_turtle_command_parsing() {
        assert_eq!(TurtleCommand::from_char('F'), TurtleCommand::Forward);
        assert_eq!(TurtleCommand::from_char('+'), TurtleCommand::TurnRight);
        assert_eq!(TurtleCommand::from_char('-'), TurtleCommand::TurnLeft);
        assert_eq!(TurtleCommand::from_char('['), TurtleCommand::PushState);
        assert_eq!(TurtleCommand::from_char(']'), TurtleCommand::PopState);
        assert_eq!(TurtleCommand::from_char('^'), TurtleCommand::PitchUp);
        assert_eq!(TurtleCommand::from_char('&'), TurtleCommand::PitchDown);
        assert_eq!(TurtleCommand::from_char('Q'), TurtleCommand::Unknown('Q'));
    }

    #[test]
    fn test_fern_preset() {
        let (mut system, turtle) = LSystemPresets::fern();
        system.iterate(3);
        let segments = turtle.interpret(system.current_string());
        assert!(!segments.is_empty());
        // Ferns should have deep branching.
        let max_depth = segments.iter().map(|s| s.depth).max().unwrap_or(0);
        assert!(max_depth >= 2, "Fern should have branching depth >= 2");
    }

    #[test]
    fn test_3d_bounding_box() {
        let (mut system, turtle) = LSystemPresets::tree_3d();
        system.iterate(2);
        let segments = turtle.interpret(system.current_string());
        let (min, max) = TurtleInterpreter3D::bounding_box(&segments);
        // The tree should extend in the Y direction (upward).
        assert!(max.y > min.y, "3D tree should have vertical extent");
    }
}
