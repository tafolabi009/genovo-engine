//! Knowledge representation and inference system.
//!
//! Provides a facts database, inference rules, truth maintenance, and shared
//! team knowledge for AI agents. Agents can assert facts about the world,
//! define inference rules that derive new facts, and query the knowledge base
//! for specific information.
//!
//! # Key concepts
//!
//! - **Fact**: A triple of (subject, property, value) representing a piece of
//!   world knowledge. For example: ("guard_01", "location", "tower_north").
//! - **InferenceRule**: An if-then rule that derives new facts from existing
//!   ones. When premises match, conclusions are asserted automatically.
//! - **TruthMaintenance**: Tracks which facts were derived from which premises
//!   so that when a premise is retracted, all dependent conclusions are also
//!   retracted (non-monotonic reasoning).
//! - **KnowledgeBase**: Per-agent knowledge store with query support.
//! - **SharedKnowledge**: Team-level knowledge that multiple agents can read
//!   and write to, enabling cooperative AI behaviors.
//! - **KnowledgeQuery**: A structured query against the knowledge base,
//!   supporting pattern matching and variable binding.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of facts a single knowledge base can hold.
pub const MAX_FACTS_PER_KB: usize = 4096;

/// Maximum depth for chained inference (prevents infinite loops).
pub const MAX_INFERENCE_DEPTH: usize = 16;

/// Maximum number of inference rules per knowledge base.
pub const MAX_RULES_PER_KB: usize = 256;

/// Maximum number of pending derivations processed per tick.
pub const MAX_DERIVATIONS_PER_TICK: usize = 512;

/// Maximum number of shared knowledge pools in a game session.
pub const MAX_SHARED_POOLS: usize = 64;

/// Maximum number of subscribers to a single shared knowledge pool.
pub const MAX_SUBSCRIBERS_PER_POOL: usize = 128;

/// Confidence threshold below which a fact is considered unreliable.
pub const MIN_CONFIDENCE_THRESHOLD: f32 = 0.1;

/// Default confidence for directly asserted facts.
pub const DEFAULT_ASSERT_CONFIDENCE: f32 = 1.0;

/// Decay rate for confidence per second when a fact has no supporting evidence.
pub const DEFAULT_CONFIDENCE_DECAY_RATE: f32 = 0.05;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// FactId / RuleId
// ---------------------------------------------------------------------------

/// Unique identifier for a fact within a knowledge base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FactId(pub u64);

impl FactId {
    /// Create a new fact ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl fmt::Display for FactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fact({})", self.0)
    }
}

/// Unique identifier for an inference rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleId(pub u64);

impl RuleId {
    /// Create a new rule ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl fmt::Display for RuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rule({})", self.0)
    }
}

/// Unique identifier for an entity in the knowledge system.
pub type EntityId = u64;

/// Unique identifier for a shared knowledge pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolId(pub u32);

// ---------------------------------------------------------------------------
// FactValue
// ---------------------------------------------------------------------------

/// A value that can be stored in a fact.
#[derive(Debug, Clone, PartialEq)]
pub enum FactValue {
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int(i64),
    /// Floating-point value.
    Float(f32),
    /// String value.
    Text(String),
    /// Entity reference.
    Entity(EntityId),
    /// Position in 3D space.
    Position(f32, f32, f32),
    /// A list of values.
    List(Vec<FactValue>),
    /// No value (used for existence checks — "entity has property").
    Nil,
}

impl FactValue {
    /// Check if this value is truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            Self::Int(i) => *i != 0,
            Self::Float(f) => f.abs() > EPSILON,
            Self::Text(s) => !s.is_empty(),
            Self::Entity(_) => true,
            Self::Position(_, _, _) => true,
            Self::List(l) => !l.is_empty(),
            Self::Nil => false,
        }
    }

    /// Try to extract as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to extract as an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to extract as a float.
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(i) => Some(*i as f32),
            _ => None,
        }
    }

    /// Try to extract as a string reference.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to extract as an entity ID.
    pub fn as_entity(&self) -> Option<EntityId> {
        match self {
            Self::Entity(e) => Some(*e),
            _ => None,
        }
    }

    /// Try to extract as a position tuple.
    pub fn as_position(&self) -> Option<(f32, f32, f32)> {
        match self {
            Self::Position(x, y, z) => Some((*x, *y, *z)),
            _ => None,
        }
    }

    /// Calculate approximate numeric distance between two fact values.
    pub fn numeric_distance(&self, other: &FactValue) -> Option<f32> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => Some((*a - *b).abs() as f32),
            (Self::Float(a), Self::Float(b)) => Some((a - b).abs()),
            (Self::Int(a), Self::Float(b)) | (Self::Float(b), Self::Int(a)) => {
                Some((*a as f32 - b).abs())
            }
            (Self::Position(x1, y1, z1), Self::Position(x2, y2, z2)) => {
                let dx = x1 - x2;
                let dy = y1 - y2;
                let dz = z1 - z2;
                Some((dx * dx + dy * dy + dz * dz).sqrt())
            }
            _ => None,
        }
    }
}

impl fmt::Display for FactValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{}", b),
            Self::Int(i) => write!(f, "{}", i),
            Self::Float(v) => write!(f, "{:.2}", v),
            Self::Text(s) => write!(f, "\"{}\"", s),
            Self::Entity(e) => write!(f, "Entity({})", e),
            Self::Position(x, y, z) => write!(f, "({:.1}, {:.1}, {:.1})", x, y, z),
            Self::List(l) => write!(f, "[{}]", l.len()),
            Self::Nil => write!(f, "nil"),
        }
    }
}

// ---------------------------------------------------------------------------
// Fact
// ---------------------------------------------------------------------------

/// The origin of a fact — how it came to exist in the knowledge base.
#[derive(Debug, Clone, PartialEq)]
pub enum FactOrigin {
    /// Directly asserted by game logic or an agent action.
    Asserted,
    /// Derived by an inference rule.
    Derived {
        /// The rule that produced this fact.
        rule_id: RuleId,
        /// The premise fact IDs that were matched.
        premises: Vec<FactId>,
    },
    /// Received from a shared knowledge pool.
    Shared {
        /// The pool that provided the fact.
        pool_id: PoolId,
        /// The original asserter entity.
        source_entity: EntityId,
    },
    /// Perceived through the perception system.
    Perceived {
        /// Timestamp when the fact was perceived.
        perceived_at: f64,
    },
}

/// A single fact in the knowledge base: (subject, property, value).
#[derive(Debug, Clone)]
pub struct Fact {
    /// Unique identifier for this fact.
    pub id: FactId,
    /// The entity or concept this fact is about.
    pub subject: String,
    /// The property or relation name.
    pub property: String,
    /// The value of the property.
    pub value: FactValue,
    /// Confidence level in this fact (0.0 to 1.0).
    pub confidence: f32,
    /// How this fact was created.
    pub origin: FactOrigin,
    /// Timestamp when the fact was asserted (game time in seconds).
    pub timestamp: f64,
    /// Whether this fact has been invalidated by truth maintenance.
    pub invalidated: bool,
    /// Tags for categorization and filtering.
    pub tags: Vec<String>,
}

impl Fact {
    /// Create a new asserted fact.
    pub fn new_asserted(
        id: FactId,
        subject: impl Into<String>,
        property: impl Into<String>,
        value: FactValue,
        timestamp: f64,
    ) -> Self {
        Self {
            id,
            subject: subject.into(),
            property: property.into(),
            value,
            confidence: DEFAULT_ASSERT_CONFIDENCE,
            origin: FactOrigin::Asserted,
            timestamp,
            invalidated: false,
            tags: Vec::new(),
        }
    }

    /// Create a new derived fact.
    pub fn new_derived(
        id: FactId,
        subject: impl Into<String>,
        property: impl Into<String>,
        value: FactValue,
        rule_id: RuleId,
        premises: Vec<FactId>,
        confidence: f32,
        timestamp: f64,
    ) -> Self {
        Self {
            id,
            subject: subject.into(),
            property: property.into(),
            value,
            confidence,
            origin: FactOrigin::Derived { rule_id, premises },
            timestamp,
            invalidated: false,
            tags: Vec::new(),
        }
    }

    /// Check if this fact matches a given subject and property.
    pub fn matches(&self, subject: &str, property: &str) -> bool {
        !self.invalidated && self.subject == subject && self.property == property
    }

    /// Check if this fact matches with a wildcard pattern.
    /// An empty string acts as a wildcard (matches anything).
    pub fn matches_pattern(&self, subject: &str, property: &str) -> bool {
        if self.invalidated {
            return false;
        }
        let subject_match = subject.is_empty() || self.subject == subject;
        let property_match = property.is_empty() || self.property == property;
        subject_match && property_match
    }

    /// Check if this fact has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Add a tag to this fact.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Return whether confidence is above threshold.
    pub fn is_confident(&self) -> bool {
        self.confidence >= MIN_CONFIDENCE_THRESHOLD
    }
}

// ---------------------------------------------------------------------------
// FactPattern — used in inference rule premises
// ---------------------------------------------------------------------------

/// A pattern for matching facts in inference rule premises.
#[derive(Debug, Clone)]
pub struct FactPattern {
    /// Subject pattern — empty string means wildcard.
    pub subject: String,
    /// Property pattern — empty string means wildcard.
    pub property: String,
    /// Optional value constraint.
    pub value_constraint: Option<ValueConstraint>,
    /// Variable binding name for the subject (for use in conclusions).
    pub bind_subject: Option<String>,
    /// Variable binding name for the value.
    pub bind_value: Option<String>,
}

impl FactPattern {
    /// Create a new pattern matching a specific subject and property.
    pub fn new(subject: impl Into<String>, property: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            property: property.into(),
            value_constraint: None,
            bind_subject: None,
            bind_value: None,
        }
    }

    /// Create a wildcard pattern matching any subject with a given property.
    pub fn any_with_property(property: impl Into<String>) -> Self {
        Self {
            subject: String::new(),
            property: property.into(),
            value_constraint: None,
            bind_subject: None,
            bind_value: None,
        }
    }

    /// Add a value constraint.
    pub fn with_constraint(mut self, constraint: ValueConstraint) -> Self {
        self.value_constraint = Some(constraint);
        self
    }

    /// Bind the matched subject to a variable name.
    pub fn bind_subject_as(mut self, name: impl Into<String>) -> Self {
        self.bind_subject = Some(name.into());
        self
    }

    /// Bind the matched value to a variable name.
    pub fn bind_value_as(mut self, name: impl Into<String>) -> Self {
        self.bind_value = Some(name.into());
        self
    }

    /// Check if a fact matches this pattern.
    pub fn matches_fact(&self, fact: &Fact) -> bool {
        if fact.invalidated {
            return false;
        }
        let subject_ok = self.subject.is_empty() || fact.subject == self.subject;
        let property_ok = self.property.is_empty() || fact.property == self.property;
        let value_ok = self
            .value_constraint
            .as_ref()
            .map_or(true, |c| c.satisfied_by(&fact.value));
        subject_ok && property_ok && value_ok
    }
}

// ---------------------------------------------------------------------------
// ValueConstraint
// ---------------------------------------------------------------------------

/// Constraint on a fact value for pattern matching.
#[derive(Debug, Clone)]
pub enum ValueConstraint {
    /// Value must equal the given value.
    Equals(FactValue),
    /// Value must not equal the given value.
    NotEquals(FactValue),
    /// Numeric value must be greater than threshold.
    GreaterThan(f32),
    /// Numeric value must be less than threshold.
    LessThan(f32),
    /// Numeric value must be within a range (inclusive).
    InRange(f32, f32),
    /// Text value must contain the given substring.
    Contains(String),
    /// Value must be truthy.
    IsTruthy,
    /// Value must be falsy.
    IsFalsy,
    /// Value must be of a specific type.
    IsType(ValueType),
}

/// Type discriminant for value constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Bool,
    Int,
    Float,
    Text,
    Entity,
    Position,
    List,
    Nil,
}

impl ValueConstraint {
    /// Check if a value satisfies this constraint.
    pub fn satisfied_by(&self, value: &FactValue) -> bool {
        match self {
            Self::Equals(expected) => value == expected,
            Self::NotEquals(rejected) => value != rejected,
            Self::GreaterThan(threshold) => {
                value.as_float().map_or(false, |f| f > *threshold)
            }
            Self::LessThan(threshold) => {
                value.as_float().map_or(false, |f| f < *threshold)
            }
            Self::InRange(min, max) => {
                value.as_float().map_or(false, |f| f >= *min && f <= *max)
            }
            Self::Contains(substring) => {
                value.as_text().map_or(false, |s| s.contains(substring.as_str()))
            }
            Self::IsTruthy => value.is_truthy(),
            Self::IsFalsy => !value.is_truthy(),
            Self::IsType(expected_type) => {
                let actual = match value {
                    FactValue::Bool(_) => ValueType::Bool,
                    FactValue::Int(_) => ValueType::Int,
                    FactValue::Float(_) => ValueType::Float,
                    FactValue::Text(_) => ValueType::Text,
                    FactValue::Entity(_) => ValueType::Entity,
                    FactValue::Position(_, _, _) => ValueType::Position,
                    FactValue::List(_) => ValueType::List,
                    FactValue::Nil => ValueType::Nil,
                };
                actual == *expected_type
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ConclusionTemplate — what an inference rule produces
// ---------------------------------------------------------------------------

/// Template for the conclusion of an inference rule.
///
/// Variables (prefixed with `$`) in subject, property, or value fields are
/// replaced with bindings from the matched premises.
#[derive(Debug, Clone)]
pub struct ConclusionTemplate {
    /// Subject template (may contain `$variable` references).
    pub subject: String,
    /// Property template.
    pub property: String,
    /// Value template.
    pub value: ConclusionValue,
    /// Confidence multiplier applied to the derived fact.
    pub confidence_multiplier: f32,
    /// Tags to apply to the derived fact.
    pub tags: Vec<String>,
}

/// Value specification in a conclusion template.
#[derive(Debug, Clone)]
pub enum ConclusionValue {
    /// A fixed literal value.
    Literal(FactValue),
    /// Use the value bound to this variable name.
    BoundVariable(String),
    /// Combine two bound numeric variables with an operation.
    Computed {
        left: String,
        op: ComputeOp,
        right: String,
    },
}

/// Arithmetic operation for computed conclusion values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Min,
    Max,
}

impl ConclusionTemplate {
    /// Create a simple conclusion with a literal value.
    pub fn literal(
        subject: impl Into<String>,
        property: impl Into<String>,
        value: FactValue,
    ) -> Self {
        Self {
            subject: subject.into(),
            property: property.into(),
            value: ConclusionValue::Literal(value),
            confidence_multiplier: 1.0,
            tags: Vec::new(),
        }
    }

    /// Create a conclusion that uses a bound variable as its value.
    pub fn from_binding(
        subject: impl Into<String>,
        property: impl Into<String>,
        variable: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            property: property.into(),
            value: ConclusionValue::BoundVariable(variable.into()),
            confidence_multiplier: 1.0,
            tags: Vec::new(),
        }
    }

    /// Set the confidence multiplier.
    pub fn with_confidence(mut self, multiplier: f32) -> Self {
        self.confidence_multiplier = multiplier.clamp(0.0, 1.0);
        self
    }

    /// Resolve variable references in the subject string.
    fn resolve_subject(&self, bindings: &HashMap<String, FactValue>) -> String {
        resolve_template_string(&self.subject, bindings)
    }

    /// Resolve variable references in the property string.
    fn resolve_property(&self, bindings: &HashMap<String, FactValue>) -> String {
        resolve_template_string(&self.property, bindings)
    }

    /// Resolve the value using the provided bindings.
    fn resolve_value(&self, bindings: &HashMap<String, FactValue>) -> Option<FactValue> {
        match &self.value {
            ConclusionValue::Literal(v) => Some(v.clone()),
            ConclusionValue::BoundVariable(name) => bindings.get(name).cloned(),
            ConclusionValue::Computed { left, op, right } => {
                let lv = bindings.get(left)?.as_float()?;
                let rv = bindings.get(right)?.as_float()?;
                let result = match op {
                    ComputeOp::Add => lv + rv,
                    ComputeOp::Subtract => lv - rv,
                    ComputeOp::Multiply => lv * rv,
                    ComputeOp::Divide => {
                        if rv.abs() < EPSILON {
                            return None;
                        }
                        lv / rv
                    }
                    ComputeOp::Min => lv.min(rv),
                    ComputeOp::Max => lv.max(rv),
                };
                Some(FactValue::Float(result))
            }
        }
    }
}

/// Replace `$var` tokens in a template string with stringified bindings.
fn resolve_template_string(template: &str, bindings: &HashMap<String, FactValue>) -> String {
    let mut result = template.to_string();
    for (name, value) in bindings {
        let token = format!("${}", name);
        if result.contains(&token) {
            let replacement = match value {
                FactValue::Text(s) => s.clone(),
                FactValue::Entity(e) => e.to_string(),
                other => format!("{}", other),
            };
            result = result.replace(&token, &replacement);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// InferenceRule
// ---------------------------------------------------------------------------

/// An inference rule: if all premises match, derive the conclusion(s).
#[derive(Debug, Clone)]
pub struct InferenceRule {
    /// Unique identifier for this rule.
    pub id: RuleId,
    /// Human-readable name for debugging.
    pub name: String,
    /// Premise patterns that must all match for the rule to fire.
    pub premises: Vec<FactPattern>,
    /// Conclusions to derive when the rule fires.
    pub conclusions: Vec<ConclusionTemplate>,
    /// Priority (higher fires first when multiple rules match).
    pub priority: i32,
    /// Whether this rule is currently enabled.
    pub enabled: bool,
    /// Maximum number of times this rule can fire per tick.
    pub max_fires_per_tick: usize,
    /// Number of times this rule has fired in the current tick.
    fires_this_tick: usize,
    /// Total number of times this rule has fired.
    pub total_fires: u64,
}

impl InferenceRule {
    /// Create a new inference rule.
    pub fn new(
        id: RuleId,
        name: impl Into<String>,
        premises: Vec<FactPattern>,
        conclusions: Vec<ConclusionTemplate>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            premises,
            conclusions,
            priority: 0,
            enabled: true,
            max_fires_per_tick: 64,
            fires_this_tick: 0,
            total_fires: 0,
        }
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the max fires per tick.
    pub fn with_max_fires(mut self, max: usize) -> Self {
        self.max_fires_per_tick = max;
        self
    }

    /// Reset the per-tick fire counter.
    pub fn reset_tick(&mut self) {
        self.fires_this_tick = 0;
    }

    /// Check if the rule can still fire this tick.
    pub fn can_fire(&self) -> bool {
        self.enabled && self.fires_this_tick < self.max_fires_per_tick
    }

    /// Record one firing of this rule.
    pub fn record_fire(&mut self) {
        self.fires_this_tick += 1;
        self.total_fires += 1;
    }
}

// ---------------------------------------------------------------------------
// Derivation — internal struct for pending derived facts
// ---------------------------------------------------------------------------

/// A pending derivation to be applied to the knowledge base.
#[derive(Debug)]
struct PendingDerivation {
    rule_id: RuleId,
    subject: String,
    property: String,
    value: FactValue,
    confidence: f32,
    premise_ids: Vec<FactId>,
    tags: Vec<String>,
}

// ---------------------------------------------------------------------------
// DependencyRecord — for truth maintenance
// ---------------------------------------------------------------------------

/// Records which derived fact depends on which premises.
#[derive(Debug, Clone)]
struct DependencyRecord {
    /// The derived fact.
    derived_fact_id: FactId,
    /// The rule that produced it.
    rule_id: RuleId,
    /// The premise facts it depends on.
    premise_fact_ids: Vec<FactId>,
}

// ---------------------------------------------------------------------------
// KnowledgeQuery
// ---------------------------------------------------------------------------

/// A query against the knowledge base.
#[derive(Debug, Clone)]
pub struct KnowledgeQuery {
    /// Subject filter (empty = wildcard).
    pub subject: String,
    /// Property filter (empty = wildcard).
    pub property: String,
    /// Optional value constraint.
    pub value_constraint: Option<ValueConstraint>,
    /// Minimum confidence level.
    pub min_confidence: f32,
    /// Whether to include invalidated facts.
    pub include_invalidated: bool,
    /// Required tags (all must be present).
    pub required_tags: Vec<String>,
    /// Maximum age of the fact in seconds (0 = no limit).
    pub max_age: f64,
}

impl KnowledgeQuery {
    /// Create a query for a specific subject and property.
    pub fn new(subject: impl Into<String>, property: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            property: property.into(),
            value_constraint: None,
            min_confidence: MIN_CONFIDENCE_THRESHOLD,
            include_invalidated: false,
            required_tags: Vec::new(),
            max_age: 0.0,
        }
    }

    /// Create a query matching all facts with a given property.
    pub fn by_property(property: impl Into<String>) -> Self {
        Self::new("", property)
    }

    /// Create a query matching all facts about a given subject.
    pub fn by_subject(subject: impl Into<String>) -> Self {
        Self::new(subject, "")
    }

    /// Add a value constraint.
    pub fn with_constraint(mut self, constraint: ValueConstraint) -> Self {
        self.value_constraint = Some(constraint);
        self
    }

    /// Set minimum confidence.
    pub fn with_min_confidence(mut self, min: f32) -> Self {
        self.min_confidence = min;
        self
    }

    /// Include invalidated facts.
    pub fn include_invalidated(mut self) -> Self {
        self.include_invalidated = true;
        self
    }

    /// Require specific tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.required_tags = tags;
        self
    }

    /// Set maximum age in seconds.
    pub fn with_max_age(mut self, max_age: f64) -> Self {
        self.max_age = max_age;
        self
    }

    /// Check if a fact matches this query.
    pub fn matches(&self, fact: &Fact, current_time: f64) -> bool {
        // Check invalidation
        if !self.include_invalidated && fact.invalidated {
            return false;
        }
        // Check subject
        if !self.subject.is_empty() && fact.subject != self.subject {
            return false;
        }
        // Check property
        if !self.property.is_empty() && fact.property != self.property {
            return false;
        }
        // Check confidence
        if fact.confidence < self.min_confidence {
            return false;
        }
        // Check value constraint
        if let Some(ref constraint) = self.value_constraint {
            if !constraint.satisfied_by(&fact.value) {
                return false;
            }
        }
        // Check tags
        for tag in &self.required_tags {
            if !fact.has_tag(tag) {
                return false;
            }
        }
        // Check age
        if self.max_age > 0.0 && (current_time - fact.timestamp) > self.max_age {
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// QueryResult
// ---------------------------------------------------------------------------

/// Result of a knowledge query.
#[derive(Debug)]
pub struct QueryResult<'a> {
    /// Matching facts, sorted by confidence (descending).
    pub facts: Vec<&'a Fact>,
    /// Total number of facts scanned.
    pub scanned: usize,
}

impl<'a> QueryResult<'a> {
    /// Get the highest-confidence matching fact.
    pub fn best(&self) -> Option<&'a Fact> {
        self.facts.first().copied()
    }

    /// Get the value of the best matching fact.
    pub fn best_value(&self) -> Option<&'a FactValue> {
        self.best().map(|f| &f.value)
    }

    /// Check if there are any matches.
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// Number of matching facts.
    pub fn count(&self) -> usize {
        self.facts.len()
    }
}

// ---------------------------------------------------------------------------
// KnowledgeEvent — events emitted by the knowledge base
// ---------------------------------------------------------------------------

/// Events emitted by the knowledge base during updates.
#[derive(Debug, Clone)]
pub enum KnowledgeEvent {
    /// A new fact was asserted.
    FactAsserted {
        fact_id: FactId,
        subject: String,
        property: String,
    },
    /// A fact was retracted.
    FactRetracted {
        fact_id: FactId,
        subject: String,
        property: String,
    },
    /// A fact was derived by inference.
    FactDerived {
        fact_id: FactId,
        rule_id: RuleId,
        subject: String,
        property: String,
    },
    /// A derived fact was invalidated due to truth maintenance.
    FactInvalidated {
        fact_id: FactId,
        reason: String,
    },
    /// A fact's confidence decayed below threshold.
    ConfidenceDecayed {
        fact_id: FactId,
        old_confidence: f32,
        new_confidence: f32,
    },
    /// An inference rule fired.
    RuleFired {
        rule_id: RuleId,
        derivation_count: usize,
    },
}

// ---------------------------------------------------------------------------
// KnowledgeBase
// ---------------------------------------------------------------------------

/// Per-agent knowledge store with inference and truth maintenance.
pub struct KnowledgeBase {
    /// Owner entity ID.
    pub owner: EntityId,
    /// All facts in the knowledge base.
    facts: Vec<Fact>,
    /// Index: subject -> fact indices.
    subject_index: HashMap<String, Vec<usize>>,
    /// Index: property -> fact indices.
    property_index: HashMap<String, Vec<usize>>,
    /// Inference rules.
    rules: Vec<InferenceRule>,
    /// Dependency records for truth maintenance.
    dependencies: Vec<DependencyRecord>,
    /// Next fact ID counter.
    next_fact_id: u64,
    /// Events generated during the last update.
    pending_events: Vec<KnowledgeEvent>,
    /// Whether the knowledge base is dirty (needs inference re-run).
    dirty: bool,
    /// Current game time.
    current_time: f64,
}

impl KnowledgeBase {
    /// Create a new empty knowledge base for the given entity.
    pub fn new(owner: EntityId) -> Self {
        Self {
            owner,
            facts: Vec::with_capacity(256),
            subject_index: HashMap::new(),
            property_index: HashMap::new(),
            rules: Vec::new(),
            dependencies: Vec::new(),
            next_fact_id: 1,
            pending_events: Vec::new(),
            dirty: false,
            current_time: 0.0,
        }
    }

    /// Get the number of active (non-invalidated) facts.
    pub fn fact_count(&self) -> usize {
        self.facts.iter().filter(|f| !f.invalidated).count()
    }

    /// Get the total number of facts including invalidated.
    pub fn total_fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Get the number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    // -----------------------------------------------------------------------
    // Fact assertion
    // -----------------------------------------------------------------------

    /// Assert a new fact. Returns the fact ID.
    pub fn assert_fact(
        &mut self,
        subject: impl Into<String>,
        property: impl Into<String>,
        value: FactValue,
    ) -> Option<FactId> {
        if self.facts.len() >= MAX_FACTS_PER_KB {
            return None;
        }

        let id = FactId::new(self.next_fact_id);
        self.next_fact_id += 1;

        let subject = subject.into();
        let property = property.into();

        let fact = Fact::new_asserted(id, subject.clone(), property.clone(), value, self.current_time);
        let idx = self.facts.len();
        self.facts.push(fact);

        self.subject_index
            .entry(subject.clone())
            .or_insert_with(Vec::new)
            .push(idx);
        self.property_index
            .entry(property.clone())
            .or_insert_with(Vec::new)
            .push(idx);

        self.pending_events.push(KnowledgeEvent::FactAsserted {
            fact_id: id,
            subject,
            property,
        });

        self.dirty = true;
        Some(id)
    }

    /// Assert a fact with a specific confidence level.
    pub fn assert_fact_with_confidence(
        &mut self,
        subject: impl Into<String>,
        property: impl Into<String>,
        value: FactValue,
        confidence: f32,
    ) -> Option<FactId> {
        let id = self.assert_fact(subject, property, value)?;
        if let Some(fact) = self.get_fact_mut(id) {
            fact.confidence = confidence.clamp(0.0, 1.0);
        }
        Some(id)
    }

    /// Assert a fact with tags.
    pub fn assert_fact_tagged(
        &mut self,
        subject: impl Into<String>,
        property: impl Into<String>,
        value: FactValue,
        tags: Vec<String>,
    ) -> Option<FactId> {
        let id = self.assert_fact(subject, property, value)?;
        if let Some(fact) = self.get_fact_mut(id) {
            fact.tags = tags;
        }
        Some(id)
    }

    /// Retract (remove) a fact by ID. Triggers truth maintenance.
    pub fn retract_fact(&mut self, fact_id: FactId) -> bool {
        let Some(fact) = self.facts.iter_mut().find(|f| f.id == fact_id) else {
            return false;
        };

        if fact.invalidated {
            return false;
        }

        let subject = fact.subject.clone();
        let property = fact.property.clone();
        fact.invalidated = true;

        self.pending_events.push(KnowledgeEvent::FactRetracted {
            fact_id,
            subject,
            property,
        });

        // Truth maintenance: invalidate all facts derived from this one
        self.propagate_retraction(fact_id);
        self.dirty = true;
        true
    }

    /// Retract all facts matching a subject and property.
    pub fn retract_matching(&mut self, subject: &str, property: &str) -> usize {
        let matching_ids: Vec<FactId> = self
            .facts
            .iter()
            .filter(|f| f.matches(subject, property))
            .map(|f| f.id)
            .collect();

        let count = matching_ids.len();
        for id in matching_ids {
            self.retract_fact(id);
        }
        count
    }

    /// Update an existing fact's value. If not found, asserts it.
    pub fn upsert_fact(
        &mut self,
        subject: impl Into<String>,
        property: impl Into<String>,
        value: FactValue,
    ) -> FactId {
        let subject = subject.into();
        let property = property.into();

        // Look for an existing matching fact
        if let Some(existing) = self
            .facts
            .iter_mut()
            .find(|f| f.matches(&subject, &property))
        {
            existing.value = value;
            existing.timestamp = self.current_time;
            existing.confidence = DEFAULT_ASSERT_CONFIDENCE;
            self.dirty = true;
            return existing.id;
        }

        // Not found — assert new
        self.assert_fact(subject, property, value)
            .unwrap_or(FactId(0))
    }

    // -----------------------------------------------------------------------
    // Truth Maintenance
    // -----------------------------------------------------------------------

    /// Propagate a retraction through the dependency graph.
    fn propagate_retraction(&mut self, retracted_id: FactId) {
        let mut queue = VecDeque::new();
        queue.push_back(retracted_id);

        let mut visited = HashSet::new();
        visited.insert(retracted_id);

        while let Some(current) = queue.pop_front() {
            // Find all facts that depend on `current`
            let dependent_ids: Vec<FactId> = self
                .dependencies
                .iter()
                .filter(|d| d.premise_fact_ids.contains(&current))
                .map(|d| d.derived_fact_id)
                .collect();

            for dep_id in dependent_ids {
                if visited.contains(&dep_id) {
                    continue;
                }
                visited.insert(dep_id);

                if let Some(fact) = self.facts.iter_mut().find(|f| f.id == dep_id) {
                    if !fact.invalidated {
                        fact.invalidated = true;
                        self.pending_events.push(KnowledgeEvent::FactInvalidated {
                            fact_id: dep_id,
                            reason: format!("Premise {} retracted", current),
                        });
                        queue.push_back(dep_id);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Fact access
    // -----------------------------------------------------------------------

    /// Get a fact by ID (immutable).
    pub fn get_fact(&self, id: FactId) -> Option<&Fact> {
        self.facts.iter().find(|f| f.id == id && !f.invalidated)
    }

    /// Get a fact by ID (mutable).
    fn get_fact_mut(&mut self, id: FactId) -> Option<&mut Fact> {
        self.facts.iter_mut().find(|f| f.id == id)
    }

    /// Get all active facts.
    pub fn all_facts(&self) -> impl Iterator<Item = &Fact> {
        self.facts.iter().filter(|f| !f.invalidated)
    }

    /// Check if the entity knows a specific fact.
    pub fn knows(&self, subject: &str, property: &str) -> bool {
        self.facts.iter().any(|f| f.matches(subject, property))
    }

    /// Check if the entity knows a fact with a specific value.
    pub fn knows_value(&self, subject: &str, property: &str, value: &FactValue) -> bool {
        self.facts
            .iter()
            .any(|f| f.matches(subject, property) && &f.value == value)
    }

    /// Get the value of a specific fact (highest confidence if multiple match).
    pub fn get_value(&self, subject: &str, property: &str) -> Option<&FactValue> {
        self.facts
            .iter()
            .filter(|f| f.matches(subject, property))
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
            .map(|f| &f.value)
    }

    // -----------------------------------------------------------------------
    // Querying
    // -----------------------------------------------------------------------

    /// Execute a query and return matching facts.
    pub fn query(&self, query: &KnowledgeQuery) -> QueryResult<'_> {
        let mut matching: Vec<&Fact> = self
            .facts
            .iter()
            .filter(|f| query.matches(f, self.current_time))
            .collect();

        let scanned = self.facts.len();

        // Sort by confidence descending
        matching.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        QueryResult {
            facts: matching,
            scanned,
        }
    }

    /// Query for facts about a specific entity.
    pub fn query_entity(&self, subject: &str) -> Vec<&Fact> {
        self.facts
            .iter()
            .filter(|f| !f.invalidated && f.subject == subject)
            .collect()
    }

    /// Query for all entities that have a specific property.
    pub fn query_property(&self, property: &str) -> Vec<&Fact> {
        self.facts
            .iter()
            .filter(|f| !f.invalidated && f.property == property)
            .collect()
    }

    /// Find the most recent fact matching a query.
    pub fn most_recent(&self, subject: &str, property: &str) -> Option<&Fact> {
        self.facts
            .iter()
            .filter(|f| f.matches_pattern(subject, property))
            .max_by(|a, b| {
                a.timestamp
                    .partial_cmp(&b.timestamp)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    // -----------------------------------------------------------------------
    // Inference rules
    // -----------------------------------------------------------------------

    /// Add an inference rule to the knowledge base.
    pub fn add_rule(&mut self, rule: InferenceRule) -> bool {
        if self.rules.len() >= MAX_RULES_PER_KB {
            return false;
        }
        self.rules.push(rule);
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
        self.dirty = true;
        true
    }

    /// Remove an inference rule by ID.
    pub fn remove_rule(&mut self, rule_id: RuleId) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.id != rule_id);
        self.rules.len() < before
    }

    /// Enable or disable a rule.
    pub fn set_rule_enabled(&mut self, rule_id: RuleId, enabled: bool) {
        if let Some(rule) = self.rules.iter_mut().find(|r| r.id == rule_id) {
            rule.enabled = enabled;
            self.dirty = true;
        }
    }

    /// Run forward-chaining inference. Returns the number of new facts derived.
    pub fn run_inference(&mut self) -> usize {
        if !self.dirty {
            return 0;
        }

        // Reset tick counters
        for rule in &mut self.rules {
            rule.reset_tick();
        }

        let mut total_derived = 0;
        let mut depth = 0;

        loop {
            if depth >= MAX_INFERENCE_DEPTH {
                break;
            }

            let derivations = self.collect_derivations();
            if derivations.is_empty() {
                break;
            }

            let batch_count = derivations.len().min(MAX_DERIVATIONS_PER_TICK - total_derived);
            for derivation in derivations.into_iter().take(batch_count) {
                self.apply_derivation(derivation);
                total_derived += 1;
            }

            if total_derived >= MAX_DERIVATIONS_PER_TICK {
                break;
            }
            depth += 1;
        }

        self.dirty = false;
        total_derived
    }

    /// Collect all possible derivations from the current rule set.
    fn collect_derivations(&mut self) -> Vec<PendingDerivation> {
        let mut derivations = Vec::new();

        // We need to iterate rules and match premises against facts.
        // For each rule, try to find bindings that satisfy all premises.
        let rule_count = self.rules.len();
        for rule_idx in 0..rule_count {
            if !self.rules[rule_idx].can_fire() {
                continue;
            }

            let rule = &self.rules[rule_idx];
            let rule_id = rule.id;
            let premises = rule.premises.clone();
            let conclusions = rule.conclusions.clone();

            // Find all binding sets that satisfy the premises
            let binding_sets = self.find_premise_bindings(&premises);

            for (bindings, premise_ids) in binding_sets {
                // Generate conclusions
                for conclusion in &conclusions {
                    let subject = conclusion.resolve_subject(&bindings);
                    let property = conclusion.resolve_property(&bindings);

                    if let Some(value) = conclusion.resolve_value(&bindings) {
                        // Check if this fact already exists
                        let already_exists = self.facts.iter().any(|f| {
                            f.matches(&subject, &property) && f.value == value
                        });

                        if !already_exists {
                            let min_premise_confidence = premise_ids
                                .iter()
                                .filter_map(|pid| self.get_fact(*pid))
                                .map(|f| f.confidence)
                                .fold(f32::MAX, f32::min);

                            let confidence =
                                min_premise_confidence * conclusion.confidence_multiplier;

                            derivations.push(PendingDerivation {
                                rule_id,
                                subject,
                                property,
                                value,
                                confidence,
                                premise_ids: premise_ids.clone(),
                                tags: conclusion.tags.clone(),
                            });
                        }
                    }
                }

                self.rules[rule_idx].record_fire();
                if !self.rules[rule_idx].can_fire() {
                    break;
                }
            }
        }

        derivations
    }

    /// Find all sets of variable bindings that satisfy the given premises.
    fn find_premise_bindings(
        &self,
        premises: &[FactPattern],
    ) -> Vec<(HashMap<String, FactValue>, Vec<FactId>)> {
        if premises.is_empty() {
            return Vec::new();
        }

        // Start with the first premise
        let mut results: Vec<(HashMap<String, FactValue>, Vec<FactId>)> = Vec::new();

        let first = &premises[0];
        for fact in self.facts.iter().filter(|f| first.matches_fact(f)) {
            let mut bindings = HashMap::new();
            let mut ids = vec![fact.id];

            if let Some(ref var) = first.bind_subject {
                bindings.insert(var.clone(), FactValue::Text(fact.subject.clone()));
            }
            if let Some(ref var) = first.bind_value {
                bindings.insert(var.clone(), fact.value.clone());
            }

            // Try to extend with remaining premises
            if self.extend_bindings(&premises[1..], &mut bindings, &mut ids) {
                results.push((bindings, ids));
            }
        }

        results
    }

    /// Recursively extend bindings with remaining premises.
    fn extend_bindings(
        &self,
        remaining: &[FactPattern],
        bindings: &mut HashMap<String, FactValue>,
        ids: &mut Vec<FactId>,
    ) -> bool {
        if remaining.is_empty() {
            return true;
        }

        let pattern = &remaining[0];

        // Resolve the subject pattern using current bindings
        let resolved_subject = resolve_template_string(&pattern.subject, bindings);
        let resolved_pattern = FactPattern {
            subject: resolved_subject,
            property: pattern.property.clone(),
            value_constraint: pattern.value_constraint.clone(),
            bind_subject: pattern.bind_subject.clone(),
            bind_value: pattern.bind_value.clone(),
        };

        for fact in self.facts.iter().filter(|f| resolved_pattern.matches_fact(f)) {
            // Don't reuse the same fact for multiple premises
            if ids.contains(&fact.id) {
                continue;
            }

            let mut new_bindings = bindings.clone();
            let mut new_ids = ids.clone();
            new_ids.push(fact.id);

            if let Some(ref var) = pattern.bind_subject {
                new_bindings.insert(var.clone(), FactValue::Text(fact.subject.clone()));
            }
            if let Some(ref var) = pattern.bind_value {
                new_bindings.insert(var.clone(), fact.value.clone());
            }

            if self.extend_bindings(&remaining[1..], &mut new_bindings, &mut new_ids) {
                *bindings = new_bindings;
                *ids = new_ids;
                return true;
            }
        }

        false
    }

    /// Apply a single derivation, creating the fact and recording the dependency.
    fn apply_derivation(&mut self, derivation: PendingDerivation) {
        let id = FactId::new(self.next_fact_id);
        self.next_fact_id += 1;

        let fact = Fact::new_derived(
            id,
            derivation.subject.clone(),
            derivation.property.clone(),
            derivation.value,
            derivation.rule_id,
            derivation.premise_ids.clone(),
            derivation.confidence,
            self.current_time,
        );

        let idx = self.facts.len();
        let subject = fact.subject.clone();
        let property = fact.property.clone();
        self.facts.push(fact);

        self.subject_index
            .entry(subject.clone())
            .or_insert_with(Vec::new)
            .push(idx);
        self.property_index
            .entry(property.clone())
            .or_insert_with(Vec::new)
            .push(idx);

        self.dependencies.push(DependencyRecord {
            derived_fact_id: id,
            rule_id: derivation.rule_id,
            premise_fact_ids: derivation.premise_ids,
        });

        self.pending_events.push(KnowledgeEvent::FactDerived {
            fact_id: id,
            rule_id: derivation.rule_id,
            subject,
            property,
        });
    }

    // -----------------------------------------------------------------------
    // Confidence decay
    // -----------------------------------------------------------------------

    /// Decay confidence of all facts based on elapsed time.
    pub fn decay_confidence(&mut self, dt: f32) {
        let decay = DEFAULT_CONFIDENCE_DECAY_RATE * dt;

        for fact in &mut self.facts {
            if fact.invalidated {
                continue;
            }
            // Only decay non-asserted facts
            if matches!(fact.origin, FactOrigin::Asserted) {
                continue;
            }

            let old = fact.confidence;
            fact.confidence = (fact.confidence - decay).max(0.0);

            if old >= MIN_CONFIDENCE_THRESHOLD && fact.confidence < MIN_CONFIDENCE_THRESHOLD {
                self.pending_events.push(KnowledgeEvent::ConfidenceDecayed {
                    fact_id: fact.id,
                    old_confidence: old,
                    new_confidence: fact.confidence,
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Update / maintenance
    // -----------------------------------------------------------------------

    /// Update the knowledge base: run inference, decay confidence, clean up.
    pub fn update(&mut self, dt: f32, game_time: f64) {
        self.current_time = game_time;
        self.decay_confidence(dt);
        self.run_inference();
    }

    /// Drain pending events.
    pub fn drain_events(&mut self) -> Vec<KnowledgeEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Compact the knowledge base by removing invalidated facts.
    /// This invalidates all existing FactId references.
    pub fn compact(&mut self) {
        self.facts.retain(|f| !f.invalidated);
        self.rebuild_indices();
        self.dependencies
            .retain(|d| self.facts.iter().any(|f| f.id == d.derived_fact_id));
    }

    /// Rebuild the subject and property indices.
    fn rebuild_indices(&mut self) {
        self.subject_index.clear();
        self.property_index.clear();

        for (idx, fact) in self.facts.iter().enumerate() {
            self.subject_index
                .entry(fact.subject.clone())
                .or_insert_with(Vec::new)
                .push(idx);
            self.property_index
                .entry(fact.property.clone())
                .or_insert_with(Vec::new)
                .push(idx);
        }
    }

    /// Clear all facts and dependencies but keep rules.
    pub fn clear_facts(&mut self) {
        self.facts.clear();
        self.subject_index.clear();
        self.property_index.clear();
        self.dependencies.clear();
        self.dirty = true;
    }

    /// Clear everything.
    pub fn clear_all(&mut self) {
        self.clear_facts();
        self.rules.clear();
    }

    /// Get statistics about the knowledge base.
    pub fn stats(&self) -> KnowledgeBaseStats {
        let active = self.facts.iter().filter(|f| !f.invalidated).count();
        let derived = self
            .facts
            .iter()
            .filter(|f| !f.invalidated && matches!(f.origin, FactOrigin::Derived { .. }))
            .count();
        let asserted = self
            .facts
            .iter()
            .filter(|f| !f.invalidated && matches!(f.origin, FactOrigin::Asserted))
            .count();

        KnowledgeBaseStats {
            total_facts: self.facts.len(),
            active_facts: active,
            derived_facts: derived,
            asserted_facts: asserted,
            invalidated_facts: self.facts.len() - active,
            rule_count: self.rules.len(),
            dependency_count: self.dependencies.len(),
        }
    }
}

/// Statistics about a knowledge base.
#[derive(Debug, Clone)]
pub struct KnowledgeBaseStats {
    pub total_facts: usize,
    pub active_facts: usize,
    pub derived_facts: usize,
    pub asserted_facts: usize,
    pub invalidated_facts: usize,
    pub rule_count: usize,
    pub dependency_count: usize,
}

// ---------------------------------------------------------------------------
// SharedKnowledge
// ---------------------------------------------------------------------------

/// Team-level shared knowledge that multiple agents can access.
///
/// This enables cooperative AI: when one agent learns something, it can share
/// that knowledge with teammates. Each agent maintains its own local KB but
/// can publish/subscribe to shared pools.
pub struct SharedKnowledge {
    /// Pool identifier.
    pub pool_id: PoolId,
    /// Human-readable name (e.g., "team_alpha_knowledge").
    pub name: String,
    /// Shared facts.
    facts: Vec<SharedFact>,
    /// Subscribed entity IDs.
    subscribers: Vec<EntityId>,
    /// Next shared fact ID.
    next_id: u64,
    /// Maximum facts in this pool.
    max_facts: usize,
    /// Pending notifications for subscribers.
    pending_notifications: Vec<SharedKnowledgeNotification>,
}

/// A fact in a shared knowledge pool.
#[derive(Debug, Clone)]
pub struct SharedFact {
    /// Unique ID within the pool.
    pub id: u64,
    /// Who asserted this fact.
    pub source: EntityId,
    /// Subject of the fact.
    pub subject: String,
    /// Property.
    pub property: String,
    /// Value.
    pub value: FactValue,
    /// Confidence from the source agent.
    pub confidence: f32,
    /// Timestamp when shared.
    pub timestamp: f64,
    /// Whether this has been revoked.
    pub revoked: bool,
}

/// Notification sent to subscribers when shared knowledge changes.
#[derive(Debug, Clone)]
pub struct SharedKnowledgeNotification {
    /// Target subscriber.
    pub target: EntityId,
    /// Type of notification.
    pub kind: SharedNotificationKind,
}

/// Kind of shared knowledge notification.
#[derive(Debug, Clone)]
pub enum SharedNotificationKind {
    /// A new fact was shared.
    NewFact {
        source: EntityId,
        subject: String,
        property: String,
        value: FactValue,
        confidence: f32,
    },
    /// A shared fact was revoked.
    FactRevoked {
        source: EntityId,
        subject: String,
        property: String,
    },
    /// A shared fact was updated.
    FactUpdated {
        source: EntityId,
        subject: String,
        property: String,
        new_value: FactValue,
        new_confidence: f32,
    },
}

impl SharedKnowledge {
    /// Create a new shared knowledge pool.
    pub fn new(pool_id: PoolId, name: impl Into<String>) -> Self {
        Self {
            pool_id,
            name: name.into(),
            facts: Vec::with_capacity(128),
            subscribers: Vec::new(),
            next_id: 1,
            max_facts: MAX_FACTS_PER_KB,
            pending_notifications: Vec::new(),
        }
    }

    /// Subscribe an entity to this pool.
    pub fn subscribe(&mut self, entity: EntityId) -> bool {
        if self.subscribers.len() >= MAX_SUBSCRIBERS_PER_POOL {
            return false;
        }
        if !self.subscribers.contains(&entity) {
            self.subscribers.push(entity);
        }
        true
    }

    /// Unsubscribe an entity.
    pub fn unsubscribe(&mut self, entity: EntityId) {
        self.subscribers.retain(|&e| e != entity);
    }

    /// Check if an entity is subscribed.
    pub fn is_subscribed(&self, entity: EntityId) -> bool {
        self.subscribers.contains(&entity)
    }

    /// Get subscriber count.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Share a fact with the pool. Notifies all subscribers except the source.
    pub fn share_fact(
        &mut self,
        source: EntityId,
        subject: impl Into<String>,
        property: impl Into<String>,
        value: FactValue,
        confidence: f32,
        timestamp: f64,
    ) -> Option<u64> {
        if self.facts.len() >= self.max_facts {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;

        let subject = subject.into();
        let property = property.into();

        // Check for existing fact to update
        if let Some(existing) = self
            .facts
            .iter_mut()
            .find(|f| !f.revoked && f.subject == subject && f.property == property)
        {
            existing.value = value.clone();
            existing.confidence = confidence;
            existing.timestamp = timestamp;
            existing.source = source;

            // Notify subscribers of update
            for &sub in &self.subscribers {
                if sub != source {
                    self.pending_notifications.push(SharedKnowledgeNotification {
                        target: sub,
                        kind: SharedNotificationKind::FactUpdated {
                            source,
                            subject: subject.clone(),
                            property: property.clone(),
                            new_value: value.clone(),
                            new_confidence: confidence,
                        },
                    });
                }
            }
            return Some(existing.id);
        }

        let fact = SharedFact {
            id,
            source,
            subject: subject.clone(),
            property: property.clone(),
            value: value.clone(),
            confidence,
            timestamp,
            revoked: false,
        };

        self.facts.push(fact);

        // Notify subscribers
        for &sub in &self.subscribers {
            if sub != source {
                self.pending_notifications.push(SharedKnowledgeNotification {
                    target: sub,
                    kind: SharedNotificationKind::NewFact {
                        source,
                        subject: subject.clone(),
                        property: property.clone(),
                        value: value.clone(),
                        confidence,
                    },
                });
            }
        }

        Some(id)
    }

    /// Revoke a fact from the pool.
    pub fn revoke_fact(&mut self, source: EntityId, subject: &str, property: &str) -> bool {
        let Some(fact) = self
            .facts
            .iter_mut()
            .find(|f| !f.revoked && f.subject == subject && f.property == property)
        else {
            return false;
        };

        fact.revoked = true;

        for &sub in &self.subscribers {
            if sub != source {
                self.pending_notifications.push(SharedKnowledgeNotification {
                    target: sub,
                    kind: SharedNotificationKind::FactRevoked {
                        source,
                        subject: subject.to_string(),
                        property: property.to_string(),
                    },
                });
            }
        }

        true
    }

    /// Query shared facts.
    pub fn query(&self, subject: &str, property: &str) -> Vec<&SharedFact> {
        self.facts
            .iter()
            .filter(|f| {
                !f.revoked
                    && (subject.is_empty() || f.subject == subject)
                    && (property.is_empty() || f.property == property)
            })
            .collect()
    }

    /// Get all active shared facts.
    pub fn all_facts(&self) -> impl Iterator<Item = &SharedFact> {
        self.facts.iter().filter(|f| !f.revoked)
    }

    /// Drain pending notifications.
    pub fn drain_notifications(&mut self) -> Vec<SharedKnowledgeNotification> {
        std::mem::take(&mut self.pending_notifications)
    }

    /// Compact by removing revoked facts.
    pub fn compact(&mut self) {
        self.facts.retain(|f| !f.revoked);
    }

    /// Get the number of active facts.
    pub fn fact_count(&self) -> usize {
        self.facts.iter().filter(|f| !f.revoked).count()
    }
}

// ---------------------------------------------------------------------------
// SharedKnowledgeManager
// ---------------------------------------------------------------------------

/// Manages multiple shared knowledge pools.
pub struct SharedKnowledgeManager {
    /// All pools indexed by pool ID.
    pools: HashMap<PoolId, SharedKnowledge>,
    /// Next pool ID.
    next_pool_id: u32,
}

impl SharedKnowledgeManager {
    /// Create a new manager.
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            next_pool_id: 1,
        }
    }

    /// Create a new shared knowledge pool. Returns the pool ID.
    pub fn create_pool(&mut self, name: impl Into<String>) -> Option<PoolId> {
        if self.pools.len() >= MAX_SHARED_POOLS {
            return None;
        }
        let id = PoolId(self.next_pool_id);
        self.next_pool_id += 1;
        let pool = SharedKnowledge::new(id, name);
        self.pools.insert(id, pool);
        Some(id)
    }

    /// Remove a pool.
    pub fn remove_pool(&mut self, pool_id: PoolId) -> bool {
        self.pools.remove(&pool_id).is_some()
    }

    /// Get a pool by ID.
    pub fn get_pool(&self, pool_id: PoolId) -> Option<&SharedKnowledge> {
        self.pools.get(&pool_id)
    }

    /// Get a mutable pool by ID.
    pub fn get_pool_mut(&mut self, pool_id: PoolId) -> Option<&mut SharedKnowledge> {
        self.pools.get_mut(&pool_id)
    }

    /// Subscribe an entity to a pool.
    pub fn subscribe(&mut self, pool_id: PoolId, entity: EntityId) -> bool {
        self.pools
            .get_mut(&pool_id)
            .map_or(false, |p| p.subscribe(entity))
    }

    /// Unsubscribe an entity from a pool.
    pub fn unsubscribe(&mut self, pool_id: PoolId, entity: EntityId) {
        if let Some(pool) = self.pools.get_mut(&pool_id) {
            pool.unsubscribe(entity);
        }
    }

    /// Unsubscribe an entity from all pools.
    pub fn unsubscribe_all(&mut self, entity: EntityId) {
        for pool in self.pools.values_mut() {
            pool.unsubscribe(entity);
        }
    }

    /// Get all pools an entity is subscribed to.
    pub fn subscribed_pools(&self, entity: EntityId) -> Vec<PoolId> {
        self.pools
            .iter()
            .filter(|(_, p)| p.is_subscribed(entity))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Drain all notifications from all pools.
    pub fn drain_all_notifications(&mut self) -> Vec<(PoolId, SharedKnowledgeNotification)> {
        let mut all = Vec::new();
        for (id, pool) in &mut self.pools {
            for notification in pool.drain_notifications() {
                all.push((*id, notification));
            }
        }
        all
    }

    /// Get pool count.
    pub fn pool_count(&self) -> usize {
        self.pools.len()
    }
}

impl Default for SharedKnowledgeManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// KnowledgeSystem — top-level manager
// ---------------------------------------------------------------------------

/// Top-level knowledge system managing per-agent KBs and shared pools.
pub struct KnowledgeSystem {
    /// Per-agent knowledge bases.
    knowledge_bases: HashMap<EntityId, KnowledgeBase>,
    /// Shared knowledge manager.
    shared: SharedKnowledgeManager,
    /// Global inference rules applied to all agents.
    global_rules: Vec<InferenceRule>,
    /// Next global rule ID.
    next_global_rule_id: u64,
}

impl KnowledgeSystem {
    /// Create a new knowledge system.
    pub fn new() -> Self {
        Self {
            knowledge_bases: HashMap::new(),
            shared: SharedKnowledgeManager::new(),
            global_rules: Vec::new(),
            next_global_rule_id: 10000,
        }
    }

    /// Register an agent, creating their knowledge base.
    pub fn register_agent(&mut self, entity: EntityId) {
        if !self.knowledge_bases.contains_key(&entity) {
            let mut kb = KnowledgeBase::new(entity);
            // Add global rules
            for rule in &self.global_rules {
                kb.add_rule(rule.clone());
            }
            self.knowledge_bases.insert(entity, kb);
        }
    }

    /// Unregister an agent, removing their knowledge base.
    pub fn unregister_agent(&mut self, entity: EntityId) {
        self.knowledge_bases.remove(&entity);
        self.shared.unsubscribe_all(entity);
    }

    /// Get an agent's knowledge base.
    pub fn get_kb(&self, entity: EntityId) -> Option<&KnowledgeBase> {
        self.knowledge_bases.get(&entity)
    }

    /// Get a mutable reference to an agent's knowledge base.
    pub fn get_kb_mut(&mut self, entity: EntityId) -> Option<&mut KnowledgeBase> {
        self.knowledge_bases.get_mut(&entity)
    }

    /// Add a global inference rule that applies to all agents.
    pub fn add_global_rule(&mut self, mut rule: InferenceRule) -> RuleId {
        let id = RuleId::new(self.next_global_rule_id);
        self.next_global_rule_id += 1;
        rule.id = id;

        // Add to all existing KBs
        for kb in self.knowledge_bases.values_mut() {
            kb.add_rule(rule.clone());
        }

        self.global_rules.push(rule);
        id
    }

    /// Get the shared knowledge manager.
    pub fn shared(&self) -> &SharedKnowledgeManager {
        &self.shared
    }

    /// Get a mutable reference to the shared knowledge manager.
    pub fn shared_mut(&mut self) -> &mut SharedKnowledgeManager {
        &mut self.shared
    }

    /// Update all knowledge bases.
    pub fn update(&mut self, dt: f32, game_time: f64) {
        // Process shared notifications first
        let notifications = self.shared.drain_all_notifications();
        for (pool_id, notification) in notifications {
            if let Some(kb) = self.knowledge_bases.get_mut(&notification.target) {
                match notification.kind {
                    SharedNotificationKind::NewFact {
                        source,
                        subject,
                        property,
                        value,
                        confidence,
                    } => {
                        if let Some(id) = kb.assert_fact_with_confidence(
                            &subject,
                            &property,
                            value.clone(),
                            confidence * 0.9, // slight confidence reduction for shared info
                        ) {
                            if let Some(fact) = kb.get_fact_mut(id) {
                                fact.origin = FactOrigin::Shared {
                                    pool_id,
                                    source_entity: source,
                                };
                            }
                        }
                    }
                    SharedNotificationKind::FactRevoked {
                        source: _,
                        subject,
                        property,
                    } => {
                        kb.retract_matching(&subject, &property);
                    }
                    SharedNotificationKind::FactUpdated {
                        source: _,
                        subject,
                        property,
                        new_value,
                        new_confidence,
                    } => {
                        if let Some(fact) = kb
                            .facts
                            .iter_mut()
                            .find(|f| f.matches(&subject, &property))
                        {
                            fact.value = new_value;
                            fact.confidence = new_confidence * 0.9;
                            fact.timestamp = game_time;
                        }
                    }
                }
            }
        }

        // Update each knowledge base
        for kb in self.knowledge_bases.values_mut() {
            kb.update(dt, game_time);
        }
    }

    /// Get the number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.knowledge_bases.len()
    }

    /// Check if an agent knows a fact.
    pub fn agent_knows(&self, entity: EntityId, subject: &str, property: &str) -> bool {
        self.knowledge_bases
            .get(&entity)
            .map_or(false, |kb| kb.knows(subject, property))
    }

    /// Share a fact from an agent's KB to a pool.
    pub fn share_from_agent(
        &mut self,
        entity: EntityId,
        pool_id: PoolId,
        subject: &str,
        property: &str,
        game_time: f64,
    ) -> bool {
        let fact_data = self
            .knowledge_bases
            .get(&entity)
            .and_then(|kb| {
                kb.all_facts()
                    .find(|f| f.subject == subject && f.property == property)
                    .map(|f| (f.value.clone(), f.confidence))
            });

        if let Some((value, confidence)) = fact_data {
            if let Some(pool) = self.shared.get_pool_mut(pool_id) {
                pool.share_fact(entity, subject, property, value, confidence, game_time);
                return true;
            }
        }
        false
    }
}

impl Default for KnowledgeSystem {
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
    fn test_fact_assertion_and_query() {
        let mut kb = KnowledgeBase::new(1);
        let id = kb.assert_fact("guard_01", "location", FactValue::Text("tower".into()));
        assert!(id.is_some());
        assert!(kb.knows("guard_01", "location"));
        assert!(!kb.knows("guard_01", "health"));

        let val = kb.get_value("guard_01", "location");
        assert_eq!(val, Some(&FactValue::Text("tower".into())));
    }

    #[test]
    fn test_fact_retraction_and_truth_maintenance() {
        let mut kb = KnowledgeBase::new(1);

        let premise_id = kb
            .assert_fact("enemy", "visible", FactValue::Bool(true))
            .unwrap();

        // Manually create a derived fact that depends on the premise
        let derived_id = FactId::new(999);
        let derived = Fact::new_derived(
            derived_id,
            "enemy",
            "threat",
            FactValue::Bool(true),
            RuleId(1),
            vec![premise_id],
            1.0,
            0.0,
        );
        kb.facts.push(derived);
        kb.dependencies.push(DependencyRecord {
            derived_fact_id: derived_id,
            rule_id: RuleId(1),
            premise_fact_ids: vec![premise_id],
        });

        // Retract the premise
        assert!(kb.retract_fact(premise_id));

        // The derived fact should also be invalidated
        let derived_fact = kb.facts.iter().find(|f| f.id == derived_id).unwrap();
        assert!(derived_fact.invalidated);
    }

    #[test]
    fn test_knowledge_query_with_constraints() {
        let mut kb = KnowledgeBase::new(1);
        kb.assert_fact("sword", "damage", FactValue::Float(25.0));
        kb.assert_fact("dagger", "damage", FactValue::Float(10.0));
        kb.assert_fact("axe", "damage", FactValue::Float(35.0));

        let query = KnowledgeQuery::by_property("damage")
            .with_constraint(ValueConstraint::GreaterThan(20.0));

        let result = kb.query(&query);
        assert_eq!(result.count(), 2);
    }

    #[test]
    fn test_shared_knowledge() {
        let mut manager = SharedKnowledgeManager::new();
        let pool_id = manager.create_pool("team_alpha").unwrap();

        manager.subscribe(pool_id, 1);
        manager.subscribe(pool_id, 2);

        let pool = manager.get_pool_mut(pool_id).unwrap();
        pool.share_fact(
            1,
            "enemy_base",
            "location",
            FactValue::Position(100.0, 0.0, 200.0),
            1.0,
            0.0,
        );

        let notifications = pool.drain_notifications();
        // Agent 2 should be notified (agent 1 is the source)
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].target, 2);
    }

    #[test]
    fn test_fact_value_operations() {
        let a = FactValue::Float(10.0);
        let b = FactValue::Float(20.0);
        assert_eq!(a.numeric_distance(&b), Some(10.0));

        let pos1 = FactValue::Position(0.0, 0.0, 0.0);
        let pos2 = FactValue::Position(3.0, 4.0, 0.0);
        let dist = pos1.numeric_distance(&pos2).unwrap();
        assert!((dist - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_upsert() {
        let mut kb = KnowledgeBase::new(1);
        let id1 = kb.upsert_fact("npc", "mood", FactValue::Text("happy".into()));
        let id2 = kb.upsert_fact("npc", "mood", FactValue::Text("angry".into()));
        assert_eq!(id1, id2);
        assert_eq!(
            kb.get_value("npc", "mood"),
            Some(&FactValue::Text("angry".into()))
        );
    }
}
