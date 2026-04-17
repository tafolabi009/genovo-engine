// engine/physics/src/collision_filters_v2.rs
//
// Advanced collision filtering for the Genovo engine.
//
// Provides flexible collision filtering beyond simple layer masks:
//
// - **Collision groups** -- Named groups that define sets of bodies.
// - **Collision masks per shape** -- Each shape can have its own collision mask.
// - **Ignore pairs** -- Specific body pairs that skip collision detection.
// - **Temporary ignore** -- Duration-based collision ignoring (e.g., after spawn).
// - **Collision rules engine** -- Programmable rules for complex filtering logic.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_GROUPS: usize = 64;

// ---------------------------------------------------------------------------
// Body / shape identifiers
// ---------------------------------------------------------------------------

/// Body identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FilterBodyId(pub u32);

/// Shape identifier (a body can have multiple shapes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FilterShapeId(pub u32);

// ---------------------------------------------------------------------------
// Collision group
// ---------------------------------------------------------------------------

/// A named collision group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollisionGroupId(pub u32);

/// Collision group definition.
#[derive(Debug, Clone)]
pub struct CollisionGroup {
    /// Unique ID.
    pub id: CollisionGroupId,
    /// Name of the group.
    pub name: String,
    /// Bit index (0..MAX_GROUPS) for bitmask operations.
    pub bit_index: u8,
}

impl fmt::Display for CollisionGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Group({}: {})", self.id.0, self.name)
    }
}

// ---------------------------------------------------------------------------
// Collision mask
// ---------------------------------------------------------------------------

/// A bitmask representing which groups a body/shape belongs to and
/// which groups it collides with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollisionMask {
    /// Which groups this body/shape is a member of.
    pub membership: u64,
    /// Which groups this body/shape collides with.
    pub filter: u64,
}

impl CollisionMask {
    /// Create a mask that collides with everything.
    pub fn all() -> Self {
        Self {
            membership: u64::MAX,
            filter: u64::MAX,
        }
    }

    /// Create a mask with no collisions.
    pub fn none() -> Self {
        Self {
            membership: 0,
            filter: 0,
        }
    }

    /// Create a mask for a single group.
    pub fn group(bit: u8) -> Self {
        let mask = 1u64 << bit;
        Self {
            membership: mask,
            filter: u64::MAX,
        }
    }

    /// Add membership to a group.
    pub fn with_membership(mut self, bit: u8) -> Self {
        self.membership |= 1u64 << bit;
        self
    }

    /// Add filter for a group.
    pub fn with_filter(mut self, bit: u8) -> Self {
        self.filter |= 1u64 << bit;
        self
    }

    /// Remove filter for a group.
    pub fn without_filter(mut self, bit: u8) -> Self {
        self.filter &= !(1u64 << bit);
        self
    }

    /// Check if two masks allow collision.
    pub fn collides_with(&self, other: &CollisionMask) -> bool {
        (self.membership & other.filter) != 0 && (other.membership & self.filter) != 0
    }

    /// Check if this mask is a member of a specific group.
    pub fn is_member_of(&self, bit: u8) -> bool {
        (self.membership & (1u64 << bit)) != 0
    }
}

impl Default for CollisionMask {
    fn default() -> Self {
        Self::all()
    }
}

impl fmt::Display for CollisionMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mask(member:{:#018x}, filter:{:#018x})",
            self.membership, self.filter
        )
    }
}

// ---------------------------------------------------------------------------
// Ignore pair
// ---------------------------------------------------------------------------

/// An ordered pair of body IDs for the ignore set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct IgnorePair(FilterBodyId, FilterBodyId);

impl IgnorePair {
    fn new(a: FilterBodyId, b: FilterBodyId) -> Self {
        if a.0 <= b.0 {
            Self(a, b)
        } else {
            Self(b, a)
        }
    }
}

/// A temporary ignore entry with a duration.
#[derive(Debug, Clone)]
struct TemporaryIgnore {
    pair: IgnorePair,
    remaining_time: f32,
}

// ---------------------------------------------------------------------------
// Collision rule
// ---------------------------------------------------------------------------

/// A collision rule for programmatic filtering.
#[derive(Debug, Clone)]
pub struct CollisionRule {
    /// Rule name.
    pub name: String,
    /// Rule priority (higher priority = checked first).
    pub priority: i32,
    /// The condition that must be met.
    pub condition: RuleCondition,
    /// The action to take if the condition is met.
    pub action: RuleAction,
    /// Whether this rule is enabled.
    pub enabled: bool,
}

/// Condition for a collision rule.
#[derive(Debug, Clone)]
pub enum RuleCondition {
    /// Both bodies are in the specified group.
    BothInGroup(CollisionGroupId),
    /// Either body is in the specified group.
    EitherInGroup(CollisionGroupId),
    /// One body is in group A and the other in group B.
    GroupPair(CollisionGroupId, CollisionGroupId),
    /// Custom predicate identifier (evaluated externally).
    Custom(String),
    /// Always matches.
    Always,
    /// Logical AND of conditions.
    And(Box<RuleCondition>, Box<RuleCondition>),
    /// Logical OR of conditions.
    Or(Box<RuleCondition>, Box<RuleCondition>),
    /// Logical NOT of a condition.
    Not(Box<RuleCondition>),
}

/// Action for a collision rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleAction {
    /// Allow the collision.
    Allow,
    /// Block (ignore) the collision.
    Block,
    /// Allow but mark as sensor (no physics response).
    SensorOnly,
}

// ---------------------------------------------------------------------------
// Per-shape filter data
// ---------------------------------------------------------------------------

/// Per-shape collision filter data.
#[derive(Debug, Clone)]
pub struct ShapeFilter {
    /// The body this shape belongs to.
    pub body: FilterBodyId,
    /// Shape ID.
    pub shape: FilterShapeId,
    /// Collision mask for this shape.
    pub mask: CollisionMask,
    /// Override: if Some, this shape ignores the body-level mask.
    pub override_mask: bool,
}

// ---------------------------------------------------------------------------
// Collision filter system
// ---------------------------------------------------------------------------

/// Statistics for the collision filter system.
#[derive(Debug, Clone, Default)]
pub struct CollisionFilterStats {
    /// Number of defined groups.
    pub group_count: usize,
    /// Number of registered bodies.
    pub body_count: usize,
    /// Number of per-shape overrides.
    pub shape_override_count: usize,
    /// Number of permanent ignore pairs.
    pub permanent_ignore_count: usize,
    /// Number of temporary ignore pairs.
    pub temporary_ignore_count: usize,
    /// Number of collision rules.
    pub rule_count: usize,
    /// Number of filter queries this frame.
    pub queries_this_frame: u64,
    /// Number of blocked collisions this frame.
    pub blocked_this_frame: u64,
}

/// The collision filtering system.
pub struct CollisionFilterSystem {
    /// Named collision groups.
    groups: Vec<CollisionGroup>,
    /// Next group ID.
    next_group_id: u32,
    /// Per-body collision masks.
    body_masks: HashMap<FilterBodyId, CollisionMask>,
    /// Per-shape collision masks (overrides body mask).
    shape_masks: HashMap<FilterShapeId, ShapeFilter>,
    /// Body-to-group mapping.
    body_groups: HashMap<FilterBodyId, HashSet<CollisionGroupId>>,
    /// Permanent ignore pairs.
    ignore_pairs: HashSet<IgnorePair>,
    /// Temporary ignore pairs.
    temp_ignores: Vec<TemporaryIgnore>,
    /// Collision rules.
    rules: Vec<CollisionRule>,
    /// Statistics.
    stats: CollisionFilterStats,
    /// Frame query counter.
    frame_queries: u64,
    /// Frame blocked counter.
    frame_blocked: u64,
}

impl CollisionFilterSystem {
    /// Create a new collision filter system.
    pub fn new() -> Self {
        Self {
            groups: Vec::new(),
            next_group_id: 0,
            body_masks: HashMap::new(),
            shape_masks: HashMap::new(),
            body_groups: HashMap::new(),
            ignore_pairs: HashSet::new(),
            temp_ignores: Vec::new(),
            rules: Vec::new(),
            stats: CollisionFilterStats::default(),
            frame_queries: 0,
            frame_blocked: 0,
        }
    }

    /// Define a new collision group.
    pub fn define_group(&mut self, name: &str) -> CollisionGroupId {
        let id = CollisionGroupId(self.next_group_id);
        let bit_index = self.groups.len() as u8;
        assert!(
            (bit_index as usize) < MAX_GROUPS,
            "Maximum collision groups exceeded"
        );

        self.groups.push(CollisionGroup {
            id,
            name: name.to_string(),
            bit_index,
        });
        self.next_group_id += 1;
        id
    }

    /// Get a group by name.
    pub fn group_by_name(&self, name: &str) -> Option<&CollisionGroup> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// Set the collision mask for a body.
    pub fn set_body_mask(&mut self, body: FilterBodyId, mask: CollisionMask) {
        self.body_masks.insert(body, mask);
    }

    /// Get the collision mask for a body.
    pub fn body_mask(&self, body: FilterBodyId) -> CollisionMask {
        self.body_masks.get(&body).copied().unwrap_or_default()
    }

    /// Set a per-shape collision mask override.
    pub fn set_shape_mask(&mut self, body: FilterBodyId, shape: FilterShapeId, mask: CollisionMask) {
        self.shape_masks.insert(
            shape,
            ShapeFilter {
                body,
                shape,
                mask,
                override_mask: true,
            },
        );
    }

    /// Add a body to a collision group.
    pub fn add_to_group(&mut self, body: FilterBodyId, group: CollisionGroupId) {
        self.body_groups.entry(body).or_default().insert(group);
        // Update the body mask.
        if let Some(grp) = self.groups.iter().find(|g| g.id == group) {
            let mask = self.body_masks.entry(body).or_insert(CollisionMask::default());
            mask.membership |= 1u64 << grp.bit_index;
        }
    }

    /// Remove a body from a collision group.
    pub fn remove_from_group(&mut self, body: FilterBodyId, group: CollisionGroupId) {
        if let Some(groups) = self.body_groups.get_mut(&body) {
            groups.remove(&group);
        }
        if let Some(grp) = self.groups.iter().find(|g| g.id == group) {
            if let Some(mask) = self.body_masks.get_mut(&body) {
                mask.membership &= !(1u64 << grp.bit_index);
            }
        }
    }

    /// Add a permanent ignore pair.
    pub fn ignore_pair(&mut self, a: FilterBodyId, b: FilterBodyId) {
        self.ignore_pairs.insert(IgnorePair::new(a, b));
    }

    /// Remove a permanent ignore pair.
    pub fn unignore_pair(&mut self, a: FilterBodyId, b: FilterBodyId) {
        self.ignore_pairs.remove(&IgnorePair::new(a, b));
    }

    /// Add a temporary ignore pair with a duration.
    pub fn ignore_pair_for(&mut self, a: FilterBodyId, b: FilterBodyId, duration: f32) {
        self.temp_ignores.push(TemporaryIgnore {
            pair: IgnorePair::new(a, b),
            remaining_time: duration,
        });
    }

    /// Add a collision rule.
    pub fn add_rule(&mut self, rule: CollisionRule) {
        self.rules.push(rule);
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Remove a collision rule by name.
    pub fn remove_rule(&mut self, name: &str) {
        self.rules.retain(|r| r.name != name);
    }

    /// Check if two bodies should collide.
    pub fn should_collide(&mut self, a: FilterBodyId, b: FilterBodyId) -> bool {
        self.frame_queries += 1;

        // Check ignore pairs.
        let pair = IgnorePair::new(a, b);
        if self.ignore_pairs.contains(&pair) {
            self.frame_blocked += 1;
            return false;
        }

        // Check temporary ignores.
        for ti in &self.temp_ignores {
            if ti.pair == pair {
                self.frame_blocked += 1;
                return false;
            }
        }

        // Check masks.
        let mask_a = self.body_mask(a);
        let mask_b = self.body_mask(b);
        if !mask_a.collides_with(&mask_b) {
            self.frame_blocked += 1;
            return false;
        }

        // Check rules (highest priority first).
        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }
            if self.evaluate_condition(&rule.condition, a, b) {
                match rule.action {
                    RuleAction::Block => {
                        self.frame_blocked += 1;
                        return false;
                    }
                    RuleAction::Allow => return true,
                    RuleAction::SensorOnly => return true, // Allow but mark as sensor externally.
                }
            }
        }

        true
    }

    /// Evaluate a rule condition.
    fn evaluate_condition(&self, condition: &RuleCondition, a: FilterBodyId, b: FilterBodyId) -> bool {
        match condition {
            RuleCondition::Always => true,
            RuleCondition::BothInGroup(gid) => {
                self.body_in_group(a, *gid) && self.body_in_group(b, *gid)
            }
            RuleCondition::EitherInGroup(gid) => {
                self.body_in_group(a, *gid) || self.body_in_group(b, *gid)
            }
            RuleCondition::GroupPair(g1, g2) => {
                (self.body_in_group(a, *g1) && self.body_in_group(b, *g2))
                    || (self.body_in_group(a, *g2) && self.body_in_group(b, *g1))
            }
            RuleCondition::Custom(_) => false, // External evaluation needed.
            RuleCondition::And(c1, c2) => {
                self.evaluate_condition(c1, a, b) && self.evaluate_condition(c2, a, b)
            }
            RuleCondition::Or(c1, c2) => {
                self.evaluate_condition(c1, a, b) || self.evaluate_condition(c2, a, b)
            }
            RuleCondition::Not(c) => !self.evaluate_condition(c, a, b),
        }
    }

    /// Check if a body is in a group.
    fn body_in_group(&self, body: FilterBodyId, group: CollisionGroupId) -> bool {
        self.body_groups
            .get(&body)
            .map(|groups| groups.contains(&group))
            .unwrap_or(false)
    }

    /// Update the system (tick temporary ignores).
    pub fn update(&mut self, dt: f32) {
        // Decay temporary ignores.
        self.temp_ignores.retain_mut(|ti| {
            ti.remaining_time -= dt;
            ti.remaining_time > 0.0
        });

        // Update statistics.
        self.stats.group_count = self.groups.len();
        self.stats.body_count = self.body_masks.len();
        self.stats.shape_override_count = self.shape_masks.len();
        self.stats.permanent_ignore_count = self.ignore_pairs.len();
        self.stats.temporary_ignore_count = self.temp_ignores.len();
        self.stats.rule_count = self.rules.len();
        self.stats.queries_this_frame = self.frame_queries;
        self.stats.blocked_this_frame = self.frame_blocked;

        // Reset frame counters.
        self.frame_queries = 0;
        self.frame_blocked = 0;
    }

    /// Get statistics.
    pub fn stats(&self) -> &CollisionFilterStats {
        &self.stats
    }

    /// Remove all data for a body.
    pub fn remove_body(&mut self, body: FilterBodyId) {
        self.body_masks.remove(&body);
        self.body_groups.remove(&body);
        self.ignore_pairs.retain(|p| p.0 != body && p.1 != body);
        self.temp_ignores.retain(|ti| ti.pair.0 != body && ti.pair.1 != body);
        self.shape_masks.retain(|_, sf| sf.body != body);
    }
}

impl Default for CollisionFilterSystem {
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
    fn test_mask_collision() {
        let a = CollisionMask::group(0);
        let b = CollisionMask::group(1);
        assert!(a.collides_with(&b)); // Both filter all.
        let c = CollisionMask { membership: 1, filter: 1 }; // Only collides with group 0.
        assert!(a.collides_with(&c));
        assert!(!b.collides_with(&c));
    }

    #[test]
    fn test_ignore_pair() {
        let mut sys = CollisionFilterSystem::new();
        let a = FilterBodyId(0);
        let b = FilterBodyId(1);
        sys.set_body_mask(a, CollisionMask::all());
        sys.set_body_mask(b, CollisionMask::all());

        assert!(sys.should_collide(a, b));
        sys.ignore_pair(a, b);
        assert!(!sys.should_collide(a, b));
    }

    #[test]
    fn test_temporary_ignore() {
        let mut sys = CollisionFilterSystem::new();
        let a = FilterBodyId(0);
        let b = FilterBodyId(1);
        sys.set_body_mask(a, CollisionMask::all());
        sys.set_body_mask(b, CollisionMask::all());

        sys.ignore_pair_for(a, b, 1.0);
        assert!(!sys.should_collide(a, b));

        sys.update(1.5);
        assert!(sys.should_collide(a, b));
    }

    #[test]
    fn test_groups() {
        let mut sys = CollisionFilterSystem::new();
        let player_group = sys.define_group("player");
        let enemy_group = sys.define_group("enemy");

        let player = FilterBodyId(0);
        let enemy = FilterBodyId(1);

        sys.add_to_group(player, player_group);
        sys.add_to_group(enemy, enemy_group);

        // Add rule: same-group bodies don't collide.
        sys.add_rule(CollisionRule {
            name: "no_friendly_fire".to_string(),
            priority: 10,
            condition: RuleCondition::BothInGroup(player_group),
            action: RuleAction::Block,
            enabled: true,
        });

        let player2 = FilterBodyId(2);
        sys.add_to_group(player2, player_group);
        sys.set_body_mask(player, CollisionMask::all());
        sys.set_body_mask(player2, CollisionMask::all());
        sys.set_body_mask(enemy, CollisionMask::all());

        assert!(!sys.should_collide(player, player2)); // Same group, blocked.
        assert!(sys.should_collide(player, enemy)); // Different groups, allowed.
    }
}
