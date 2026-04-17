// engine/ai/src/learning.rs
//
// Simple AI learning for the Genovo engine.
//
// Implements tabular Q-learning for game AI agents:
//
// - **Q-table reinforcement learning** -- State-action value table.
// - **State-action pairs** -- Discrete state and action spaces.
// - **Reward tracking** -- Accumulate and discount rewards.
// - **Exploration/exploitation** -- Epsilon-greedy policy.
// - **Policy serialization** -- Save/load learned policies.
// - **Learning rate decay** -- Reduce learning rate over time.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_LEARNING_RATE: f32 = 0.1;
const DEFAULT_DISCOUNT_FACTOR: f32 = 0.95;
const DEFAULT_EPSILON: f32 = 0.3;
const DEFAULT_EPSILON_DECAY: f32 = 0.999;
const DEFAULT_MIN_EPSILON: f32 = 0.01;

// ---------------------------------------------------------------------------
// State and action
// ---------------------------------------------------------------------------

/// A discrete state identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateId(pub Vec<i32>);

impl StateId {
    /// Create from a single integer.
    pub fn single(id: i32) -> Self {
        Self(vec![id])
    }

    /// Create from multiple features.
    pub fn from_features(features: &[i32]) -> Self {
        Self(features.to_vec())
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.0.len()
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S({:?})", self.0)
    }
}

/// A discrete action identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionId(pub u32);

impl fmt::Display for ActionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "A({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Q-table entry
// ---------------------------------------------------------------------------

/// A single Q-value entry.
#[derive(Debug, Clone)]
struct QEntry {
    value: f32,
    visit_count: u64,
    last_update: u64,
}

impl QEntry {
    fn new() -> Self {
        Self {
            value: 0.0,
            visit_count: 0,
            last_update: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Q-learning configuration
// ---------------------------------------------------------------------------

/// Configuration for Q-learning.
#[derive(Debug, Clone)]
pub struct QLearningConfig {
    /// Learning rate (alpha).
    pub learning_rate: f32,
    /// Discount factor (gamma).
    pub discount_factor: f32,
    /// Exploration rate (epsilon).
    pub epsilon: f32,
    /// Epsilon decay per episode.
    pub epsilon_decay: f32,
    /// Minimum epsilon.
    pub min_epsilon: f32,
    /// Learning rate decay per episode.
    pub lr_decay: f32,
    /// Minimum learning rate.
    pub min_learning_rate: f32,
    /// Available actions.
    pub actions: Vec<ActionId>,
    /// Initial Q-value for unexplored state-action pairs.
    pub initial_q_value: f32,
    /// Whether to use optimistic initialization (higher initial Q values).
    pub optimistic_init: bool,
    /// Optimistic initial value.
    pub optimistic_value: f32,
}

impl Default for QLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            discount_factor: DEFAULT_DISCOUNT_FACTOR,
            epsilon: DEFAULT_EPSILON,
            epsilon_decay: DEFAULT_EPSILON_DECAY,
            min_epsilon: DEFAULT_MIN_EPSILON,
            lr_decay: 0.9999,
            min_learning_rate: 0.001,
            actions: (0..4).map(ActionId).collect(),
            initial_q_value: 0.0,
            optimistic_init: false,
            optimistic_value: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Q-learning statistics
// ---------------------------------------------------------------------------

/// Statistics for the learning process.
#[derive(Debug, Clone, Default)]
pub struct LearningStats {
    /// Total episodes completed.
    pub episodes: u64,
    /// Total steps across all episodes.
    pub total_steps: u64,
    /// Steps in the current episode.
    pub current_episode_steps: u64,
    /// Total reward in the current episode.
    pub current_episode_reward: f32,
    /// Average reward per episode (recent).
    pub avg_reward: f32,
    /// Best episode reward.
    pub best_reward: f32,
    /// Worst episode reward.
    pub worst_reward: f32,
    /// Number of unique states discovered.
    pub unique_states: usize,
    /// Number of Q-table entries.
    pub q_table_size: usize,
    /// Current epsilon.
    pub current_epsilon: f32,
    /// Current learning rate.
    pub current_lr: f32,
    /// Recent rewards for moving average.
    recent_rewards: Vec<f32>,
}

impl LearningStats {
    fn record_episode_reward(&mut self, reward: f32) {
        self.recent_rewards.push(reward);
        if self.recent_rewards.len() > 100 {
            self.recent_rewards.remove(0);
        }
        self.avg_reward = self.recent_rewards.iter().sum::<f32>() / self.recent_rewards.len() as f32;
        if reward > self.best_reward || self.episodes == 1 {
            self.best_reward = reward;
        }
        if reward < self.worst_reward || self.episodes == 1 {
            self.worst_reward = reward;
        }
    }
}

impl fmt::Display for LearningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Episodes: {}, Avg reward: {:.2}, States: {}, epsilon: {:.4}, lr: {:.4}",
            self.episodes, self.avg_reward, self.unique_states, self.current_epsilon, self.current_lr
        )
    }
}

// ---------------------------------------------------------------------------
// Experience tuple
// ---------------------------------------------------------------------------

/// A single experience (s, a, r, s').
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: StateId,
    pub action: ActionId,
    pub reward: f32,
    pub next_state: StateId,
    pub done: bool,
}

// ---------------------------------------------------------------------------
// Q-learner
// ---------------------------------------------------------------------------

/// Simple random number generator for action selection.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.state >> 33) as f32) / (u32::MAX as f32)
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_f32() * max as f32) as usize % max
    }
}

/// Q-learning agent.
pub struct QLearner {
    config: QLearningConfig,
    q_table: HashMap<(StateId, ActionId), QEntry>,
    stats: LearningStats,
    step_count: u64,
    rng: SimpleRng,
    current_epsilon: f32,
    current_lr: f32,
    episode_reward: f32,
    episode_steps: u64,
}

impl QLearner {
    /// Create a new Q-learner.
    pub fn new(config: QLearningConfig) -> Self {
        let epsilon = config.epsilon;
        let lr = config.learning_rate;
        Self {
            config,
            q_table: HashMap::new(),
            stats: LearningStats::default(),
            step_count: 0,
            rng: SimpleRng::new(42),
            current_epsilon: epsilon,
            current_lr: lr,
            episode_reward: 0.0,
            episode_steps: 0,
        }
    }

    /// Get the Q-value for a state-action pair.
    pub fn q_value(&self, state: &StateId, action: ActionId) -> f32 {
        self.q_table
            .get(&(state.clone(), action))
            .map(|e| e.value)
            .unwrap_or_else(|| {
                if self.config.optimistic_init {
                    self.config.optimistic_value
                } else {
                    self.config.initial_q_value
                }
            })
    }

    /// Get the best action for a state (greedy).
    pub fn best_action(&self, state: &StateId) -> ActionId {
        let mut best = self.config.actions[0];
        let mut best_q = f32::NEG_INFINITY;
        for &action in &self.config.actions {
            let q = self.q_value(state, action);
            if q > best_q {
                best_q = q;
                best = action;
            }
        }
        best
    }

    /// Select an action using epsilon-greedy policy.
    pub fn select_action(&mut self, state: &StateId) -> ActionId {
        if self.rng.next_f32() < self.current_epsilon {
            // Explore: random action.
            let idx = self.rng.next_usize(self.config.actions.len());
            self.config.actions[idx]
        } else {
            // Exploit: best action.
            self.best_action(state)
        }
    }

    /// Learn from an experience tuple.
    pub fn learn(&mut self, experience: &Experience) {
        self.step_count += 1;
        self.episode_steps += 1;
        self.episode_reward += experience.reward;

        // Q-learning update: Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        let current_q = self.q_value(&experience.state, experience.action);
        let max_next_q = if experience.done {
            0.0
        } else {
            self.config
                .actions
                .iter()
                .map(|&a| self.q_value(&experience.next_state, a))
                .fold(f32::NEG_INFINITY, f32::max)
        };

        let target = experience.reward + self.config.discount_factor * max_next_q;
        let new_q = current_q + self.current_lr * (target - current_q);

        let key = (experience.state.clone(), experience.action);
        let entry = self.q_table.entry(key).or_insert_with(QEntry::new);
        entry.value = new_q;
        entry.visit_count += 1;
        entry.last_update = self.step_count;

        // End of episode.
        if experience.done {
            self.end_episode();
        }

        self.update_stats();
    }

    /// Batch learn from multiple experiences.
    pub fn learn_batch(&mut self, experiences: &[Experience]) {
        for exp in experiences {
            self.learn(exp);
        }
    }

    /// End the current episode and decay parameters.
    fn end_episode(&mut self) {
        self.stats.episodes += 1;
        self.stats.record_episode_reward(self.episode_reward);

        // Decay epsilon.
        self.current_epsilon = (self.current_epsilon * self.config.epsilon_decay)
            .max(self.config.min_epsilon);

        // Decay learning rate.
        self.current_lr = (self.current_lr * self.config.lr_decay)
            .max(self.config.min_learning_rate);

        self.episode_reward = 0.0;
        self.episode_steps = 0;
    }

    /// Manually end episode (for environments that don't signal "done").
    pub fn force_end_episode(&mut self) {
        self.end_episode();
    }

    /// Update statistics.
    fn update_stats(&mut self) {
        self.stats.total_steps = self.step_count;
        self.stats.current_episode_steps = self.episode_steps;
        self.stats.current_episode_reward = self.episode_reward;
        self.stats.q_table_size = self.q_table.len();
        self.stats.current_epsilon = self.current_epsilon;
        self.stats.current_lr = self.current_lr;

        // Count unique states.
        let mut states = std::collections::HashSet::new();
        for (key, _) in &self.q_table {
            states.insert(key.0.clone());
        }
        self.stats.unique_states = states.len();
    }

    /// Get statistics.
    pub fn stats(&self) -> &LearningStats {
        &self.stats
    }

    /// Get the current epsilon.
    pub fn epsilon(&self) -> f32 {
        self.current_epsilon
    }

    /// Set epsilon manually.
    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.current_epsilon = epsilon.clamp(0.0, 1.0);
    }

    /// Get all Q-values for a state.
    pub fn q_values(&self, state: &StateId) -> Vec<(ActionId, f32)> {
        self.config
            .actions
            .iter()
            .map(|&a| (a, self.q_value(state, a)))
            .collect()
    }

    /// Serialize the Q-table to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.q_table.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.current_epsilon.to_le_bytes());
        bytes.extend_from_slice(&self.current_lr.to_le_bytes());
        bytes.extend_from_slice(&self.stats.episodes.to_le_bytes());
        for ((state, action), entry) in &self.q_table {
            bytes.extend_from_slice(&(state.0.len() as u32).to_le_bytes());
            for &s in &state.0 {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            bytes.extend_from_slice(&action.0.to_le_bytes());
            bytes.extend_from_slice(&entry.value.to_le_bytes());
            bytes.extend_from_slice(&entry.visit_count.to_le_bytes());
        }
        bytes
    }

    /// Clear the Q-table and reset learning.
    pub fn reset(&mut self) {
        self.q_table.clear();
        self.step_count = 0;
        self.current_epsilon = self.config.epsilon;
        self.current_lr = self.config.learning_rate;
        self.episode_reward = 0.0;
        self.episode_steps = 0;
        self.stats = LearningStats::default();
    }

    /// Get the visit count for a state-action pair.
    pub fn visit_count(&self, state: &StateId, action: ActionId) -> u64 {
        self.q_table
            .get(&(state.clone(), action))
            .map(|e| e.visit_count)
            .unwrap_or(0)
    }

    /// Get the total Q-table memory estimate.
    pub fn memory_usage(&self) -> usize {
        self.q_table.len() * (std::mem::size_of::<(StateId, ActionId)>() + std::mem::size_of::<QEntry>())
    }
}

impl Default for QLearner {
    fn default() -> Self {
        Self::new(QLearningConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_learning_basic() {
        let mut learner = QLearner::new(QLearningConfig {
            actions: vec![ActionId(0), ActionId(1)],
            learning_rate: 0.5,
            discount_factor: 0.9,
            epsilon: 0.0, // Pure exploitation for testing.
            ..Default::default()
        });

        // Train: state 0, action 1 is good.
        learner.learn(&Experience {
            state: StateId::single(0),
            action: ActionId(1),
            reward: 10.0,
            next_state: StateId::single(1),
            done: true,
        });

        assert!(learner.q_value(&StateId::single(0), ActionId(1)) > 0.0);
        assert_eq!(learner.best_action(&StateId::single(0)), ActionId(1));
    }

    #[test]
    fn test_epsilon_decay() {
        let mut learner = QLearner::new(QLearningConfig {
            epsilon: 0.5,
            epsilon_decay: 0.9,
            min_epsilon: 0.01,
            ..Default::default()
        });

        let initial = learner.epsilon();
        learner.force_end_episode();
        assert!(learner.epsilon() < initial);
    }

    #[test]
    fn test_serialization() {
        let mut learner = QLearner::default();
        learner.learn(&Experience {
            state: StateId::single(0),
            action: ActionId(0),
            reward: 1.0,
            next_state: StateId::single(1),
            done: false,
        });

        let bytes = learner.serialize();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut learner = QLearner::default();
        for i in 0..10 {
            learner.learn(&Experience {
                state: StateId::single(i),
                action: ActionId(0),
                reward: 1.0,
                next_state: StateId::single(i + 1),
                done: i == 9,
            });
        }

        assert_eq!(learner.stats().episodes, 1);
        assert!(learner.stats().unique_states > 0);
    }
}
