// engine/ai/src/learning.rs
//
// Reinforcement learning for the Genovo engine.
//
// Implements multiple tabular RL algorithms for game AI agents:
//
// - **Q-learning** -- Off-policy TD control with state-action value table.
// - **SARSA** -- On-policy TD control alternative.
// - **Double Q-learning** -- Two Q-tables to reduce overestimation bias.
// - **Experience replay** -- Store transitions, sample random mini-batches.
// - **State discretization** -- Convert continuous features into discrete bins.
// - **Reward shaping** -- Potential-based reward augmentation.
// - **Policy visualization** -- Best action map, value heat map.
// - **Convergence detection** -- Stop training when policy is stable.
// - **Multi-agent coordination** -- Shared experience, independent learners.
// - **Benchmarking** -- Measure learning speed, compare algorithms.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_LEARNING_RATE: f32 = 0.1;
const DEFAULT_DISCOUNT_FACTOR: f32 = 0.95;
const DEFAULT_EPSILON: f32 = 0.3;
const DEFAULT_EPSILON_DECAY: f32 = 0.999;
const DEFAULT_MIN_EPSILON: f32 = 0.01;
const DEFAULT_REPLAY_CAPACITY: usize = 10000;
const DEFAULT_BATCH_SIZE: usize = 32;
const CONVERGENCE_WINDOW: usize = 50;
const CONVERGENCE_THRESHOLD: f32 = 0.001;
const DEFAULT_MAX_EPISODES: u64 = 100_000;
const DEFAULT_MAX_STEPS_PER_EPISODE: u64 = 1000;

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

    fn with_value(value: f32) -> Self {
        Self {
            value,
            visit_count: 0,
            last_update: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Simple RNG
// ---------------------------------------------------------------------------

/// Simple random number generator for action selection and replay sampling.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        let val = self.next_u64();
        ((val >> 33) as f32) / (u32::MAX as f32)
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() % max as u64) as usize
    }

    /// Fisher-Yates partial shuffle to pick `count` unique indices from `0..len`.
    fn sample_indices(&mut self, len: usize, count: usize) -> Vec<usize> {
        let count = count.min(len);
        let mut indices: Vec<usize> = (0..len).collect();
        for i in 0..count {
            let j = i + self.next_usize(len - i);
            indices.swap(i, j);
        }
        indices.truncate(count);
        indices
    }
}

// ---------------------------------------------------------------------------
// State Discretizer
// ---------------------------------------------------------------------------

/// Configuration for discretizing a single continuous feature dimension.
#[derive(Debug, Clone)]
pub struct FeatureBin {
    /// Minimum value for this feature.
    pub min: f32,
    /// Maximum value for this feature.
    pub max: f32,
    /// Number of bins to divide the range into.
    pub num_bins: u32,
}

impl FeatureBin {
    pub fn new(min: f32, max: f32, num_bins: u32) -> Self {
        Self { min, max, num_bins: num_bins.max(1) }
    }

    /// Convert a continuous value to a bin index.
    pub fn discretize(&self, value: f32) -> i32 {
        let clamped = value.clamp(self.min, self.max);
        let range = self.max - self.min;
        if range <= 1e-9 {
            return 0;
        }
        let normalized = (clamped - self.min) / range;
        let bin = (normalized * self.num_bins as f32) as i32;
        bin.min(self.num_bins as i32 - 1)
    }

    /// Get the center value of a bin.
    pub fn bin_center(&self, bin: i32) -> f32 {
        let range = self.max - self.min;
        let bin_width = range / self.num_bins as f32;
        self.min + (bin as f32 + 0.5) * bin_width
    }
}

/// Discretizes a continuous state vector into a discrete StateId using per-dimension binning.
#[derive(Debug, Clone)]
pub struct StateDiscretizer {
    bins: Vec<FeatureBin>,
}

impl StateDiscretizer {
    /// Create from a list of per-dimension bin configs.
    pub fn new(bins: Vec<FeatureBin>) -> Self {
        Self { bins }
    }

    /// Create uniform bins for all dimensions.
    pub fn uniform(dims: usize, min: f32, max: f32, num_bins: u32) -> Self {
        Self {
            bins: (0..dims).map(|_| FeatureBin::new(min, max, num_bins)).collect(),
        }
    }

    /// Convert a continuous state vector to a discrete StateId.
    pub fn discretize(&self, continuous_state: &[f32]) -> StateId {
        let features: Vec<i32> = self.bins.iter()
            .zip(continuous_state.iter())
            .map(|(bin, &value)| bin.discretize(value))
            .collect();
        StateId(features)
    }

    /// Get the total number of possible discrete states.
    pub fn total_states(&self) -> u64 {
        self.bins.iter().map(|b| b.num_bins as u64).product()
    }

    /// Reconstruct a representative continuous state from a discrete one.
    pub fn reconstruct(&self, state: &StateId) -> Vec<f32> {
        self.bins.iter()
            .zip(state.0.iter())
            .map(|(bin, &idx)| bin.bin_center(idx))
            .collect()
    }

    /// Number of feature dimensions.
    pub fn dims(&self) -> usize {
        self.bins.len()
    }
}

// ---------------------------------------------------------------------------
// Reward Shaping
// ---------------------------------------------------------------------------

/// Potential-based reward shaping function.
/// Adds F(s, a, s') = gamma * Phi(s') - Phi(s) to the reward, which is
/// guaranteed to not change the optimal policy (Ng et al., 1999).
#[derive(Debug, Clone)]
pub struct RewardShaper {
    /// Potential function values for states (user-defined heuristic).
    potentials: HashMap<StateId, f32>,
    /// Discount factor gamma (must match the learner's discount factor).
    gamma: f32,
    /// Default potential for unknown states.
    default_potential: f32,
}

impl RewardShaper {
    pub fn new(gamma: f32) -> Self {
        Self {
            potentials: HashMap::new(),
            gamma,
            default_potential: 0.0,
        }
    }

    /// Set potential for a state.
    pub fn set_potential(&mut self, state: StateId, potential: f32) {
        self.potentials.insert(state, potential);
    }

    /// Set a default potential for states not explicitly assigned.
    pub fn set_default_potential(&mut self, potential: f32) {
        self.default_potential = potential;
    }

    /// Get potential for a state.
    pub fn potential(&self, state: &StateId) -> f32 {
        self.potentials.get(state).copied().unwrap_or(self.default_potential)
    }

    /// Compute shaped reward: r + gamma * Phi(s') - Phi(s).
    pub fn shape(&self, reward: f32, state: &StateId, next_state: &StateId, done: bool) -> f32 {
        let phi_s = self.potential(state);
        let phi_s_next = if done { 0.0 } else { self.potential(next_state) };
        reward + self.gamma * phi_s_next - phi_s
    }

    /// Set potentials from a distance-to-goal heuristic. States closer to goal
    /// get higher potential.
    pub fn set_distance_potentials(
        &mut self,
        states: &[StateId],
        distances: &[f32],
        max_distance: f32,
    ) {
        for (state, &dist) in states.iter().zip(distances.iter()) {
            let potential = (max_distance - dist).max(0.0) / max_distance;
            self.potentials.insert(state.clone(), potential);
        }
    }

    /// Number of states with assigned potentials.
    pub fn num_potentials(&self) -> usize {
        self.potentials.len()
    }
}

// ---------------------------------------------------------------------------
// Q-learning configuration
// ---------------------------------------------------------------------------

/// Configuration for Q-learning (also used as base config for SARSA, Double Q).
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
    /// Per-episode step counts for benchmarking.
    episode_lengths: Vec<u64>,
    /// Per-episode reward totals for benchmarking.
    episode_rewards: Vec<f32>,
    /// Whether convergence has been detected.
    pub converged: bool,
    /// Episode at which convergence was detected.
    pub convergence_episode: Option<u64>,
}

impl LearningStats {
    fn record_episode_reward(&mut self, reward: f32, steps: u64) {
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
        self.episode_lengths.push(steps);
        self.episode_rewards.push(reward);
    }

    /// Get the reward variance over the last N episodes.
    pub fn recent_reward_variance(&self, window: usize) -> f32 {
        let n = self.recent_rewards.len().min(window);
        if n < 2 {
            return f32::MAX;
        }
        let start = self.recent_rewards.len() - n;
        let slice = &self.recent_rewards[start..];
        let mean = slice.iter().sum::<f32>() / n as f32;
        let variance = slice.iter().map(|r| (r - mean) * (r - mean)).sum::<f32>() / (n - 1) as f32;
        variance
    }

    /// Get the mean reward over the last N episodes.
    pub fn recent_mean_reward(&self, window: usize) -> f32 {
        let n = self.recent_rewards.len().min(window);
        if n == 0 {
            return 0.0;
        }
        let start = self.recent_rewards.len() - n;
        self.recent_rewards[start..].iter().sum::<f32>() / n as f32
    }

    /// Get episode lengths as a slice (for benchmarking).
    pub fn episode_lengths(&self) -> &[u64] {
        &self.episode_lengths
    }

    /// Get episode rewards as a slice (for benchmarking).
    pub fn episode_rewards(&self) -> &[f32] {
        &self.episode_rewards
    }

    /// Compute the learning speed: average reward improvement per episode
    /// over a window.
    pub fn learning_speed(&self, window: usize) -> f32 {
        let n = self.episode_rewards.len();
        if n < window + 1 {
            return 0.0;
        }
        let recent_mean = self.episode_rewards[n - window..].iter().sum::<f32>() / window as f32;
        let earlier_mean = self.episode_rewards[n - window * 2..n - window]
            .iter()
            .sum::<f32>()
            / window as f32;
        recent_mean - earlier_mean
    }
}

impl fmt::Display for LearningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Episodes: {}, Avg reward: {:.2}, States: {}, epsilon: {:.4}, lr: {:.4}{}",
            self.episodes,
            self.avg_reward,
            self.unique_states,
            self.current_epsilon,
            self.current_lr,
            if self.converged { " [CONVERGED]" } else { "" },
        )
    }
}

// ---------------------------------------------------------------------------
// Experience tuple
// ---------------------------------------------------------------------------

/// A single experience (s, a, r, s', done).
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: StateId,
    pub action: ActionId,
    pub reward: f32,
    pub next_state: StateId,
    pub done: bool,
}

/// A SARSA experience (s, a, r, s', a') where a' is the action actually taken in s'.
#[derive(Debug, Clone)]
pub struct SarsaExperience {
    pub state: StateId,
    pub action: ActionId,
    pub reward: f32,
    pub next_state: StateId,
    pub next_action: ActionId,
    pub done: bool,
}

// ---------------------------------------------------------------------------
// Experience Replay Buffer
// ---------------------------------------------------------------------------

/// Circular buffer storing past experiences for random sampling.
pub struct ExperienceReplayBuffer {
    buffer: Vec<Experience>,
    capacity: usize,
    write_pos: usize,
    len: usize,
    rng: SimpleRng,
}

impl ExperienceReplayBuffer {
    /// Create a new replay buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
            len: 0,
            rng: SimpleRng::new(12345),
        }
    }

    /// Add an experience to the buffer.
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(experience);
        } else {
            self.buffer[self.write_pos] = experience;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.len = self.len.saturating_add(1).min(self.capacity);
    }

    /// Sample a random batch of experiences.
    pub fn sample(&mut self, batch_size: usize) -> Vec<Experience> {
        let actual_size = batch_size.min(self.len);
        if actual_size == 0 {
            return Vec::new();
        }
        let indices = self.rng.sample_indices(self.len, actual_size);
        indices.iter().map(|&i| self.buffer[i].clone()).collect()
    }

    /// Number of experiences currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the buffer is full.
    pub fn is_full(&self) -> bool {
        self.len >= self.capacity
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.write_pos = 0;
        self.len = 0;
    }

    /// Capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Compute the mean reward across all stored experiences.
    pub fn mean_reward(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        let sum: f32 = self.buffer[..self.len].iter().map(|e| e.reward).sum();
        sum / self.len as f32
    }

    /// Compute statistics about the buffer contents.
    pub fn stats(&self) -> ReplayBufferStats {
        if self.len == 0 {
            return ReplayBufferStats::default();
        }
        let rewards: Vec<f32> = self.buffer[..self.len].iter().map(|e| e.reward).collect();
        let min_r = rewards.iter().cloned().fold(f32::MAX, f32::min);
        let max_r = rewards.iter().cloned().fold(f32::MIN, f32::max);
        let mean_r = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let done_count = self.buffer[..self.len].iter().filter(|e| e.done).count();
        let unique_states: HashSet<&StateId> = self.buffer[..self.len].iter().map(|e| &e.state).collect();
        ReplayBufferStats {
            size: self.len,
            capacity: self.capacity,
            min_reward: min_r,
            max_reward: max_r,
            mean_reward: mean_r,
            terminal_count: done_count,
            unique_states: unique_states.len(),
        }
    }
}

/// Statistics about the replay buffer.
#[derive(Debug, Clone, Default)]
pub struct ReplayBufferStats {
    pub size: usize,
    pub capacity: usize,
    pub min_reward: f32,
    pub max_reward: f32,
    pub mean_reward: f32,
    pub terminal_count: usize,
    pub unique_states: usize,
}

// ---------------------------------------------------------------------------
// Convergence Detector
// ---------------------------------------------------------------------------

/// Detects when a learning policy has stabilized.
pub struct ConvergenceDetector {
    /// Rolling window of policy change magnitudes.
    policy_changes: Vec<f32>,
    /// Rolling window of reward changes.
    reward_changes: Vec<f32>,
    /// Window size.
    window_size: usize,
    /// Threshold below which we consider converged.
    threshold: f32,
    /// Snapshot of last policy for comparison.
    last_policy: HashMap<StateId, ActionId>,
    /// Number of consecutive stable windows.
    stable_count: u32,
    /// Required stable windows to declare convergence.
    required_stable: u32,
}

impl ConvergenceDetector {
    pub fn new(window_size: usize, threshold: f32) -> Self {
        Self {
            policy_changes: Vec::new(),
            reward_changes: Vec::new(),
            window_size,
            threshold,
            last_policy: HashMap::new(),
            stable_count: 0,
            required_stable: 3,
        }
    }

    /// Check convergence by comparing current policy to previous snapshot.
    /// Returns true if the policy has stabilized.
    pub fn check(
        &mut self,
        current_policy: &HashMap<StateId, ActionId>,
        episode_reward: f32,
    ) -> bool {
        // Count how many states changed their best action since last snapshot.
        let mut changed = 0usize;
        let mut total = 0usize;
        for (state, &action) in current_policy {
            total += 1;
            match self.last_policy.get(state) {
                Some(&prev_action) if prev_action != action => {
                    changed += 1;
                }
                None => {
                    changed += 1;
                }
                _ => {}
            }
        }

        let change_ratio = if total > 0 {
            changed as f32 / total as f32
        } else {
            1.0
        };

        self.policy_changes.push(change_ratio);
        if self.policy_changes.len() > self.window_size {
            self.policy_changes.remove(0);
        }

        self.reward_changes.push(episode_reward);
        if self.reward_changes.len() > self.window_size {
            self.reward_changes.remove(0);
        }

        self.last_policy = current_policy.clone();

        // Check if policy changes are below threshold on average.
        if self.policy_changes.len() >= self.window_size {
            let avg_change = self.policy_changes.iter().sum::<f32>()
                / self.policy_changes.len() as f32;
            if avg_change < self.threshold {
                self.stable_count += 1;
            } else {
                self.stable_count = 0;
            }
        }

        self.stable_count >= self.required_stable
    }

    /// Get the current average policy change rate.
    pub fn change_rate(&self) -> f32 {
        if self.policy_changes.is_empty() {
            return 1.0;
        }
        self.policy_changes.iter().sum::<f32>() / self.policy_changes.len() as f32
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.policy_changes.clear();
        self.reward_changes.clear();
        self.last_policy.clear();
        self.stable_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Policy Visualization
// ---------------------------------------------------------------------------

/// Data for visualizing a learned policy.
#[derive(Debug, Clone)]
pub struct PolicyVisualization {
    /// Best action per state.
    pub best_actions: HashMap<StateId, ActionId>,
    /// State values V(s) = max_a Q(s,a).
    pub state_values: HashMap<StateId, f32>,
    /// Action-value pairs per state.
    pub q_values: HashMap<StateId, Vec<(ActionId, f32)>>,
    /// Visit count per state.
    pub visit_counts: HashMap<StateId, u64>,
}

impl PolicyVisualization {
    /// Get a 2D grid of state values for visualization (assumes 2D state space).
    /// Returns (values, width, height).
    pub fn as_value_grid(&self, x_range: (i32, i32), z_range: (i32, i32)) -> (Vec<f32>, usize, usize) {
        let width = (x_range.1 - x_range.0 + 1) as usize;
        let height = (z_range.1 - z_range.0 + 1) as usize;
        let mut grid = vec![0.0f32; width * height];
        for z in z_range.0..=z_range.1 {
            for x in x_range.0..=x_range.1 {
                let state = StateId::from_features(&[x, z]);
                let value = self.state_values.get(&state).copied().unwrap_or(0.0);
                let gx = (x - x_range.0) as usize;
                let gz = (z - z_range.0) as usize;
                grid[gz * width + gx] = value;
            }
        }
        (grid, width, height)
    }

    /// Get the action distribution (count of each action chosen as best) across all states.
    pub fn action_distribution(&self) -> HashMap<ActionId, usize> {
        let mut dist = HashMap::new();
        for &action in self.best_actions.values() {
            *dist.entry(action).or_insert(0) += 1;
        }
        dist
    }

    /// Get the number of states with assigned values.
    pub fn num_states(&self) -> usize {
        self.state_values.len()
    }
}

// ---------------------------------------------------------------------------
// Q-learner
// ---------------------------------------------------------------------------

/// Q-learning agent (off-policy TD control).
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
    replay_buffer: Option<ExperienceReplayBuffer>,
    convergence_detector: Option<ConvergenceDetector>,
    reward_shaper: Option<RewardShaper>,
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
            replay_buffer: None,
            convergence_detector: None,
            reward_shaper: None,
        }
    }

    /// Enable experience replay with the given buffer capacity and batch size.
    pub fn enable_replay(&mut self, capacity: usize, batch_size: usize) {
        self.replay_buffer = Some(ExperienceReplayBuffer::new(capacity));
        let _ = batch_size; // stored implicitly; used during learn_with_replay
    }

    /// Enable convergence detection.
    pub fn enable_convergence_detection(&mut self, window: usize, threshold: f32) {
        self.convergence_detector = Some(ConvergenceDetector::new(window, threshold));
    }

    /// Set a reward shaper.
    pub fn set_reward_shaper(&mut self, shaper: RewardShaper) {
        self.reward_shaper = Some(shaper);
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
            let idx = self.rng.next_usize(self.config.actions.len());
            self.config.actions[idx]
        } else {
            self.best_action(state)
        }
    }

    /// Select action using softmax (Boltzmann) policy with temperature.
    pub fn select_action_softmax(&mut self, state: &StateId, temperature: f32) -> ActionId {
        if temperature <= 1e-9 {
            return self.best_action(state);
        }

        let q_values: Vec<f32> = self.config.actions.iter()
            .map(|&a| self.q_value(state, a))
            .collect();

        // Subtract max for numerical stability.
        let max_q = q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_values: Vec<f32> = q_values.iter()
            .map(|&q| ((q - max_q) / temperature).exp())
            .collect();
        let sum: f32 = exp_values.iter().sum();

        let r = self.rng.next_f32() * sum;
        let mut cumulative = 0.0;
        for (i, &exp_v) in exp_values.iter().enumerate() {
            cumulative += exp_v;
            if r <= cumulative {
                return self.config.actions[i];
            }
        }
        *self.config.actions.last().unwrap()
    }

    /// Select action using UCB1 (Upper Confidence Bound).
    pub fn select_action_ucb(&self, state: &StateId, exploration_constant: f32) -> ActionId {
        let total_visits: u64 = self.config.actions.iter()
            .map(|&a| self.visit_count(state, a))
            .sum();
        let ln_total = if total_visits > 0 {
            (total_visits as f32).ln()
        } else {
            1.0
        };

        let mut best = self.config.actions[0];
        let mut best_score = f32::NEG_INFINITY;

        for &action in &self.config.actions {
            let q = self.q_value(state, action);
            let visits = self.visit_count(state, action);
            let ucb_bonus = if visits > 0 {
                exploration_constant * (ln_total / visits as f32).sqrt()
            } else {
                f32::MAX
            };
            let score = q + ucb_bonus;
            if score > best_score {
                best_score = score;
                best = action;
            }
        }
        best
    }

    /// Learn from an experience tuple using standard Q-learning update.
    pub fn learn(&mut self, experience: &Experience) {
        self.step_count += 1;
        self.episode_steps += 1;

        // Apply reward shaping if configured.
        let shaped_reward = if let Some(ref shaper) = self.reward_shaper {
            shaper.shape(experience.reward, &experience.state, &experience.next_state, experience.done)
        } else {
            experience.reward
        };

        self.episode_reward += experience.reward; // Track unshaped reward for stats.

        // Store in replay buffer if enabled.
        if let Some(ref mut buffer) = self.replay_buffer {
            buffer.push(Experience {
                state: experience.state.clone(),
                action: experience.action,
                reward: shaped_reward,
                next_state: experience.next_state.clone(),
                done: experience.done,
            });
        }

        // Q-learning update: Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        self.q_update(&experience.state, experience.action, shaped_reward, &experience.next_state, experience.done);

        // End of episode.
        if experience.done {
            self.end_episode();
        }

        self.update_stats();
    }

    /// Internal Q-value update.
    fn q_update(&mut self, state: &StateId, action: ActionId, reward: f32, next_state: &StateId, done: bool) {
        let current_q = self.q_value(state, action);
        let max_next_q = if done {
            0.0
        } else {
            self.config
                .actions
                .iter()
                .map(|&a| self.q_value(next_state, a))
                .fold(f32::NEG_INFINITY, f32::max)
        };

        let target = reward + self.config.discount_factor * max_next_q;
        let new_q = current_q + self.current_lr * (target - current_q);

        let key = (state.clone(), action);
        let entry = self.q_table.entry(key).or_insert_with(QEntry::new);
        entry.value = new_q;
        entry.visit_count += 1;
        entry.last_update = self.step_count;
    }

    /// Learn from a batch sampled from the replay buffer.
    pub fn learn_from_replay(&mut self, batch_size: usize) {
        if let Some(ref mut buffer) = self.replay_buffer {
            let batch = buffer.sample(batch_size);
            for exp in &batch {
                let current_q = self.q_value(&exp.state, exp.action);
                let max_next_q = if exp.done {
                    0.0
                } else {
                    self.config.actions.iter()
                        .map(|&a| self.q_value(&exp.next_state, a))
                        .fold(f32::NEG_INFINITY, f32::max)
                };
                let target = exp.reward + self.config.discount_factor * max_next_q;
                let new_q = current_q + self.current_lr * (target - current_q);
                let key = (exp.state.clone(), exp.action);
                let entry = self.q_table.entry(key).or_insert_with(QEntry::new);
                entry.value = new_q;
                entry.visit_count += 1;
                entry.last_update = self.step_count;
            }
        }
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
        self.stats.record_episode_reward(self.episode_reward, self.episode_steps);

        // Check convergence if enabled.
        if let Some(ref mut detector) = self.convergence_detector {
            let policy = self.extract_policy();
            let converged = detector.check(&policy, self.episode_reward);
            if converged && !self.stats.converged {
                self.stats.converged = true;
                self.stats.convergence_episode = Some(self.stats.episodes);
            }
        }

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

        let mut states = HashSet::new();
        for (key, _) in &self.q_table {
            states.insert(key.0.clone());
        }
        self.stats.unique_states = states.len();
    }

    /// Extract the current policy (best action for every visited state).
    pub fn extract_policy(&self) -> HashMap<StateId, ActionId> {
        let mut states = HashSet::new();
        for (key, _) in &self.q_table {
            states.insert(key.0.clone());
        }
        let mut policy = HashMap::new();
        for state in states {
            let best = self.best_action(&state);
            policy.insert(state, best);
        }
        policy
    }

    /// Generate a full policy visualization.
    pub fn visualize_policy(&self) -> PolicyVisualization {
        let mut states = HashSet::new();
        for (key, _) in &self.q_table {
            states.insert(key.0.clone());
        }

        let mut best_actions = HashMap::new();
        let mut state_values = HashMap::new();
        let mut q_values_map = HashMap::new();
        let mut visit_counts = HashMap::new();

        for state in states {
            let qvs: Vec<(ActionId, f32)> = self.config.actions.iter()
                .map(|&a| (a, self.q_value(&state, a)))
                .collect();

            let best = qvs.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|&(a, _)| a)
                .unwrap_or(self.config.actions[0]);

            let value = qvs.iter()
                .map(|&(_, v)| v)
                .fold(f32::NEG_INFINITY, f32::max);

            let total_visits: u64 = self.config.actions.iter()
                .map(|&a| self.visit_count(&state, a))
                .sum();

            best_actions.insert(state.clone(), best);
            state_values.insert(state.clone(), value);
            q_values_map.insert(state.clone(), qvs);
            visit_counts.insert(state, total_visits);
        }

        PolicyVisualization {
            best_actions,
            state_values,
            q_values: q_values_map,
            visit_counts,
        }
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

    /// Deserialize from bytes, restoring Q-table and parameters.
    pub fn deserialize(&mut self, bytes: &[u8]) -> Result<(), &'static str> {
        if bytes.len() < 20 {
            return Err("Buffer too small");
        }
        let mut offset = 0;

        let table_len = u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().map_err(|_| "parse error")?
        ) as usize;
        offset += 4;

        self.current_epsilon = f32::from_le_bytes(
            bytes[offset..offset + 4].try_into().map_err(|_| "parse error")?
        );
        offset += 4;

        self.current_lr = f32::from_le_bytes(
            bytes[offset..offset + 4].try_into().map_err(|_| "parse error")?
        );
        offset += 4;

        self.stats.episodes = u64::from_le_bytes(
            bytes[offset..offset + 8].try_into().map_err(|_| "parse error")?
        );
        offset += 8;

        self.q_table.clear();
        for _ in 0..table_len {
            if offset + 4 > bytes.len() {
                return Err("Unexpected end of data");
            }
            let state_len = u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().map_err(|_| "parse error")?
            ) as usize;
            offset += 4;

            let mut features = Vec::with_capacity(state_len);
            for _ in 0..state_len {
                if offset + 4 > bytes.len() {
                    return Err("Unexpected end of data");
                }
                features.push(i32::from_le_bytes(
                    bytes[offset..offset + 4].try_into().map_err(|_| "parse error")?
                ));
                offset += 4;
            }

            if offset + 4 > bytes.len() {
                return Err("Unexpected end of data");
            }
            let action = ActionId(u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().map_err(|_| "parse error")?
            ));
            offset += 4;

            if offset + 4 > bytes.len() {
                return Err("Unexpected end of data");
            }
            let value = f32::from_le_bytes(
                bytes[offset..offset + 4].try_into().map_err(|_| "parse error")?
            );
            offset += 4;

            if offset + 8 > bytes.len() {
                return Err("Unexpected end of data");
            }
            let visit_count = u64::from_le_bytes(
                bytes[offset..offset + 8].try_into().map_err(|_| "parse error")?
            );
            offset += 8;

            let state = StateId(features);
            self.q_table.insert(
                (state, action),
                QEntry { value, visit_count, last_update: 0 },
            );
        }

        Ok(())
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
        if let Some(ref mut buffer) = self.replay_buffer {
            buffer.clear();
        }
        if let Some(ref mut detector) = self.convergence_detector {
            detector.reset();
        }
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

    /// Whether convergence has been detected.
    pub fn has_converged(&self) -> bool {
        self.stats.converged
    }

    /// Get the replay buffer stats (if replay is enabled).
    pub fn replay_stats(&self) -> Option<ReplayBufferStats> {
        self.replay_buffer.as_ref().map(|b| b.stats())
    }
}

impl Default for QLearner {
    fn default() -> Self {
        Self::new(QLearningConfig::default())
    }
}

// ---------------------------------------------------------------------------
// SARSA Learner
// ---------------------------------------------------------------------------

/// SARSA agent (on-policy TD control).
/// Unlike Q-learning, SARSA uses the action actually taken in the next state
/// for its update, making it more conservative (safer) in stochastic environments.
pub struct SarsaLearner {
    config: QLearningConfig,
    q_table: HashMap<(StateId, ActionId), QEntry>,
    stats: LearningStats,
    step_count: u64,
    rng: SimpleRng,
    current_epsilon: f32,
    current_lr: f32,
    episode_reward: f32,
    episode_steps: u64,
    /// The action selected for the next step (needed for on-policy update).
    pending_action: Option<ActionId>,
    /// Expected SARSA flag: if true, uses expected value over policy instead of
    /// sampled next action (reduces variance).
    pub expected_sarsa: bool,
}

impl SarsaLearner {
    pub fn new(config: QLearningConfig) -> Self {
        let epsilon = config.epsilon;
        let lr = config.learning_rate;
        Self {
            config,
            q_table: HashMap::new(),
            stats: LearningStats::default(),
            step_count: 0,
            rng: SimpleRng::new(7777),
            current_epsilon: epsilon,
            current_lr: lr,
            episode_reward: 0.0,
            episode_steps: 0,
            pending_action: None,
            expected_sarsa: false,
        }
    }

    /// Create an Expected SARSA learner.
    pub fn expected(config: QLearningConfig) -> Self {
        let mut learner = Self::new(config);
        learner.expected_sarsa = true;
        learner
    }

    /// Get Q-value.
    pub fn q_value(&self, state: &StateId, action: ActionId) -> f32 {
        self.q_table
            .get(&(state.clone(), action))
            .map(|e| e.value)
            .unwrap_or(self.config.initial_q_value)
    }

    /// Best action (greedy).
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

    /// Select action (epsilon-greedy).
    pub fn select_action(&mut self, state: &StateId) -> ActionId {
        if self.rng.next_f32() < self.current_epsilon {
            let idx = self.rng.next_usize(self.config.actions.len());
            self.config.actions[idx]
        } else {
            self.best_action(state)
        }
    }

    /// Compute the expected value under the epsilon-greedy policy.
    fn expected_value(&self, state: &StateId) -> f32 {
        let n_actions = self.config.actions.len() as f32;
        let greedy = self.best_action(state);
        let greedy_q = self.q_value(state, greedy);

        // Expected value = (1 - eps) * Q(s, greedy) + eps/|A| * sum_a Q(s,a)
        let sum_q: f32 = self.config.actions.iter()
            .map(|&a| self.q_value(state, a))
            .sum();

        (1.0 - self.current_epsilon) * greedy_q
            + (self.current_epsilon / n_actions) * sum_q
    }

    /// Learn from a SARSA experience (s, a, r, s', a').
    pub fn learn_sarsa(&mut self, exp: &SarsaExperience) {
        self.step_count += 1;
        self.episode_steps += 1;
        self.episode_reward += exp.reward;

        let current_q = self.q_value(&exp.state, exp.action);

        let next_q = if exp.done {
            0.0
        } else if self.expected_sarsa {
            self.expected_value(&exp.next_state)
        } else {
            self.q_value(&exp.next_state, exp.next_action)
        };

        let target = exp.reward + self.config.discount_factor * next_q;
        let new_q = current_q + self.current_lr * (target - current_q);

        let key = (exp.state.clone(), exp.action);
        let entry = self.q_table.entry(key).or_insert_with(QEntry::new);
        entry.value = new_q;
        entry.visit_count += 1;
        entry.last_update = self.step_count;

        if exp.done {
            self.end_episode();
        }
    }

    /// Convenience method: learn from a standard experience by selecting next action internally.
    pub fn learn(&mut self, experience: &Experience) {
        let next_action = if experience.done {
            self.config.actions[0] // doesn't matter, won't be used
        } else {
            self.pending_action.unwrap_or_else(|| self.select_action(&experience.next_state))
        };

        self.learn_sarsa(&SarsaExperience {
            state: experience.state.clone(),
            action: experience.action,
            reward: experience.reward,
            next_state: experience.next_state.clone(),
            next_action,
            done: experience.done,
        });

        // Pre-select the action for the next state (on-policy).
        if !experience.done {
            self.pending_action = Some(self.select_action(&experience.next_state));
        } else {
            self.pending_action = None;
        }
    }

    /// Get the pending action for the current state (if any).
    pub fn pending_action(&self) -> Option<ActionId> {
        self.pending_action
    }

    fn end_episode(&mut self) {
        self.stats.episodes += 1;
        self.stats.record_episode_reward(self.episode_reward, self.episode_steps);
        self.current_epsilon = (self.current_epsilon * self.config.epsilon_decay)
            .max(self.config.min_epsilon);
        self.current_lr = (self.current_lr * self.config.lr_decay)
            .max(self.config.min_learning_rate);
        self.episode_reward = 0.0;
        self.episode_steps = 0;
        self.pending_action = None;
    }

    pub fn force_end_episode(&mut self) {
        self.end_episode();
    }

    pub fn stats(&self) -> &LearningStats {
        &self.stats
    }

    pub fn epsilon(&self) -> f32 {
        self.current_epsilon
    }

    pub fn extract_policy(&self) -> HashMap<StateId, ActionId> {
        let mut states = HashSet::new();
        for (key, _) in &self.q_table {
            states.insert(key.0.clone());
        }
        let mut policy = HashMap::new();
        for state in states {
            policy.insert(state.clone(), self.best_action(&state));
        }
        policy
    }

    pub fn reset(&mut self) {
        self.q_table.clear();
        self.step_count = 0;
        self.current_epsilon = self.config.epsilon;
        self.current_lr = self.config.learning_rate;
        self.episode_reward = 0.0;
        self.episode_steps = 0;
        self.stats = LearningStats::default();
        self.pending_action = None;
    }

    pub fn q_values(&self, state: &StateId) -> Vec<(ActionId, f32)> {
        self.config.actions.iter().map(|&a| (a, self.q_value(state, a))).collect()
    }
}

// ---------------------------------------------------------------------------
// Double Q-learner
// ---------------------------------------------------------------------------

/// Double Q-learning agent.
/// Uses two independent Q-tables to decouple action selection from evaluation,
/// reducing the maximization bias present in standard Q-learning.
pub struct DoubleQLearner {
    config: QLearningConfig,
    q_table_a: HashMap<(StateId, ActionId), QEntry>,
    q_table_b: HashMap<(StateId, ActionId), QEntry>,
    stats: LearningStats,
    step_count: u64,
    rng: SimpleRng,
    current_epsilon: f32,
    current_lr: f32,
    episode_reward: f32,
    episode_steps: u64,
}

impl DoubleQLearner {
    pub fn new(config: QLearningConfig) -> Self {
        let epsilon = config.epsilon;
        let lr = config.learning_rate;
        Self {
            config,
            q_table_a: HashMap::new(),
            q_table_b: HashMap::new(),
            stats: LearningStats::default(),
            step_count: 0,
            rng: SimpleRng::new(31415),
            current_epsilon: epsilon,
            current_lr: lr,
            episode_reward: 0.0,
            episode_steps: 0,
        }
    }

    /// Get Q-value from table A.
    fn q_a(&self, state: &StateId, action: ActionId) -> f32 {
        self.q_table_a
            .get(&(state.clone(), action))
            .map(|e| e.value)
            .unwrap_or(self.config.initial_q_value)
    }

    /// Get Q-value from table B.
    fn q_b(&self, state: &StateId, action: ActionId) -> f32 {
        self.q_table_b
            .get(&(state.clone(), action))
            .map(|e| e.value)
            .unwrap_or(self.config.initial_q_value)
    }

    /// Get the combined Q-value (average of both tables).
    pub fn q_value(&self, state: &StateId, action: ActionId) -> f32 {
        (self.q_a(state, action) + self.q_b(state, action)) * 0.5
    }

    /// Best action using combined Q-values.
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

    /// Best action using only table A.
    fn best_action_a(&self, state: &StateId) -> ActionId {
        let mut best = self.config.actions[0];
        let mut best_q = f32::NEG_INFINITY;
        for &action in &self.config.actions {
            let q = self.q_a(state, action);
            if q > best_q {
                best_q = q;
                best = action;
            }
        }
        best
    }

    /// Best action using only table B.
    fn best_action_b(&self, state: &StateId) -> ActionId {
        let mut best = self.config.actions[0];
        let mut best_q = f32::NEG_INFINITY;
        for &action in &self.config.actions {
            let q = self.q_b(state, action);
            if q > best_q {
                best_q = q;
                best = action;
            }
        }
        best
    }

    /// Select action using epsilon-greedy on combined Q-values.
    pub fn select_action(&mut self, state: &StateId) -> ActionId {
        if self.rng.next_f32() < self.current_epsilon {
            let idx = self.rng.next_usize(self.config.actions.len());
            self.config.actions[idx]
        } else {
            self.best_action(state)
        }
    }

    /// Learn from an experience using the Double Q-learning update.
    /// With 50% probability, update table A using table B for evaluation, or vice versa.
    pub fn learn(&mut self, experience: &Experience) {
        self.step_count += 1;
        self.episode_steps += 1;
        self.episode_reward += experience.reward;

        let update_a = self.rng.next_f32() < 0.5;

        if update_a {
            // Update A: use A to select best action in s', use B to evaluate it.
            let current_q = self.q_a(&experience.state, experience.action);
            let next_q = if experience.done {
                0.0
            } else {
                let best_a = self.best_action_a(&experience.next_state);
                self.q_b(&experience.next_state, best_a)
            };
            let target = experience.reward + self.config.discount_factor * next_q;
            let new_q = current_q + self.current_lr * (target - current_q);

            let key = (experience.state.clone(), experience.action);
            let entry = self.q_table_a.entry(key).or_insert_with(QEntry::new);
            entry.value = new_q;
            entry.visit_count += 1;
            entry.last_update = self.step_count;
        } else {
            // Update B: use B to select best action in s', use A to evaluate it.
            let current_q = self.q_b(&experience.state, experience.action);
            let next_q = if experience.done {
                0.0
            } else {
                let best_b = self.best_action_b(&experience.next_state);
                self.q_a(&experience.next_state, best_b)
            };
            let target = experience.reward + self.config.discount_factor * next_q;
            let new_q = current_q + self.current_lr * (target - current_q);

            let key = (experience.state.clone(), experience.action);
            let entry = self.q_table_b.entry(key).or_insert_with(QEntry::new);
            entry.value = new_q;
            entry.visit_count += 1;
            entry.last_update = self.step_count;
        }

        if experience.done {
            self.end_episode();
        }
    }

    fn end_episode(&mut self) {
        self.stats.episodes += 1;
        self.stats.record_episode_reward(self.episode_reward, self.episode_steps);
        self.current_epsilon = (self.current_epsilon * self.config.epsilon_decay)
            .max(self.config.min_epsilon);
        self.current_lr = (self.current_lr * self.config.lr_decay)
            .max(self.config.min_learning_rate);
        self.episode_reward = 0.0;
        self.episode_steps = 0;
    }

    pub fn force_end_episode(&mut self) {
        self.end_episode();
    }

    pub fn stats(&self) -> &LearningStats {
        &self.stats
    }

    pub fn epsilon(&self) -> f32 {
        self.current_epsilon
    }

    pub fn extract_policy(&self) -> HashMap<StateId, ActionId> {
        let mut states = HashSet::new();
        for (key, _) in &self.q_table_a {
            states.insert(key.0.clone());
        }
        for (key, _) in &self.q_table_b {
            states.insert(key.0.clone());
        }
        let mut policy = HashMap::new();
        for state in states {
            policy.insert(state.clone(), self.best_action(&state));
        }
        policy
    }

    pub fn reset(&mut self) {
        self.q_table_a.clear();
        self.q_table_b.clear();
        self.step_count = 0;
        self.current_epsilon = self.config.epsilon;
        self.current_lr = self.config.learning_rate;
        self.episode_reward = 0.0;
        self.episode_steps = 0;
        self.stats = LearningStats::default();
    }

    pub fn q_values(&self, state: &StateId) -> Vec<(ActionId, f32)> {
        self.config.actions.iter().map(|&a| (a, self.q_value(state, a))).collect()
    }

    /// Memory usage estimate.
    pub fn memory_usage(&self) -> usize {
        (self.q_table_a.len() + self.q_table_b.len())
            * (std::mem::size_of::<(StateId, ActionId)>() + std::mem::size_of::<QEntry>())
    }

    /// Measure the max absolute difference between Q_A and Q_B across all
    /// state-action pairs. A low value indicates the two tables agree.
    pub fn table_disagreement(&self) -> f32 {
        let mut max_diff = 0.0f32;
        let mut all_keys: HashSet<(StateId, ActionId)> = HashSet::new();
        for key in self.q_table_a.keys() {
            all_keys.insert(key.clone());
        }
        for key in self.q_table_b.keys() {
            all_keys.insert(key.clone());
        }
        for (state, action) in &all_keys {
            let diff = (self.q_a(state, *action) - self.q_b(state, *action)).abs();
            max_diff = max_diff.max(diff);
        }
        max_diff
    }
}

// ---------------------------------------------------------------------------
// Training Loop (Episode Management)
// ---------------------------------------------------------------------------

/// Result of a single episode.
#[derive(Debug, Clone)]
pub struct EpisodeResult {
    /// Total reward earned in the episode.
    pub total_reward: f32,
    /// Number of steps in the episode.
    pub steps: u64,
    /// Whether the episode terminated naturally (vs max-steps cutoff).
    pub natural_termination: bool,
    /// Final state.
    pub final_state: StateId,
}

/// Trait for environments that can be stepped through.
pub trait Environment {
    /// Reset the environment and return the initial state.
    fn reset(&mut self) -> StateId;
    /// Take an action and return (next_state, reward, done).
    fn step(&mut self, action: ActionId) -> (StateId, f32, bool);
    /// Get the available actions in the current state.
    fn available_actions(&self) -> &[ActionId];
}

/// Run a full training loop for a Q-learner on an environment.
pub fn train_qlearner(
    learner: &mut QLearner,
    env: &mut dyn Environment,
    max_episodes: u64,
    max_steps_per_episode: u64,
    replay_batch_size: usize,
) -> Vec<EpisodeResult> {
    let mut results = Vec::new();

    for _episode in 0..max_episodes {
        if learner.has_converged() {
            break;
        }

        let mut state = env.reset();
        let mut total_reward = 0.0f32;
        let mut steps = 0u64;
        let mut natural_termination = false;

        for _step in 0..max_steps_per_episode {
            let action = learner.select_action(&state);
            let (next_state, reward, done) = env.step(action);

            learner.learn(&Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });

            // Extra learning from replay buffer.
            if replay_batch_size > 0 {
                learner.learn_from_replay(replay_batch_size);
            }

            total_reward += reward;
            steps += 1;
            state = next_state;

            if done {
                natural_termination = true;
                break;
            }
        }

        if !natural_termination {
            // Force end the episode if we hit max steps.
            learner.force_end_episode();
        }

        results.push(EpisodeResult {
            total_reward,
            steps,
            natural_termination,
            final_state: state,
        });
    }

    results
}

/// Run a full training loop for a SARSA learner.
pub fn train_sarsa(
    learner: &mut SarsaLearner,
    env: &mut dyn Environment,
    max_episodes: u64,
    max_steps_per_episode: u64,
) -> Vec<EpisodeResult> {
    let mut results = Vec::new();

    for _episode in 0..max_episodes {
        let mut state = env.reset();
        let mut action = learner.select_action(&state);
        let mut total_reward = 0.0f32;
        let mut steps = 0u64;
        let mut natural_termination = false;

        for _step in 0..max_steps_per_episode {
            let (next_state, reward, done) = env.step(action);
            let next_action = if done {
                learner.config.actions[0]
            } else {
                learner.select_action(&next_state)
            };

            learner.learn_sarsa(&SarsaExperience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                next_action,
                done,
            });

            total_reward += reward;
            steps += 1;
            state = next_state;
            action = next_action;

            if done {
                natural_termination = true;
                break;
            }
        }

        if !natural_termination {
            learner.force_end_episode();
        }

        results.push(EpisodeResult {
            total_reward,
            steps,
            natural_termination,
            final_state: state,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Multi-agent Learning Coordinator
// ---------------------------------------------------------------------------

/// Coordinates learning across multiple independent agents sharing an environment.
/// Each agent has its own Q-table but can optionally share experience.
pub struct MultiAgentCoordinator {
    agents: Vec<QLearner>,
    shared_replay: Option<ExperienceReplayBuffer>,
    /// Communication strategy.
    pub strategy: CoordinationStrategy,
    /// Episode count.
    episode_count: u64,
    /// Per-agent episode rewards.
    agent_rewards: Vec<Vec<f32>>,
}

/// Strategy for coordinating multiple agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinationStrategy {
    /// Agents learn independently, no sharing.
    Independent,
    /// Agents share a single replay buffer.
    SharedReplay,
    /// Periodically average Q-values across agents.
    PeriodicAveraging {
        /// Average every N episodes.
        interval: u64,
    },
    /// Best-performing agent's Q-table is copied to others periodically.
    EliteCopy {
        interval: u64,
    },
}

impl MultiAgentCoordinator {
    pub fn new(configs: Vec<QLearningConfig>, strategy: CoordinationStrategy) -> Self {
        let n = configs.len();
        let agents: Vec<QLearner> = configs.into_iter()
            .enumerate()
            .map(|(i, config)| {
                let mut learner = QLearner::new(config);
                learner.rng = SimpleRng::new(42 + i as u64 * 1000);
                learner
            })
            .collect();

        let shared_replay = match strategy {
            CoordinationStrategy::SharedReplay => {
                Some(ExperienceReplayBuffer::new(DEFAULT_REPLAY_CAPACITY * 2))
            }
            _ => None,
        };

        Self {
            agents,
            shared_replay,
            strategy,
            episode_count: 0,
            agent_rewards: vec![Vec::new(); n],
        }
    }

    /// Number of agents.
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Get agent by index.
    pub fn agent(&self, index: usize) -> &QLearner {
        &self.agents[index]
    }

    /// Get mutable agent by index.
    pub fn agent_mut(&mut self, index: usize) -> &mut QLearner {
        &mut self.agents[index]
    }

    /// Feed an experience to a specific agent.
    pub fn feed_experience(&mut self, agent_index: usize, experience: &Experience) {
        self.agents[agent_index].learn(experience);

        // If using shared replay, also store in the shared buffer.
        if let Some(ref mut buffer) = self.shared_replay {
            buffer.push(experience.clone());
        }
    }

    /// Signal end of episode for all agents and apply coordination.
    pub fn end_episode(&mut self, agent_rewards_this_episode: &[f32]) {
        self.episode_count += 1;

        for (i, &reward) in agent_rewards_this_episode.iter().enumerate() {
            if i < self.agent_rewards.len() {
                self.agent_rewards[i].push(reward);
            }
        }

        match self.strategy {
            CoordinationStrategy::SharedReplay => {
                // Each agent learns from the shared buffer.
                if let Some(ref mut buffer) = self.shared_replay {
                    let batch = buffer.sample(DEFAULT_BATCH_SIZE);
                    // We need to apply the batch to each agent manually since we
                    // can't borrow the buffer and agents simultaneously inside
                    // learn_from_replay. Instead, do the updates directly.
                    for agent in &mut self.agents {
                        for exp in &batch {
                            let current_q = agent.q_value(&exp.state, exp.action);
                            let max_next_q = if exp.done {
                                0.0
                            } else {
                                agent.config.actions.iter()
                                    .map(|&a| agent.q_value(&exp.next_state, a))
                                    .fold(f32::NEG_INFINITY, f32::max)
                            };
                            let target = exp.reward + agent.config.discount_factor * max_next_q;
                            let new_q = current_q + agent.current_lr * (target - current_q);
                            let key = (exp.state.clone(), exp.action);
                            let entry = agent.q_table.entry(key).or_insert_with(QEntry::new);
                            entry.value = new_q;
                            entry.visit_count += 1;
                        }
                    }
                }
            }
            CoordinationStrategy::PeriodicAveraging { interval } => {
                if self.episode_count % interval == 0 && self.agents.len() > 1 {
                    self.average_q_tables();
                }
            }
            CoordinationStrategy::EliteCopy { interval } => {
                if self.episode_count % interval == 0 && self.agents.len() > 1 {
                    self.copy_elite();
                }
            }
            CoordinationStrategy::Independent => {}
        }
    }

    /// Average Q-values across all agents' tables.
    fn average_q_tables(&mut self) {
        // Collect all keys.
        let mut all_keys: HashSet<(StateId, ActionId)> = HashSet::new();
        for agent in &self.agents {
            for key in agent.q_table.keys() {
                all_keys.insert(key.clone());
            }
        }

        // Compute averages.
        let n = self.agents.len() as f32;
        let mut averages: HashMap<(StateId, ActionId), f32> = HashMap::new();
        for key in &all_keys {
            let sum: f32 = self.agents.iter()
                .map(|agent| {
                    agent.q_table.get(key).map(|e| e.value).unwrap_or(0.0)
                })
                .sum();
            averages.insert(key.clone(), sum / n);
        }

        // Apply averages to all agents.
        for agent in &mut self.agents {
            for (key, &avg) in &averages {
                let entry = agent.q_table.entry(key.clone()).or_insert_with(QEntry::new);
                entry.value = avg;
            }
        }
    }

    /// Copy the best-performing agent's Q-table to all others.
    fn copy_elite(&mut self) {
        if self.agents.is_empty() {
            return;
        }
        // Find the agent with the highest recent average reward.
        let best_idx = self.agent_rewards.iter()
            .enumerate()
            .map(|(i, rewards)| {
                let recent: f32 = rewards.iter().rev().take(20).sum::<f32>()
                    / rewards.len().min(20).max(1) as f32;
                (i, recent)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let elite_table = self.agents[best_idx].q_table.clone();
        for (i, agent) in self.agents.iter_mut().enumerate() {
            if i != best_idx {
                agent.q_table = elite_table.clone();
            }
        }
    }

    /// Get the best agent (highest recent average reward).
    pub fn best_agent_index(&self) -> usize {
        self.agent_rewards.iter()
            .enumerate()
            .map(|(i, rewards)| {
                let recent: f32 = rewards.iter().rev().take(20).sum::<f32>()
                    / rewards.len().min(20).max(1) as f32;
                (i, recent)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get per-agent average rewards (for benchmarking comparison).
    pub fn agent_average_rewards(&self) -> Vec<f32> {
        self.agent_rewards.iter()
            .map(|rewards| {
                if rewards.is_empty() {
                    0.0
                } else {
                    rewards.iter().sum::<f32>() / rewards.len() as f32
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Benchmarking
// ---------------------------------------------------------------------------

/// Results from benchmarking multiple learning algorithms.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Algorithm name.
    pub algorithm: String,
    /// Per-episode rewards.
    pub episode_rewards: Vec<f32>,
    /// Per-episode step counts.
    pub episode_steps: Vec<u64>,
    /// Total training time (wall clock, if measured externally).
    pub total_time_ms: f64,
    /// Episode at which convergence was detected (None if not converged).
    pub convergence_episode: Option<u64>,
    /// Mean reward over last 100 episodes.
    pub final_mean_reward: f32,
    /// Reward standard deviation over last 100 episodes.
    pub final_reward_std: f32,
    /// Total number of unique states discovered.
    pub states_discovered: usize,
    /// Total number of Q-table entries.
    pub q_table_size: usize,
}

impl BenchmarkResult {
    /// Compute from episode results and stats.
    pub fn from_results(
        algorithm: &str,
        results: &[EpisodeResult],
        stats: &LearningStats,
    ) -> Self {
        let rewards: Vec<f32> = results.iter().map(|r| r.total_reward).collect();
        let steps: Vec<u64> = results.iter().map(|r| r.steps).collect();

        let n = rewards.len();
        let window = n.min(100);
        let final_rewards = if n > 0 { &rewards[n - window..] } else { &rewards[..] };
        let mean = if !final_rewards.is_empty() {
            final_rewards.iter().sum::<f32>() / final_rewards.len() as f32
        } else {
            0.0
        };
        let variance = if final_rewards.len() > 1 {
            final_rewards.iter()
                .map(|r| (r - mean) * (r - mean))
                .sum::<f32>()
                / (final_rewards.len() - 1) as f32
        } else {
            0.0
        };

        BenchmarkResult {
            algorithm: algorithm.to_string(),
            episode_rewards: rewards,
            episode_steps: steps,
            total_time_ms: 0.0,
            convergence_episode: stats.convergence_episode,
            final_mean_reward: mean,
            final_reward_std: variance.sqrt(),
            states_discovered: stats.unique_states,
            q_table_size: stats.q_table_size,
        }
    }

    /// Compute the area under the learning curve (sum of rewards across episodes).
    /// Higher is better -- it measures both final performance and learning speed.
    pub fn area_under_curve(&self) -> f32 {
        self.episode_rewards.iter().sum()
    }

    /// Compute the episode at which mean reward first exceeds a threshold.
    pub fn episodes_to_threshold(&self, threshold: f32, window: usize) -> Option<usize> {
        if self.episode_rewards.len() < window {
            return None;
        }
        for i in window..=self.episode_rewards.len() {
            let mean = self.episode_rewards[i - window..i].iter().sum::<f32>() / window as f32;
            if mean >= threshold {
                return Some(i);
            }
        }
        None
    }
}

/// Compare two benchmark results.
pub fn compare_benchmarks(a: &BenchmarkResult, b: &BenchmarkResult) -> BenchmarkComparison {
    BenchmarkComparison {
        algo_a: a.algorithm.clone(),
        algo_b: b.algorithm.clone(),
        reward_diff: a.final_mean_reward - b.final_mean_reward,
        convergence_diff: match (a.convergence_episode, b.convergence_episode) {
            (Some(ea), Some(eb)) => Some(ea as i64 - eb as i64),
            _ => None,
        },
        auc_diff: a.area_under_curve() - b.area_under_curve(),
        a_better_final_reward: a.final_mean_reward > b.final_mean_reward,
        a_faster_convergence: match (a.convergence_episode, b.convergence_episode) {
            (Some(ea), Some(eb)) => ea < eb,
            (Some(_), None) => true,
            _ => false,
        },
    }
}

/// Comparison between two benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    pub algo_a: String,
    pub algo_b: String,
    pub reward_diff: f32,
    pub convergence_diff: Option<i64>,
    pub auc_diff: f32,
    pub a_better_final_reward: bool,
    pub a_faster_convergence: bool,
}

impl fmt::Display for BenchmarkComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} vs {}: reward_diff={:.3}, auc_diff={:.1}, better_reward={}, faster={}",
            self.algo_a, self.algo_b, self.reward_diff, self.auc_diff,
            if self.a_better_final_reward { &self.algo_a } else { &self.algo_b },
            if self.a_faster_convergence { &self.algo_a } else { &self.algo_b },
        )
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
            epsilon: 0.0,
            ..Default::default()
        });

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
    fn test_serialization_roundtrip() {
        let mut learner = QLearner::default();
        learner.learn(&Experience {
            state: StateId::single(0),
            action: ActionId(0),
            reward: 1.0,
            next_state: StateId::single(1),
            done: false,
        });
        learner.learn(&Experience {
            state: StateId::single(1),
            action: ActionId(2),
            reward: 5.0,
            next_state: StateId::single(2),
            done: true,
        });

        let bytes = learner.serialize();
        assert!(!bytes.is_empty());

        let mut restored = QLearner::default();
        assert!(restored.deserialize(&bytes).is_ok());
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

    #[test]
    fn test_experience_replay_buffer() {
        let mut buffer = ExperienceReplayBuffer::new(5);
        for i in 0..10 {
            buffer.push(Experience {
                state: StateId::single(i),
                action: ActionId(0),
                reward: i as f32,
                next_state: StateId::single(i + 1),
                done: false,
            });
        }
        assert_eq!(buffer.len(), 5);
        assert!(buffer.is_full());

        let batch = buffer.sample(3);
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_state_discretizer() {
        let disc = StateDiscretizer::uniform(2, 0.0, 10.0, 5);
        let state = disc.discretize(&[2.5, 7.5]);
        assert_eq!(state.dim(), 2);
        assert_eq!(state.0[0], 1); // 2.5 is in bin 1 (0-2, 2-4, ...)
        assert_eq!(state.0[1], 3); // 7.5 is in bin 3
    }

    #[test]
    fn test_reward_shaping() {
        let mut shaper = RewardShaper::new(0.9);
        shaper.set_potential(StateId::single(0), 0.0);
        shaper.set_potential(StateId::single(1), 1.0);

        let shaped = shaper.shape(1.0, &StateId::single(0), &StateId::single(1), false);
        // r + gamma * Phi(s') - Phi(s) = 1.0 + 0.9 * 1.0 - 0.0 = 1.9
        assert!((shaped - 1.9).abs() < 1e-5);
    }

    #[test]
    fn test_sarsa_basic() {
        let mut learner = SarsaLearner::new(QLearningConfig {
            actions: vec![ActionId(0), ActionId(1)],
            learning_rate: 0.5,
            epsilon: 0.0,
            ..Default::default()
        });

        learner.learn_sarsa(&SarsaExperience {
            state: StateId::single(0),
            action: ActionId(1),
            reward: 10.0,
            next_state: StateId::single(1),
            next_action: ActionId(0),
            done: true,
        });

        assert!(learner.q_value(&StateId::single(0), ActionId(1)) > 0.0);
    }

    #[test]
    fn test_double_q_learning() {
        let mut learner = DoubleQLearner::new(QLearningConfig {
            actions: vec![ActionId(0), ActionId(1)],
            learning_rate: 0.5,
            epsilon: 0.0,
            ..Default::default()
        });

        for _ in 0..20 {
            learner.learn(&Experience {
                state: StateId::single(0),
                action: ActionId(1),
                reward: 10.0,
                next_state: StateId::single(1),
                done: true,
            });
        }

        // Action 1 should be preferred.
        assert!(learner.q_value(&StateId::single(0), ActionId(1))
            > learner.q_value(&StateId::single(0), ActionId(0)));
    }

    #[test]
    fn test_convergence_detector() {
        let mut detector = ConvergenceDetector::new(5, 0.05);
        let mut policy = HashMap::new();
        policy.insert(StateId::single(0), ActionId(0));
        policy.insert(StateId::single(1), ActionId(1));

        // Feed the same policy repeatedly.
        for i in 0..20 {
            let converged = detector.check(&policy, 10.0);
            if i > 10 {
                // Should eventually detect convergence.
                if converged {
                    break;
                }
            }
        }
        assert!(detector.change_rate() < 0.1);
    }

    #[test]
    fn test_softmax_action_selection() {
        let mut learner = QLearner::new(QLearningConfig {
            actions: vec![ActionId(0), ActionId(1), ActionId(2)],
            ..Default::default()
        });

        // Set Q-values so action 1 is best.
        learner.q_table.insert(
            (StateId::single(0), ActionId(0)),
            QEntry::with_value(1.0),
        );
        learner.q_table.insert(
            (StateId::single(0), ActionId(1)),
            QEntry::with_value(10.0),
        );
        learner.q_table.insert(
            (StateId::single(0), ActionId(2)),
            QEntry::with_value(2.0),
        );

        // With very low temperature, should almost always pick action 1.
        let mut counts = [0u32; 3];
        for _ in 0..100 {
            let a = learner.select_action_softmax(&StateId::single(0), 0.1);
            counts[a.0 as usize] += 1;
        }
        assert!(counts[1] > 80);
    }

    #[test]
    fn test_policy_visualization() {
        let mut learner = QLearner::new(QLearningConfig {
            actions: vec![ActionId(0), ActionId(1)],
            epsilon: 0.0,
            ..Default::default()
        });

        learner.q_table.insert(
            (StateId::from_features(&[0, 0]), ActionId(0)),
            QEntry::with_value(5.0),
        );
        learner.q_table.insert(
            (StateId::from_features(&[0, 0]), ActionId(1)),
            QEntry::with_value(3.0),
        );
        learner.q_table.insert(
            (StateId::from_features(&[1, 0]), ActionId(0)),
            QEntry::with_value(2.0),
        );
        learner.q_table.insert(
            (StateId::from_features(&[1, 0]), ActionId(1)),
            QEntry::with_value(7.0),
        );

        let viz = learner.visualize_policy();
        assert_eq!(viz.num_states(), 2);
        assert_eq!(*viz.best_actions.get(&StateId::from_features(&[0, 0])).unwrap(), ActionId(0));
        assert_eq!(*viz.best_actions.get(&StateId::from_features(&[1, 0])).unwrap(), ActionId(1));
    }

    #[test]
    fn test_multi_agent_coordinator() {
        let configs = vec![QLearningConfig::default(), QLearningConfig::default()];
        let mut coord = MultiAgentCoordinator::new(configs, CoordinationStrategy::Independent);
        assert_eq!(coord.num_agents(), 2);

        coord.feed_experience(0, &Experience {
            state: StateId::single(0),
            action: ActionId(0),
            reward: 5.0,
            next_state: StateId::single(1),
            done: true,
        });

        coord.end_episode(&[5.0, 0.0]);
        let avgs = coord.agent_average_rewards();
        assert_eq!(avgs.len(), 2);
    }

    #[test]
    fn test_ucb_action_selection() {
        let mut learner = QLearner::new(QLearningConfig {
            actions: vec![ActionId(0), ActionId(1), ActionId(2)],
            ..Default::default()
        });

        // With no visits, UCB should prefer unvisited actions (infinite bonus).
        let action = learner.select_action_ucb(&StateId::single(0), 1.0);
        // Any action is valid since none have been visited.
        assert!(action.0 < 3);
    }

    #[test]
    fn test_benchmark_result() {
        let results = vec![
            EpisodeResult { total_reward: 10.0, steps: 50, natural_termination: true, final_state: StateId::single(0) },
            EpisodeResult { total_reward: 15.0, steps: 40, natural_termination: true, final_state: StateId::single(0) },
            EpisodeResult { total_reward: 20.0, steps: 30, natural_termination: true, final_state: StateId::single(0) },
        ];
        let stats = LearningStats {
            episodes: 3,
            unique_states: 5,
            q_table_size: 10,
            ..Default::default()
        };

        let bench = BenchmarkResult::from_results("QLearning", &results, &stats);
        assert_eq!(bench.algorithm, "QLearning");
        assert!((bench.area_under_curve() - 45.0).abs() < 1e-5);
    }
}
