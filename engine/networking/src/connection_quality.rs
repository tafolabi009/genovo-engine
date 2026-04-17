//! Connection quality assessment for the Genovo engine networking layer.
//!
//! Monitors network conditions (latency, jitter, packet loss, bandwidth) and
//! produces a composite quality score with recommendations for adapting the
//! game's network behavior.

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Quality Indicator
// ---------------------------------------------------------------------------

/// Overall connection quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityIndicator {
    /// Score 90-100.
    Excellent,
    /// Score 70-89.
    Good,
    /// Score 50-69.
    Fair,
    /// Score 25-49.
    Poor,
    /// Score 0-24.
    Bad,
}

impl QualityIndicator {
    /// Derive from a numeric score (0-100).
    pub fn from_score(score: u8) -> Self {
        match score {
            90..=100 => QualityIndicator::Excellent,
            70..=89 => QualityIndicator::Good,
            50..=69 => QualityIndicator::Fair,
            25..=49 => QualityIndicator::Poor,
            _ => QualityIndicator::Bad,
        }
    }

    /// Returns a color hint for UI display (RGBA).
    pub fn color(&self) -> [f32; 4] {
        match self {
            QualityIndicator::Excellent => [0.0, 1.0, 0.0, 1.0],
            QualityIndicator::Good => [0.5, 1.0, 0.0, 1.0],
            QualityIndicator::Fair => [1.0, 1.0, 0.0, 1.0],
            QualityIndicator::Poor => [1.0, 0.5, 0.0, 1.0],
            QualityIndicator::Bad => [1.0, 0.0, 0.0, 1.0],
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            QualityIndicator::Excellent => "Connection is excellent, ideal for competitive play",
            QualityIndicator::Good => "Connection is good, suitable for most game modes",
            QualityIndicator::Fair => "Connection has minor issues, some lag may occur",
            QualityIndicator::Poor => "Connection is poor, noticeable lag and packet loss",
            QualityIndicator::Bad => "Connection is very bad, gameplay severely impacted",
        }
    }
}

impl fmt::Display for QualityIndicator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QualityIndicator::Excellent => write!(f, "Excellent"),
            QualityIndicator::Good => write!(f, "Good"),
            QualityIndicator::Fair => write!(f, "Fair"),
            QualityIndicator::Poor => write!(f, "Poor"),
            QualityIndicator::Bad => write!(f, "Bad"),
        }
    }
}

// ---------------------------------------------------------------------------
// Recommendation
// ---------------------------------------------------------------------------

/// An action recommended by the quality assessor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Recommendation {
    /// No changes needed.
    NoAction,
    /// Reduce the network send rate.
    ReduceSendRate { target_rate_bps: u64 },
    /// Increase the jitter buffer size.
    IncreaseBuffer { target_ms: u32 },
    /// Decrease the jitter buffer size (conditions improved).
    DecreaseBuffer { target_ms: u32 },
    /// Increase interpolation delay.
    IncreaseInterpolationDelay { target_ms: u32 },
    /// Enable packet redundancy (send duplicates).
    EnableRedundancy,
    /// Disable packet redundancy.
    DisableRedundancy,
    /// Switch to relay (direct connection too poor).
    SwitchToRelay,
    /// Switch from relay to direct (conditions improved).
    SwitchToDirect,
    /// Reduce tick rate.
    ReduceTickRate { target_hz: u32 },
    /// Reduce update frequency for non-critical entities.
    ReduceEntityUpdateRate,
    /// Warn the player about connection issues.
    WarnPlayer(String),
}

impl fmt::Display for Recommendation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Recommendation::NoAction => write!(f, "No action needed"),
            Recommendation::ReduceSendRate { target_rate_bps } => {
                write!(f, "Reduce send rate to {} bps", target_rate_bps)
            }
            Recommendation::IncreaseBuffer { target_ms } => {
                write!(f, "Increase buffer to {}ms", target_ms)
            }
            Recommendation::DecreaseBuffer { target_ms } => {
                write!(f, "Decrease buffer to {}ms", target_ms)
            }
            Recommendation::IncreaseInterpolationDelay { target_ms } => {
                write!(f, "Increase interp delay to {}ms", target_ms)
            }
            Recommendation::EnableRedundancy => write!(f, "Enable packet redundancy"),
            Recommendation::DisableRedundancy => write!(f, "Disable packet redundancy"),
            Recommendation::SwitchToRelay => write!(f, "Switch to relay"),
            Recommendation::SwitchToDirect => write!(f, "Switch to direct"),
            Recommendation::ReduceTickRate { target_hz } => {
                write!(f, "Reduce tick rate to {} Hz", target_hz)
            }
            Recommendation::ReduceEntityUpdateRate => {
                write!(f, "Reduce entity update rate")
            }
            Recommendation::WarnPlayer(msg) => write!(f, "Warn player: {}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// Ping Graph Data
// ---------------------------------------------------------------------------

/// Data point for the ping graph visualization.
#[derive(Debug, Clone, Copy)]
pub struct PingGraphPoint {
    /// RTT in milliseconds.
    pub rtt_ms: f64,
    /// Frame/sample index.
    pub sample_index: u64,
    /// Timestamp.
    pub timestamp: Instant,
    /// Whether this sample had packet loss.
    pub lost: bool,
}

/// Rolling ping graph data for UI visualization.
#[derive(Debug, Clone)]
pub struct PingGraph {
    /// Data points.
    points: VecDeque<PingGraphPoint>,
    /// Maximum number of points to store.
    max_points: usize,
    /// Next sample index.
    next_index: u64,
}

impl PingGraph {
    /// Create a new ping graph with the given capacity.
    pub fn new(max_points: usize) -> Self {
        Self {
            points: VecDeque::with_capacity(max_points),
            max_points,
            next_index: 0,
        }
    }

    /// Add a data point.
    pub fn push(&mut self, rtt_ms: f64, lost: bool) {
        if self.points.len() >= self.max_points {
            self.points.pop_front();
        }
        self.points.push_back(PingGraphPoint {
            rtt_ms,
            sample_index: self.next_index,
            timestamp: Instant::now(),
            lost,
        });
        self.next_index += 1;
    }

    /// Get all data points.
    pub fn points(&self) -> &VecDeque<PingGraphPoint> {
        &self.points
    }

    /// Get the minimum RTT in the window.
    pub fn min_rtt(&self) -> f64 {
        self.points
            .iter()
            .filter(|p| !p.lost)
            .map(|p| p.rtt_ms)
            .fold(f64::MAX, f64::min)
    }

    /// Get the maximum RTT in the window.
    pub fn max_rtt(&self) -> f64 {
        self.points
            .iter()
            .filter(|p| !p.lost)
            .map(|p| p.rtt_ms)
            .fold(0.0f64, f64::max)
    }

    /// Get the average RTT in the window.
    pub fn avg_rtt(&self) -> f64 {
        let valid: Vec<f64> = self
            .points
            .iter()
            .filter(|p| !p.lost)
            .map(|p| p.rtt_ms)
            .collect();
        if valid.is_empty() {
            0.0
        } else {
            valid.iter().sum::<f64>() / valid.len() as f64
        }
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.points.clear();
        self.next_index = 0;
    }
}

// ---------------------------------------------------------------------------
// ConnectionQualityConfig
// ---------------------------------------------------------------------------

/// Configuration for the connection quality assessor.
#[derive(Debug, Clone)]
pub struct ConnectionQualityConfig {
    /// Window size for computing rolling averages.
    pub window_size: usize,
    /// RTT threshold for "excellent" quality (ms).
    pub excellent_rtt_ms: f64,
    /// RTT threshold for "good" quality (ms).
    pub good_rtt_ms: f64,
    /// RTT threshold for "fair" quality (ms).
    pub fair_rtt_ms: f64,
    /// RTT threshold for "poor" quality (ms).
    pub poor_rtt_ms: f64,
    /// Jitter threshold for quality degradation (ms).
    pub jitter_threshold_ms: f64,
    /// Packet loss threshold for quality degradation (0.0-1.0).
    pub loss_threshold: f64,
    /// Number of ping graph data points.
    pub ping_graph_points: usize,
    /// Minimum interval between recommendation changes.
    pub recommendation_cooldown: Duration,
}

impl Default for ConnectionQualityConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            excellent_rtt_ms: 30.0,
            good_rtt_ms: 60.0,
            fair_rtt_ms: 120.0,
            poor_rtt_ms: 250.0,
            jitter_threshold_ms: 20.0,
            loss_threshold: 0.05,
            ping_graph_points: 200,
            recommendation_cooldown: Duration::from_secs(5),
        }
    }
}

// ---------------------------------------------------------------------------
// ConnectionQualityAssessor
// ---------------------------------------------------------------------------

/// Assesses connection quality based on RTT, jitter, packet loss, and bandwidth.
pub struct ConnectionQualityAssessor {
    /// Configuration.
    config: ConnectionQualityConfig,
    /// Rolling RTT samples (milliseconds).
    rtt_samples: VecDeque<f64>,
    /// Rolling jitter samples (ms).
    jitter_samples: VecDeque<f64>,
    /// Packet loss window: true = received, false = lost.
    loss_window: VecDeque<bool>,
    /// Bandwidth estimation samples (bytes per second).
    bandwidth_samples: VecDeque<f64>,
    /// Ping graph for visualization.
    ping_graph: PingGraph,
    /// Current quality score (0-100).
    current_score: u8,
    /// Current quality indicator.
    current_indicator: QualityIndicator,
    /// Current recommendations.
    current_recommendations: Vec<Recommendation>,
    /// Last time recommendations were updated.
    last_recommendation_time: Instant,
    /// Previous RTT for jitter calculation.
    prev_rtt_ms: Option<f64>,
    /// Total packets tracked.
    total_packets: u64,
    /// Total lost packets.
    total_lost: u64,
    /// Smoothed RTT (EWMA).
    smoothed_rtt: f64,
    /// RTT variance (for adaptive algorithms).
    rtt_variance: f64,
    /// Estimated bandwidth (bytes/sec).
    estimated_bandwidth: f64,
}

impl ConnectionQualityAssessor {
    /// Create a new assessor with default configuration.
    pub fn new() -> Self {
        Self::with_config(ConnectionQualityConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: ConnectionQualityConfig) -> Self {
        let graph_points = config.ping_graph_points;
        Self {
            config,
            rtt_samples: VecDeque::new(),
            jitter_samples: VecDeque::new(),
            loss_window: VecDeque::new(),
            bandwidth_samples: VecDeque::new(),
            ping_graph: PingGraph::new(graph_points),
            current_score: 100,
            current_indicator: QualityIndicator::Excellent,
            current_recommendations: vec![Recommendation::NoAction],
            last_recommendation_time: Instant::now(),
            prev_rtt_ms: None,
            total_packets: 0,
            total_lost: 0,
            smoothed_rtt: 0.0,
            rtt_variance: 0.0,
            estimated_bandwidth: 0.0,
        }
    }

    /// Record a successful ping/pong round trip.
    pub fn record_rtt(&mut self, rtt_ms: f64) {
        // Add to samples.
        self.rtt_samples.push_back(rtt_ms);
        if self.rtt_samples.len() > self.config.window_size {
            self.rtt_samples.pop_front();
        }

        // Calculate jitter (variation from previous RTT).
        if let Some(prev) = self.prev_rtt_ms {
            let jitter = (rtt_ms - prev).abs();
            self.jitter_samples.push_back(jitter);
            if self.jitter_samples.len() > self.config.window_size {
                self.jitter_samples.pop_front();
            }
        }
        self.prev_rtt_ms = Some(rtt_ms);

        // EWMA smoothed RTT (alpha = 0.125 as in TCP).
        if self.smoothed_rtt == 0.0 {
            self.smoothed_rtt = rtt_ms;
            self.rtt_variance = rtt_ms / 2.0;
        } else {
            let alpha = 0.125;
            let beta = 0.25;
            self.rtt_variance =
                (1.0 - beta) * self.rtt_variance + beta * (rtt_ms - self.smoothed_rtt).abs();
            self.smoothed_rtt = (1.0 - alpha) * self.smoothed_rtt + alpha * rtt_ms;
        }

        // Ping graph.
        self.ping_graph.push(rtt_ms, false);

        // Mark packet as received.
        self.record_packet(true);

        // Recalculate quality.
        self.recalculate();
    }

    /// Record a packet as received or lost.
    pub fn record_packet(&mut self, received: bool) {
        self.total_packets += 1;
        if !received {
            self.total_lost += 1;
            self.ping_graph.push(0.0, true);
        }

        self.loss_window.push_back(received);
        if self.loss_window.len() > self.config.window_size {
            self.loss_window.pop_front();
        }
    }

    /// Record a bandwidth sample.
    pub fn record_bandwidth(&mut self, bytes_per_sec: f64) {
        self.bandwidth_samples.push_back(bytes_per_sec);
        if self.bandwidth_samples.len() > self.config.window_size {
            self.bandwidth_samples.pop_front();
        }

        // EWMA bandwidth estimate.
        if self.estimated_bandwidth == 0.0 {
            self.estimated_bandwidth = bytes_per_sec;
        } else {
            self.estimated_bandwidth =
                0.9 * self.estimated_bandwidth + 0.1 * bytes_per_sec;
        }
    }

    /// Recalculate the quality score and recommendations.
    fn recalculate(&mut self) {
        let avg_rtt = self.avg_rtt();
        let avg_jitter = self.avg_jitter();
        let loss_rate = self.packet_loss_rate();

        // Score components (each out of ~33, total max 100).
        // RTT score: 33 points.
        let rtt_score = if avg_rtt <= self.config.excellent_rtt_ms {
            33.0
        } else if avg_rtt <= self.config.good_rtt_ms {
            25.0
        } else if avg_rtt <= self.config.fair_rtt_ms {
            17.0
        } else if avg_rtt <= self.config.poor_rtt_ms {
            8.0
        } else {
            0.0
        };

        // Jitter score: 33 points.
        let jitter_score = if avg_jitter <= 5.0 {
            33.0
        } else if avg_jitter <= 15.0 {
            25.0
        } else if avg_jitter <= self.config.jitter_threshold_ms {
            17.0
        } else if avg_jitter <= 50.0 {
            8.0
        } else {
            0.0
        };

        // Loss score: 34 points.
        let loss_score = if loss_rate <= 0.005 {
            34.0
        } else if loss_rate <= 0.02 {
            25.0
        } else if loss_rate <= self.config.loss_threshold {
            17.0
        } else if loss_rate <= 0.15 {
            8.0
        } else {
            0.0
        };

        let total: f64 = rtt_score + jitter_score + loss_score;
        let total = total.round() as u8;
        self.current_score = total.min(100);
        self.current_indicator = QualityIndicator::from_score(self.current_score);

        // Generate recommendations.
        if self.last_recommendation_time.elapsed() >= self.config.recommendation_cooldown {
            self.current_recommendations = self.generate_recommendations(avg_rtt, avg_jitter, loss_rate);
            self.last_recommendation_time = Instant::now();
        }
    }

    fn generate_recommendations(
        &self,
        avg_rtt: f64,
        avg_jitter: f64,
        loss_rate: f64,
    ) -> Vec<Recommendation> {
        let mut recs = Vec::new();

        if loss_rate > 0.10 {
            recs.push(Recommendation::EnableRedundancy);
            recs.push(Recommendation::ReduceSendRate {
                target_rate_bps: (self.estimated_bandwidth * 0.5) as u64,
            });
        } else if loss_rate > self.config.loss_threshold {
            recs.push(Recommendation::ReduceSendRate {
                target_rate_bps: (self.estimated_bandwidth * 0.75) as u64,
            });
        }

        if avg_jitter > self.config.jitter_threshold_ms * 2.0 {
            let buffer_ms = (avg_jitter * 2.5) as u32;
            recs.push(Recommendation::IncreaseBuffer {
                target_ms: buffer_ms.max(50),
            });
        } else if avg_jitter > self.config.jitter_threshold_ms {
            let buffer_ms = (avg_jitter * 1.5) as u32;
            recs.push(Recommendation::IncreaseBuffer {
                target_ms: buffer_ms.max(30),
            });
        } else if avg_jitter < 5.0 && avg_rtt < self.config.good_rtt_ms {
            recs.push(Recommendation::DecreaseBuffer { target_ms: 20 });
        }

        if avg_rtt > self.config.poor_rtt_ms {
            recs.push(Recommendation::IncreaseInterpolationDelay {
                target_ms: (avg_rtt * 0.5) as u32,
            });
            recs.push(Recommendation::ReduceEntityUpdateRate);
        }

        if self.current_score < 25 {
            recs.push(Recommendation::WarnPlayer(
                "Your connection quality is very poor. Consider switching to a closer server."
                    .to_string(),
            ));
        }

        if recs.is_empty() {
            recs.push(Recommendation::NoAction);
        }

        recs
    }

    // -- Accessors --

    /// Current quality score (0-100).
    pub fn score(&self) -> u8 {
        self.current_score
    }

    /// Current quality indicator.
    pub fn indicator(&self) -> QualityIndicator {
        self.current_indicator
    }

    /// Current recommendations.
    pub fn recommendations(&self) -> &[Recommendation] {
        &self.current_recommendations
    }

    /// Average RTT in the window (ms).
    pub fn avg_rtt(&self) -> f64 {
        if self.rtt_samples.is_empty() {
            return 0.0;
        }
        self.rtt_samples.iter().sum::<f64>() / self.rtt_samples.len() as f64
    }

    /// Smoothed RTT (EWMA, ms).
    pub fn smoothed_rtt(&self) -> f64 {
        self.smoothed_rtt
    }

    /// Minimum RTT in the window (ms).
    pub fn min_rtt(&self) -> f64 {
        self.rtt_samples
            .iter()
            .copied()
            .fold(f64::MAX, f64::min)
    }

    /// Maximum RTT in the window (ms).
    pub fn max_rtt(&self) -> f64 {
        self.rtt_samples
            .iter()
            .copied()
            .fold(0.0f64, f64::max)
    }

    /// Average jitter in the window (ms).
    pub fn avg_jitter(&self) -> f64 {
        if self.jitter_samples.is_empty() {
            return 0.0;
        }
        self.jitter_samples.iter().sum::<f64>() / self.jitter_samples.len() as f64
    }

    /// Jitter variance.
    pub fn jitter_variance(&self) -> f64 {
        if self.jitter_samples.len() < 2 {
            return 0.0;
        }
        let mean = self.avg_jitter();
        let variance: f64 = self
            .jitter_samples
            .iter()
            .map(|j| (j - mean) * (j - mean))
            .sum::<f64>()
            / (self.jitter_samples.len() - 1) as f64;
        variance
    }

    /// Jitter standard deviation (ms).
    pub fn jitter_stddev(&self) -> f64 {
        self.jitter_variance().sqrt()
    }

    /// Packet loss rate in the window (0.0-1.0).
    pub fn packet_loss_rate(&self) -> f64 {
        if self.loss_window.is_empty() {
            return 0.0;
        }
        let lost = self.loss_window.iter().filter(|&&r| !r).count();
        lost as f64 / self.loss_window.len() as f64
    }

    /// Lifetime packet loss rate.
    pub fn lifetime_loss_rate(&self) -> f64 {
        if self.total_packets == 0 {
            return 0.0;
        }
        self.total_lost as f64 / self.total_packets as f64
    }

    /// Estimated bandwidth (bytes/sec).
    pub fn estimated_bandwidth(&self) -> f64 {
        self.estimated_bandwidth
    }

    /// RTT variance (ms^2).
    pub fn rtt_variance(&self) -> f64 {
        self.rtt_variance
    }

    /// Reference to the ping graph data.
    pub fn ping_graph(&self) -> &PingGraph {
        &self.ping_graph
    }

    /// Total packets tracked.
    pub fn total_packets(&self) -> u64 {
        self.total_packets
    }

    /// Generate a snapshot of the current quality state.
    pub fn snapshot(&self) -> QualitySnapshot {
        QualitySnapshot {
            score: self.current_score,
            indicator: self.current_indicator,
            avg_rtt_ms: self.avg_rtt(),
            smoothed_rtt_ms: self.smoothed_rtt,
            min_rtt_ms: self.min_rtt(),
            max_rtt_ms: self.max_rtt(),
            avg_jitter_ms: self.avg_jitter(),
            jitter_stddev_ms: self.jitter_stddev(),
            packet_loss_rate: self.packet_loss_rate(),
            estimated_bandwidth_bps: self.estimated_bandwidth as u64,
            total_packets: self.total_packets,
            total_lost: self.total_lost,
        }
    }

    /// Reset all collected data.
    pub fn reset(&mut self) {
        self.rtt_samples.clear();
        self.jitter_samples.clear();
        self.loss_window.clear();
        self.bandwidth_samples.clear();
        self.ping_graph.clear();
        self.current_score = 100;
        self.current_indicator = QualityIndicator::Excellent;
        self.current_recommendations = vec![Recommendation::NoAction];
        self.prev_rtt_ms = None;
        self.total_packets = 0;
        self.total_lost = 0;
        self.smoothed_rtt = 0.0;
        self.rtt_variance = 0.0;
        self.estimated_bandwidth = 0.0;
    }
}

impl Default for ConnectionQualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ConnectionQualityAssessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConnectionQualityAssessor")
            .field("score", &self.current_score)
            .field("indicator", &self.current_indicator)
            .field("rtt", &format!("{:.1}ms", self.smoothed_rtt))
            .field("jitter", &format!("{:.1}ms", self.avg_jitter()))
            .field("loss", &format!("{:.1}%", self.packet_loss_rate() * 100.0))
            .finish()
    }
}

/// A snapshot of quality metrics at a point in time.
#[derive(Debug, Clone)]
pub struct QualitySnapshot {
    pub score: u8,
    pub indicator: QualityIndicator,
    pub avg_rtt_ms: f64,
    pub smoothed_rtt_ms: f64,
    pub min_rtt_ms: f64,
    pub max_rtt_ms: f64,
    pub avg_jitter_ms: f64,
    pub jitter_stddev_ms: f64,
    pub packet_loss_rate: f64,
    pub estimated_bandwidth_bps: u64,
    pub total_packets: u64,
    pub total_lost: u64,
}

impl fmt::Display for QualitySnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Connection Quality: {} (score {})", self.indicator, self.score)?;
        writeln!(f, "  RTT:      {:.1}ms (min {:.1}, max {:.1})", self.avg_rtt_ms, self.min_rtt_ms, self.max_rtt_ms)?;
        writeln!(f, "  Jitter:   {:.1}ms (stddev {:.1})", self.avg_jitter_ms, self.jitter_stddev_ms)?;
        writeln!(f, "  Loss:     {:.2}%", self.packet_loss_rate * 100.0)?;
        writeln!(f, "  Bandwidth:{:.0} KB/s", self.estimated_bandwidth_bps as f64 / 1024.0)?;
        writeln!(f, "  Packets:  {} total, {} lost", self.total_packets, self.total_lost)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_from_score() {
        assert_eq!(QualityIndicator::from_score(95), QualityIndicator::Excellent);
        assert_eq!(QualityIndicator::from_score(75), QualityIndicator::Good);
        assert_eq!(QualityIndicator::from_score(55), QualityIndicator::Fair);
        assert_eq!(QualityIndicator::from_score(30), QualityIndicator::Poor);
        assert_eq!(QualityIndicator::from_score(10), QualityIndicator::Bad);
    }

    #[test]
    fn test_excellent_connection() {
        let mut assessor = ConnectionQualityAssessor::new();
        for _ in 0..50 {
            assessor.record_rtt(15.0);
        }

        assert!(assessor.score() >= 90);
        assert_eq!(assessor.indicator(), QualityIndicator::Excellent);
    }

    #[test]
    fn test_poor_connection() {
        let mut assessor = ConnectionQualityAssessor::new();
        for i in 0..50 {
            assessor.record_rtt(200.0 + (i as f64 * 10.0));
        }
        // Add some packet loss.
        for _ in 0..10 {
            assessor.record_packet(false);
        }

        assert!(assessor.score() < 50);
    }

    #[test]
    fn test_jitter_tracking() {
        let mut assessor = ConnectionQualityAssessor::new();
        assessor.record_rtt(50.0);
        assessor.record_rtt(70.0); // 20ms jitter.
        assessor.record_rtt(45.0); // 25ms jitter.
        assessor.record_rtt(60.0); // 15ms jitter.

        assert!(assessor.avg_jitter() > 0.0);
        assert!(assessor.jitter_stddev() > 0.0);
    }

    #[test]
    fn test_packet_loss_rate() {
        let mut assessor = ConnectionQualityAssessor::new();
        for _ in 0..90 {
            assessor.record_packet(true);
        }
        for _ in 0..10 {
            assessor.record_packet(false);
        }

        let loss = assessor.packet_loss_rate();
        assert!((loss - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_ping_graph() {
        let mut graph = PingGraph::new(10);
        for i in 0..15 {
            graph.push(i as f64 * 10.0, false);
        }

        assert_eq!(graph.len(), 10); // Capped at max_points.
        assert!(graph.min_rtt() >= 50.0); // First 5 were evicted.
    }

    #[test]
    fn test_recommendations() {
        let mut assessor = ConnectionQualityAssessor::with_config(ConnectionQualityConfig {
            recommendation_cooldown: Duration::ZERO,
            ..Default::default()
        });

        // Simulate very bad conditions.
        for _ in 0..50 {
            assessor.record_rtt(300.0);
            assessor.record_packet(false);
            assessor.record_packet(false);
        }

        let recs = assessor.recommendations();
        assert!(!recs.is_empty());
        assert!(!recs.contains(&Recommendation::NoAction));
    }

    #[test]
    fn test_bandwidth_estimation() {
        let mut assessor = ConnectionQualityAssessor::new();
        assessor.record_bandwidth(100_000.0);
        assessor.record_bandwidth(110_000.0);
        assessor.record_bandwidth(105_000.0);

        assert!(assessor.estimated_bandwidth() > 0.0);
    }

    #[test]
    fn test_snapshot() {
        let mut assessor = ConnectionQualityAssessor::new();
        for _ in 0..20 {
            assessor.record_rtt(25.0);
        }

        let snap = assessor.snapshot();
        assert!(snap.avg_rtt_ms > 0.0);
        assert_eq!(snap.total_packets, 20);
    }
}
