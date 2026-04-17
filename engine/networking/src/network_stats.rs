//! Comprehensive network statistics for the Genovo networking module.
//!
//! Provides real-time monitoring and historical tracking of network performance
//! metrics including bandwidth, packet counts, round-trip time (RTT), jitter,
//! packet loss, per-channel statistics, and an overall network quality indicator.
//!
//! # Features
//!
//! - Bandwidth tracking (bytes in/out per second, rolling averages)
//! - Packet counting (sent, received, lost, out-of-order)
//! - RTT histogram with configurable bucket widths
//! - Jitter measurement and graphing
//! - Packet loss rate over time (sliding window)
//! - Per-channel statistics (reliable, unreliable, voice, etc.)
//! - Network quality indicator (Excellent, Good, Fair, Poor, Critical)
//! - Data visualization helpers for debug overlays
//! - Periodic snapshots for historical analysis
//!
//! # Example
//!
//! ```ignore
//! let mut stats = NetworkStatsCollector::new();
//! stats.record_packet_sent(128);
//! stats.record_packet_received(256, 45.0);
//! stats.update(0.016);
//!
//! let quality = stats.quality_indicator();
//! println!("Network quality: {quality}");
//! println!("RTT: {:.1}ms", stats.summary().avg_rtt_ms);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of samples to keep for rolling averages.
const ROLLING_WINDOW_SIZE: usize = 120;

/// Number of buckets in the RTT histogram.
const RTT_HISTOGRAM_BUCKETS: usize = 50;

/// Width of each RTT histogram bucket in milliseconds.
const RTT_BUCKET_WIDTH_MS: f64 = 5.0;

/// Maximum RTT value tracked in the histogram (250ms).
const RTT_HISTOGRAM_MAX_MS: f64 = RTT_BUCKET_WIDTH_MS * RTT_HISTOGRAM_BUCKETS as f64;

/// Default packet loss window size (number of packets).
const PACKET_LOSS_WINDOW: usize = 100;

/// Number of data points for jitter graph.
const JITTER_GRAPH_POINTS: usize = 200;

/// Number of data points for bandwidth graph.
const BANDWIDTH_GRAPH_POINTS: usize = 200;

/// Smoothing factor for exponential moving average (0..1).
const EMA_ALPHA: f64 = 0.1;

/// Number of periodic snapshots to retain.
const MAX_SNAPSHOTS: usize = 600;

/// Quality thresholds: RTT (ms).
const QUALITY_RTT_EXCELLENT: f64 = 30.0;
const QUALITY_RTT_GOOD: f64 = 60.0;
const QUALITY_RTT_FAIR: f64 = 120.0;
const QUALITY_RTT_POOR: f64 = 200.0;

/// Quality thresholds: packet loss (fraction 0..1).
const QUALITY_LOSS_EXCELLENT: f64 = 0.005;
const QUALITY_LOSS_GOOD: f64 = 0.02;
const QUALITY_LOSS_FAIR: f64 = 0.05;
const QUALITY_LOSS_POOR: f64 = 0.10;

/// Quality thresholds: jitter (ms).
const QUALITY_JITTER_EXCELLENT: f64 = 5.0;
const QUALITY_JITTER_GOOD: f64 = 15.0;
const QUALITY_JITTER_FAIR: f64 = 30.0;
const QUALITY_JITTER_POOR: f64 = 50.0;

// ---------------------------------------------------------------------------
// Network quality levels
// ---------------------------------------------------------------------------

/// Overall network quality indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NetworkQuality {
    /// RTT < 30ms, loss < 0.5%, jitter < 5ms.
    Excellent,
    /// RTT < 60ms, loss < 2%, jitter < 15ms.
    Good,
    /// RTT < 120ms, loss < 5%, jitter < 30ms.
    Fair,
    /// RTT < 200ms, loss < 10%, jitter < 50ms.
    Poor,
    /// RTT >= 200ms, loss >= 10%, or jitter >= 50ms.
    Critical,
}

impl NetworkQuality {
    /// Returns a numeric score (0 = Critical, 4 = Excellent).
    pub fn score(&self) -> u8 {
        match self {
            Self::Excellent => 4,
            Self::Good => 3,
            Self::Fair => 2,
            Self::Poor => 1,
            Self::Critical => 0,
        }
    }

    /// Returns a color suitable for UI display (RGBA as [f32; 4]).
    pub fn color(&self) -> [f32; 4] {
        match self {
            Self::Excellent => [0.0, 1.0, 0.0, 1.0], // Green
            Self::Good => [0.5, 1.0, 0.0, 1.0],       // Yellow-green
            Self::Fair => [1.0, 1.0, 0.0, 1.0],        // Yellow
            Self::Poor => [1.0, 0.5, 0.0, 1.0],        // Orange
            Self::Critical => [1.0, 0.0, 0.0, 1.0],    // Red
        }
    }

    /// Returns a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent — optimal conditions",
            Self::Good => "Good — minor latency expected",
            Self::Fair => "Fair — noticeable latency",
            Self::Poor => "Poor — significant latency and packet loss",
            Self::Critical => "Critical — severe network issues",
        }
    }
}

impl fmt::Display for NetworkQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Excellent => write!(f, "Excellent"),
            Self::Good => write!(f, "Good"),
            Self::Fair => write!(f, "Fair"),
            Self::Poor => write!(f, "Poor"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

// ---------------------------------------------------------------------------
// RTT Histogram
// ---------------------------------------------------------------------------

/// A histogram of round-trip time samples.
#[derive(Debug, Clone)]
pub struct RttHistogram {
    /// Bucket counts. Each bucket represents a range of RTT values.
    buckets: Vec<u64>,
    /// Width of each bucket in milliseconds.
    bucket_width: f64,
    /// Total number of samples recorded.
    total_samples: u64,
    /// Overflow count (samples above the max tracked RTT).
    overflow: u64,
    /// Minimum RTT observed.
    min_rtt: f64,
    /// Maximum RTT observed.
    max_rtt: f64,
    /// Sum of all RTT values (for computing the mean).
    sum_rtt: f64,
    /// Sum of squared RTT values (for computing variance).
    sum_sq_rtt: f64,
}

impl RttHistogram {
    /// Create a new RTT histogram.
    pub fn new() -> Self {
        Self {
            buckets: vec![0; RTT_HISTOGRAM_BUCKETS],
            bucket_width: RTT_BUCKET_WIDTH_MS,
            total_samples: 0,
            overflow: 0,
            min_rtt: f64::MAX,
            max_rtt: 0.0,
            sum_rtt: 0.0,
            sum_sq_rtt: 0.0,
        }
    }

    /// Create a histogram with custom bucket width and count.
    pub fn with_config(bucket_width: f64, bucket_count: usize) -> Self {
        Self {
            buckets: vec![0; bucket_count],
            bucket_width,
            total_samples: 0,
            overflow: 0,
            min_rtt: f64::MAX,
            max_rtt: 0.0,
            sum_rtt: 0.0,
            sum_sq_rtt: 0.0,
        }
    }

    /// Record an RTT sample.
    pub fn record(&mut self, rtt_ms: f64) {
        self.total_samples += 1;
        self.sum_rtt += rtt_ms;
        self.sum_sq_rtt += rtt_ms * rtt_ms;

        if rtt_ms < self.min_rtt {
            self.min_rtt = rtt_ms;
        }
        if rtt_ms > self.max_rtt {
            self.max_rtt = rtt_ms;
        }

        let bucket_idx = (rtt_ms / self.bucket_width) as usize;
        if bucket_idx < self.buckets.len() {
            self.buckets[bucket_idx] += 1;
        } else {
            self.overflow += 1;
        }
    }

    /// Returns the mean RTT.
    pub fn mean(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.sum_rtt / self.total_samples as f64
        }
    }

    /// Returns the standard deviation of RTT.
    pub fn std_dev(&self) -> f64 {
        if self.total_samples < 2 {
            return 0.0;
        }
        let n = self.total_samples as f64;
        let variance = (self.sum_sq_rtt / n) - (self.sum_rtt / n).powi(2);
        if variance > 0.0 {
            variance.sqrt()
        } else {
            0.0
        }
    }

    /// Returns the approximate percentile value using the histogram.
    pub fn percentile(&self, p: f64) -> f64 {
        if self.total_samples == 0 {
            return 0.0;
        }
        let target = (self.total_samples as f64 * p / 100.0).ceil() as u64;
        let mut cumulative = 0u64;
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return (i as f64 + 0.5) * self.bucket_width;
            }
        }
        // In overflow range.
        self.max_rtt
    }

    /// Returns the median RTT (P50).
    pub fn median(&self) -> f64 {
        self.percentile(50.0)
    }

    /// Returns the P95 RTT.
    pub fn p95(&self) -> f64 {
        self.percentile(95.0)
    }

    /// Returns the P99 RTT.
    pub fn p99(&self) -> f64 {
        self.percentile(99.0)
    }

    /// Returns the bucket counts as normalized values (0..1 range).
    pub fn normalized_buckets(&self) -> Vec<f64> {
        if self.total_samples == 0 {
            return vec![0.0; self.buckets.len()];
        }
        let max_count = self.buckets.iter().copied().max().unwrap_or(1) as f64;
        self.buckets
            .iter()
            .map(|&c| c as f64 / max_count)
            .collect()
    }

    /// Returns the bucket labels (center values in ms).
    pub fn bucket_labels(&self) -> Vec<f64> {
        (0..self.buckets.len())
            .map(|i| (i as f64 + 0.5) * self.bucket_width)
            .collect()
    }

    /// Reset the histogram.
    pub fn reset(&mut self) {
        self.buckets.iter_mut().for_each(|b| *b = 0);
        self.total_samples = 0;
        self.overflow = 0;
        self.min_rtt = f64::MAX;
        self.max_rtt = 0.0;
        self.sum_rtt = 0.0;
        self.sum_sq_rtt = 0.0;
    }
}

impl Default for RttHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RttHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RTT[n={}, mean={:.1}ms, median={:.1}ms, p95={:.1}ms, min={:.1}ms, max={:.1}ms]",
            self.total_samples,
            self.mean(),
            self.median(),
            self.p95(),
            if self.min_rtt == f64::MAX { 0.0 } else { self.min_rtt },
            self.max_rtt
        )
    }
}

// ---------------------------------------------------------------------------
// Rolling sample buffer
// ---------------------------------------------------------------------------

/// A fixed-size circular buffer of f64 samples for rolling statistics.
#[derive(Debug, Clone)]
pub struct RollingSamples {
    data: Vec<f64>,
    write_pos: usize,
    count: usize,
    capacity: usize,
}

impl RollingSamples {
    /// Create a new rolling sample buffer.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            write_pos: 0,
            count: 0,
            capacity,
        }
    }

    /// Push a new sample.
    pub fn push(&mut self, value: f64) {
        self.data[self.write_pos] = value;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Returns the number of samples stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the most recent sample.
    pub fn latest(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let idx = if self.write_pos == 0 {
            self.capacity - 1
        } else {
            self.write_pos - 1
        };
        self.data[idx]
    }

    /// Returns the average of all samples.
    pub fn average(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let sum: f64 = if self.count == self.capacity {
            self.data.iter().sum()
        } else {
            self.data[..self.count].iter().sum()
        };
        sum / self.count as f64
    }

    /// Returns the minimum sample value.
    pub fn min(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let slice = if self.count == self.capacity {
            &self.data[..]
        } else {
            &self.data[..self.count]
        };
        slice.iter().copied().fold(f64::MAX, f64::min)
    }

    /// Returns the maximum sample value.
    pub fn max(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let slice = if self.count == self.capacity {
            &self.data[..]
        } else {
            &self.data[..self.count]
        };
        slice.iter().copied().fold(f64::MIN, f64::max)
    }

    /// Returns all samples in chronological order (oldest first).
    pub fn as_slice_ordered(&self) -> Vec<f64> {
        if self.count < self.capacity {
            self.data[..self.count].to_vec()
        } else {
            let mut result = Vec::with_capacity(self.capacity);
            result.extend_from_slice(&self.data[self.write_pos..]);
            result.extend_from_slice(&self.data[..self.write_pos]);
            result
        }
    }

    /// Returns the samples normalized to 0..1 range for graphing.
    pub fn normalized(&self) -> Vec<f64> {
        let ordered = self.as_slice_ordered();
        if ordered.is_empty() {
            return Vec::new();
        }
        let max = ordered.iter().copied().fold(f64::MIN, f64::max);
        if max <= 0.0 {
            return vec![0.0; ordered.len()];
        }
        ordered.iter().map(|v| v / max).collect()
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.data.iter_mut().for_each(|d| *d = 0.0);
        self.write_pos = 0;
        self.count = 0;
    }
}

// ---------------------------------------------------------------------------
// Per-channel statistics
// ---------------------------------------------------------------------------

/// Statistics for a single network channel.
#[derive(Debug, Clone, Default)]
pub struct ChannelStats {
    /// Name of the channel.
    pub name: String,
    /// Total packets sent on this channel.
    pub packets_sent: u64,
    /// Total packets received on this channel.
    pub packets_received: u64,
    /// Total bytes sent on this channel.
    pub bytes_sent: u64,
    /// Total bytes received on this channel.
    pub bytes_received: u64,
    /// Number of packets lost on this channel.
    pub packets_lost: u64,
    /// Number of packets retransmitted on this channel.
    pub packets_retransmitted: u64,
    /// Average packet size (bytes).
    pub avg_packet_size: f64,
    /// Peak packets per second.
    pub peak_pps: f64,
    /// Current packets per second.
    pub current_pps: f64,
    /// Accumulator for packets in the current second.
    packets_this_second: u64,
    /// Time tracking for per-second calculation.
    second_timer: f64,
}

impl ChannelStats {
    /// Create stats for a named channel.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Record a packet sent on this channel.
    pub fn record_sent(&mut self, bytes: usize) {
        self.packets_sent += 1;
        self.bytes_sent += bytes as u64;
        self.packets_this_second += 1;
        self.update_avg_packet_size(bytes);
    }

    /// Record a packet received on this channel.
    pub fn record_received(&mut self, bytes: usize) {
        self.packets_received += 1;
        self.bytes_received += bytes as u64;
        self.update_avg_packet_size(bytes);
    }

    /// Record a lost packet on this channel.
    pub fn record_lost(&mut self) {
        self.packets_lost += 1;
    }

    /// Record a retransmitted packet on this channel.
    pub fn record_retransmit(&mut self, bytes: usize) {
        self.packets_retransmitted += 1;
        self.bytes_sent += bytes as u64;
        self.packets_this_second += 1;
    }

    /// Update per-second stats.
    pub fn update(&mut self, dt: f64) {
        self.second_timer += dt;
        if self.second_timer >= 1.0 {
            self.current_pps = self.packets_this_second as f64 / self.second_timer;
            if self.current_pps > self.peak_pps {
                self.peak_pps = self.current_pps;
            }
            self.packets_this_second = 0;
            self.second_timer = 0.0;
        }
    }

    /// Update the rolling average packet size.
    fn update_avg_packet_size(&mut self, bytes: usize) {
        let total = self.packets_sent + self.packets_received;
        if total <= 1 {
            self.avg_packet_size = bytes as f64;
        } else {
            self.avg_packet_size =
                self.avg_packet_size * 0.95 + bytes as f64 * 0.05;
        }
    }

    /// Returns the loss rate as a fraction (0..1).
    pub fn loss_rate(&self) -> f64 {
        let total = self.packets_sent + self.packets_received;
        if total == 0 {
            0.0
        } else {
            self.packets_lost as f64 / total as f64
        }
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        let name = self.name.clone();
        *self = Self::new(&name);
    }
}

impl fmt::Display for ChannelStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Channel[{}]: sent={}/{:.1}KB, recv={}/{:.1}KB, lost={}, pps={:.0}",
            self.name,
            self.packets_sent,
            self.bytes_sent as f64 / 1024.0,
            self.packets_received,
            self.bytes_received as f64 / 1024.0,
            self.packets_lost,
            self.current_pps
        )
    }
}

// ---------------------------------------------------------------------------
// Network stats summary
// ---------------------------------------------------------------------------

/// A snapshot summary of current network statistics.
#[derive(Debug, Clone)]
pub struct NetworkStatsSummary {
    /// Average RTT in milliseconds.
    pub avg_rtt_ms: f64,
    /// Median RTT in milliseconds.
    pub median_rtt_ms: f64,
    /// P95 RTT in milliseconds.
    pub p95_rtt_ms: f64,
    /// Current jitter in milliseconds.
    pub jitter_ms: f64,
    /// Packet loss rate (0..1).
    pub packet_loss: f64,
    /// Bandwidth in (bytes/sec).
    pub bandwidth_in_bps: f64,
    /// Bandwidth out (bytes/sec).
    pub bandwidth_out_bps: f64,
    /// Total packets sent.
    pub total_packets_sent: u64,
    /// Total packets received.
    pub total_packets_received: u64,
    /// Total packets lost.
    pub total_packets_lost: u64,
    /// Total bytes sent.
    pub total_bytes_sent: u64,
    /// Total bytes received.
    pub total_bytes_received: u64,
    /// Current network quality.
    pub quality: NetworkQuality,
    /// Connection uptime in seconds.
    pub uptime_secs: f64,
}

impl fmt::Display for NetworkStatsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Net[{quality}] RTT={rtt:.1}ms (p95={p95:.1}ms) jitter={jitter:.1}ms \
             loss={loss:.2}% BW in={bw_in:.1}KB/s out={bw_out:.1}KB/s",
            quality = self.quality,
            rtt = self.avg_rtt_ms,
            p95 = self.p95_rtt_ms,
            jitter = self.jitter_ms,
            loss = self.packet_loss * 100.0,
            bw_in = self.bandwidth_in_bps / 1024.0,
            bw_out = self.bandwidth_out_bps / 1024.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Periodic snapshot
// ---------------------------------------------------------------------------

/// A timestamped snapshot of network statistics for historical tracking.
#[derive(Debug, Clone)]
pub struct StatsSnapshot {
    /// Time offset from connection start (seconds).
    pub timestamp: f64,
    /// RTT at this point.
    pub rtt_ms: f64,
    /// Jitter at this point.
    pub jitter_ms: f64,
    /// Packet loss rate at this point.
    pub loss_rate: f64,
    /// Bandwidth in (bytes/sec).
    pub bandwidth_in: f64,
    /// Bandwidth out (bytes/sec).
    pub bandwidth_out: f64,
    /// Quality at this point.
    pub quality: NetworkQuality,
}

// ---------------------------------------------------------------------------
// Graph data helpers
// ---------------------------------------------------------------------------

/// Data formatted for graph rendering.
#[derive(Debug, Clone)]
pub struct GraphData {
    /// The data points (normalized 0..1).
    pub points: Vec<f64>,
    /// Label for the Y axis.
    pub y_label: String,
    /// Current value (for display).
    pub current_value: f64,
    /// Maximum value in the data set.
    pub max_value: f64,
    /// Minimum value in the data set.
    pub min_value: f64,
    /// Average value.
    pub avg_value: f64,
    /// Color for the graph line (RGBA).
    pub color: [f32; 4],
}

impl GraphData {
    /// Create a graph data set from rolling samples.
    pub fn from_samples(
        samples: &RollingSamples,
        label: &str,
        color: [f32; 4],
    ) -> Self {
        Self {
            points: samples.normalized(),
            y_label: label.to_string(),
            current_value: samples.latest(),
            max_value: samples.max(),
            min_value: samples.min(),
            avg_value: samples.average(),
            color,
        }
    }
}

// ---------------------------------------------------------------------------
// Network Stats Collector
// ---------------------------------------------------------------------------

/// The main network statistics collector.
pub struct NetworkStatsCollector {
    // --- Packet counts ---
    /// Total packets sent.
    pub total_packets_sent: u64,
    /// Total packets received.
    pub total_packets_received: u64,
    /// Total packets lost.
    pub total_packets_lost: u64,
    /// Total packets out of order.
    pub total_packets_ooo: u64,

    // --- Byte counts ---
    /// Total bytes sent.
    pub total_bytes_sent: u64,
    /// Total bytes received.
    pub total_bytes_received: u64,

    // --- RTT ---
    /// RTT histogram.
    rtt_histogram: RttHistogram,
    /// Rolling RTT samples for the graph.
    rtt_samples: RollingSamples,
    /// Exponentially smoothed RTT.
    smoothed_rtt: f64,

    // --- Jitter ---
    /// Rolling jitter samples.
    jitter_samples: RollingSamples,
    /// Last RTT value (for jitter calculation).
    last_rtt: f64,
    /// Exponentially smoothed jitter.
    smoothed_jitter: f64,

    // --- Bandwidth ---
    /// Rolling bandwidth-in samples (bytes/sec).
    bandwidth_in_samples: RollingSamples,
    /// Rolling bandwidth-out samples (bytes/sec).
    bandwidth_out_samples: RollingSamples,
    /// Bytes received in the current measurement interval.
    bytes_in_interval: u64,
    /// Bytes sent in the current measurement interval.
    bytes_out_interval: u64,
    /// Packets sent in the current measurement interval.
    packets_sent_interval: u64,
    /// Packets received in the current measurement interval.
    packets_recv_interval: u64,
    /// Timer for bandwidth measurement intervals.
    bandwidth_timer: f64,

    // --- Packet loss tracking ---
    /// Sliding window of packet delivery results (true = received, false = lost).
    loss_window: Vec<bool>,
    /// Write position in the loss window.
    loss_write_pos: usize,
    /// Rolling packet loss rate samples.
    loss_samples: RollingSamples,

    // --- Per-channel stats ---
    /// Statistics per named channel.
    channels: HashMap<String, ChannelStats>,

    // --- Snapshots ---
    /// Periodic snapshots for historical tracking.
    snapshots: Vec<StatsSnapshot>,
    /// Interval between snapshots (seconds).
    snapshot_interval: f64,
    /// Timer for snapshot collection.
    snapshot_timer: f64,

    // --- Timing ---
    /// Total uptime in seconds.
    uptime: f64,
    /// When the collector was created.
    created_at: Instant,
}

impl NetworkStatsCollector {
    /// Create a new statistics collector.
    pub fn new() -> Self {
        Self {
            total_packets_sent: 0,
            total_packets_received: 0,
            total_packets_lost: 0,
            total_packets_ooo: 0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
            rtt_histogram: RttHistogram::new(),
            rtt_samples: RollingSamples::new(ROLLING_WINDOW_SIZE),
            smoothed_rtt: 0.0,
            jitter_samples: RollingSamples::new(JITTER_GRAPH_POINTS),
            last_rtt: 0.0,
            smoothed_jitter: 0.0,
            bandwidth_in_samples: RollingSamples::new(BANDWIDTH_GRAPH_POINTS),
            bandwidth_out_samples: RollingSamples::new(BANDWIDTH_GRAPH_POINTS),
            bytes_in_interval: 0,
            bytes_out_interval: 0,
            packets_sent_interval: 0,
            packets_recv_interval: 0,
            bandwidth_timer: 0.0,
            loss_window: vec![true; PACKET_LOSS_WINDOW],
            loss_write_pos: 0,
            loss_samples: RollingSamples::new(ROLLING_WINDOW_SIZE),
            channels: HashMap::new(),
            snapshots: Vec::new(),
            snapshot_interval: 1.0,
            snapshot_timer: 0.0,
            uptime: 0.0,
            created_at: Instant::now(),
        }
    }

    /// Create a collector with a custom snapshot interval.
    pub fn with_snapshot_interval(mut self, interval_secs: f64) -> Self {
        self.snapshot_interval = interval_secs;
        self
    }

    // --- Recording events ---

    /// Record a packet being sent.
    pub fn record_packet_sent(&mut self, bytes: usize) {
        self.total_packets_sent += 1;
        self.total_bytes_sent += bytes as u64;
        self.bytes_out_interval += bytes as u64;
        self.packets_sent_interval += 1;
    }

    /// Record a packet being sent on a specific channel.
    pub fn record_packet_sent_channel(&mut self, channel: &str, bytes: usize) {
        self.record_packet_sent(bytes);
        self.get_or_create_channel(channel).record_sent(bytes);
    }

    /// Record a packet being received with its round-trip time.
    pub fn record_packet_received(&mut self, bytes: usize, rtt_ms: f64) {
        self.total_packets_received += 1;
        self.total_bytes_received += bytes as u64;
        self.bytes_in_interval += bytes as u64;
        self.packets_recv_interval += 1;

        // RTT tracking.
        self.rtt_histogram.record(rtt_ms);
        self.rtt_samples.push(rtt_ms);
        self.smoothed_rtt = self.smoothed_rtt * (1.0 - EMA_ALPHA) + rtt_ms * EMA_ALPHA;

        // Jitter calculation (RFC 3550 style).
        if self.last_rtt > 0.0 {
            let jitter = (rtt_ms - self.last_rtt).abs();
            self.jitter_samples.push(jitter);
            self.smoothed_jitter =
                self.smoothed_jitter * (1.0 - EMA_ALPHA / 4.0) + jitter * (EMA_ALPHA / 4.0);
        }
        self.last_rtt = rtt_ms;

        // Record delivery in loss window.
        self.loss_window[self.loss_write_pos] = true;
        self.loss_write_pos = (self.loss_write_pos + 1) % PACKET_LOSS_WINDOW;
    }

    /// Record a packet being received on a specific channel.
    pub fn record_packet_received_channel(&mut self, channel: &str, bytes: usize, rtt_ms: f64) {
        self.record_packet_received(bytes, rtt_ms);
        self.get_or_create_channel(channel).record_received(bytes);
    }

    /// Record a packet loss event.
    pub fn record_packet_lost(&mut self) {
        self.total_packets_lost += 1;
        self.loss_window[self.loss_write_pos] = false;
        self.loss_write_pos = (self.loss_write_pos + 1) % PACKET_LOSS_WINDOW;
    }

    /// Record a packet loss on a specific channel.
    pub fn record_packet_lost_channel(&mut self, channel: &str) {
        self.record_packet_lost();
        self.get_or_create_channel(channel).record_lost();
    }

    /// Record an out-of-order packet.
    pub fn record_out_of_order(&mut self) {
        self.total_packets_ooo += 1;
    }

    // --- Update (called each frame) ---

    /// Update statistics (call once per frame).
    pub fn update(&mut self, dt: f64) {
        self.uptime += dt;

        // Update bandwidth measurements.
        self.bandwidth_timer += dt;
        if self.bandwidth_timer >= 0.5 {
            let scale = 1.0 / self.bandwidth_timer;
            let bw_in = self.bytes_in_interval as f64 * scale;
            let bw_out = self.bytes_out_interval as f64 * scale;
            self.bandwidth_in_samples.push(bw_in);
            self.bandwidth_out_samples.push(bw_out);

            // Calculate packet loss for this interval.
            let loss_rate = self.current_loss_rate();
            self.loss_samples.push(loss_rate);

            self.bytes_in_interval = 0;
            self.bytes_out_interval = 0;
            self.packets_sent_interval = 0;
            self.packets_recv_interval = 0;
            self.bandwidth_timer = 0.0;
        }

        // Update per-channel stats.
        for channel in self.channels.values_mut() {
            channel.update(dt);
        }

        // Periodic snapshots.
        self.snapshot_timer += dt;
        if self.snapshot_timer >= self.snapshot_interval {
            self.snapshot_timer = 0.0;
            self.take_snapshot();
        }
    }

    /// Take a periodic snapshot of current statistics.
    fn take_snapshot(&mut self) {
        let snapshot = StatsSnapshot {
            timestamp: self.uptime,
            rtt_ms: self.smoothed_rtt,
            jitter_ms: self.smoothed_jitter,
            loss_rate: self.current_loss_rate(),
            bandwidth_in: self.bandwidth_in_samples.latest(),
            bandwidth_out: self.bandwidth_out_samples.latest(),
            quality: self.compute_quality(),
        };

        if self.snapshots.len() >= MAX_SNAPSHOTS {
            self.snapshots.remove(0);
        }
        self.snapshots.push(snapshot);
    }

    // --- Queries ---

    /// Compute the current network quality indicator.
    pub fn quality_indicator(&self) -> NetworkQuality {
        self.compute_quality()
    }

    fn compute_quality(&self) -> NetworkQuality {
        let rtt = self.smoothed_rtt;
        let loss = self.current_loss_rate();
        let jitter = self.smoothed_jitter;

        // Take the worst quality level from each metric.
        let rtt_quality = if rtt <= QUALITY_RTT_EXCELLENT {
            NetworkQuality::Excellent
        } else if rtt <= QUALITY_RTT_GOOD {
            NetworkQuality::Good
        } else if rtt <= QUALITY_RTT_FAIR {
            NetworkQuality::Fair
        } else if rtt <= QUALITY_RTT_POOR {
            NetworkQuality::Poor
        } else {
            NetworkQuality::Critical
        };

        let loss_quality = if loss <= QUALITY_LOSS_EXCELLENT {
            NetworkQuality::Excellent
        } else if loss <= QUALITY_LOSS_GOOD {
            NetworkQuality::Good
        } else if loss <= QUALITY_LOSS_FAIR {
            NetworkQuality::Fair
        } else if loss <= QUALITY_LOSS_POOR {
            NetworkQuality::Poor
        } else {
            NetworkQuality::Critical
        };

        let jitter_quality = if jitter <= QUALITY_JITTER_EXCELLENT {
            NetworkQuality::Excellent
        } else if jitter <= QUALITY_JITTER_GOOD {
            NetworkQuality::Good
        } else if jitter <= QUALITY_JITTER_FAIR {
            NetworkQuality::Fair
        } else if jitter <= QUALITY_JITTER_POOR {
            NetworkQuality::Poor
        } else {
            NetworkQuality::Critical
        };

        // Return the worst (highest ordinal) quality.
        [rtt_quality, loss_quality, jitter_quality]
            .into_iter()
            .max()
            .unwrap_or(NetworkQuality::Excellent)
    }

    /// Returns the current packet loss rate from the sliding window (0..1).
    pub fn current_loss_rate(&self) -> f64 {
        let lost = self.loss_window.iter().filter(|&&v| !v).count();
        lost as f64 / self.loss_window.len() as f64
    }

    /// Returns a summary of current network statistics.
    pub fn summary(&self) -> NetworkStatsSummary {
        NetworkStatsSummary {
            avg_rtt_ms: self.smoothed_rtt,
            median_rtt_ms: self.rtt_histogram.median(),
            p95_rtt_ms: self.rtt_histogram.p95(),
            jitter_ms: self.smoothed_jitter,
            packet_loss: self.current_loss_rate(),
            bandwidth_in_bps: self.bandwidth_in_samples.latest(),
            bandwidth_out_bps: self.bandwidth_out_samples.latest(),
            total_packets_sent: self.total_packets_sent,
            total_packets_received: self.total_packets_received,
            total_packets_lost: self.total_packets_lost,
            total_bytes_sent: self.total_bytes_sent,
            total_bytes_received: self.total_bytes_received,
            quality: self.compute_quality(),
            uptime_secs: self.uptime,
        }
    }

    /// Returns graph data for RTT visualization.
    pub fn rtt_graph(&self) -> GraphData {
        GraphData::from_samples(&self.rtt_samples, "RTT (ms)", [0.2, 0.8, 1.0, 1.0])
    }

    /// Returns graph data for jitter visualization.
    pub fn jitter_graph(&self) -> GraphData {
        GraphData::from_samples(
            &self.jitter_samples,
            "Jitter (ms)",
            [1.0, 0.5, 0.0, 1.0],
        )
    }

    /// Returns graph data for bandwidth-in visualization.
    pub fn bandwidth_in_graph(&self) -> GraphData {
        GraphData::from_samples(
            &self.bandwidth_in_samples,
            "BW In (B/s)",
            [0.0, 1.0, 0.5, 1.0],
        )
    }

    /// Returns graph data for bandwidth-out visualization.
    pub fn bandwidth_out_graph(&self) -> GraphData {
        GraphData::from_samples(
            &self.bandwidth_out_samples,
            "BW Out (B/s)",
            [0.5, 0.0, 1.0, 1.0],
        )
    }

    /// Returns graph data for packet loss visualization.
    pub fn loss_graph(&self) -> GraphData {
        GraphData::from_samples(
            &self.loss_samples,
            "Loss (%)",
            [1.0, 0.0, 0.0, 1.0],
        )
    }

    /// Returns the RTT histogram.
    pub fn rtt_histogram(&self) -> &RttHistogram {
        &self.rtt_histogram
    }

    /// Returns per-channel statistics.
    pub fn channel_stats(&self) -> &HashMap<String, ChannelStats> {
        &self.channels
    }

    /// Returns historical snapshots.
    pub fn snapshots(&self) -> &[StatsSnapshot] {
        &self.snapshots
    }

    /// Get or create a channel stats entry.
    fn get_or_create_channel(&mut self, name: &str) -> &mut ChannelStats {
        self.channels
            .entry(name.to_string())
            .or_insert_with(|| ChannelStats::new(name))
    }

    /// Returns the connection uptime.
    pub fn uptime(&self) -> f64 {
        self.uptime
    }

    /// Returns the smoothed RTT.
    pub fn smoothed_rtt(&self) -> f64 {
        self.smoothed_rtt
    }

    /// Returns the smoothed jitter.
    pub fn smoothed_jitter(&self) -> f64 {
        self.smoothed_jitter
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.total_packets_sent = 0;
        self.total_packets_received = 0;
        self.total_packets_lost = 0;
        self.total_packets_ooo = 0;
        self.total_bytes_sent = 0;
        self.total_bytes_received = 0;
        self.rtt_histogram.reset();
        self.rtt_samples.clear();
        self.smoothed_rtt = 0.0;
        self.jitter_samples.clear();
        self.last_rtt = 0.0;
        self.smoothed_jitter = 0.0;
        self.bandwidth_in_samples.clear();
        self.bandwidth_out_samples.clear();
        self.bytes_in_interval = 0;
        self.bytes_out_interval = 0;
        self.packets_sent_interval = 0;
        self.packets_recv_interval = 0;
        self.bandwidth_timer = 0.0;
        self.loss_window.iter_mut().for_each(|v| *v = true);
        self.loss_write_pos = 0;
        self.loss_samples.clear();
        self.channels.clear();
        self.snapshots.clear();
        self.snapshot_timer = 0.0;
        self.uptime = 0.0;
    }
}

impl Default for NetworkStatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for NetworkStatsCollector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let summary = self.summary();
        write!(f, "{summary}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_samples() {
        let mut samples = RollingSamples::new(5);
        assert!(samples.is_empty());

        samples.push(10.0);
        samples.push(20.0);
        samples.push(30.0);
        assert_eq!(samples.len(), 3);
        assert_eq!(samples.latest(), 30.0);
        assert_eq!(samples.average(), 20.0);
        assert_eq!(samples.min(), 10.0);
        assert_eq!(samples.max(), 30.0);
    }

    #[test]
    fn test_rolling_samples_overflow() {
        let mut samples = RollingSamples::new(3);
        samples.push(1.0);
        samples.push(2.0);
        samples.push(3.0);
        samples.push(4.0);
        assert_eq!(samples.len(), 3);
        assert_eq!(samples.latest(), 4.0);
        // Should contain [2, 3, 4] in order.
        let ordered = samples.as_slice_ordered();
        assert_eq!(ordered, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rtt_histogram() {
        let mut hist = RttHistogram::new();
        hist.record(10.0);
        hist.record(20.0);
        hist.record(30.0);
        assert_eq!(hist.total_samples, 3);
        assert_eq!(hist.mean(), 20.0);
        assert_eq!(hist.min_rtt, 10.0);
        assert_eq!(hist.max_rtt, 30.0);
    }

    #[test]
    fn test_rtt_histogram_percentiles() {
        let mut hist = RttHistogram::new();
        for i in 0..100 {
            hist.record(i as f64);
        }
        let p50 = hist.percentile(50.0);
        assert!(p50 > 40.0 && p50 < 60.0);
        let p95 = hist.p95();
        assert!(p95 > 85.0);
    }

    #[test]
    fn test_network_quality() {
        assert!(NetworkQuality::Excellent < NetworkQuality::Critical);
        assert_eq!(NetworkQuality::Excellent.score(), 4);
        assert_eq!(NetworkQuality::Critical.score(), 0);
    }

    #[test]
    fn test_stats_collector_basic() {
        let mut stats = NetworkStatsCollector::new();
        stats.record_packet_sent(128);
        stats.record_packet_received(256, 45.0);
        assert_eq!(stats.total_packets_sent, 1);
        assert_eq!(stats.total_packets_received, 1);
        assert_eq!(stats.total_bytes_sent, 128);
        assert_eq!(stats.total_bytes_received, 256);
    }

    #[test]
    fn test_stats_collector_quality() {
        let mut stats = NetworkStatsCollector::new();
        // Record many good RTT samples.
        for _ in 0..50 {
            stats.record_packet_received(100, 15.0);
        }
        stats.update(1.0);
        let quality = stats.quality_indicator();
        assert_eq!(quality, NetworkQuality::Excellent);
    }

    #[test]
    fn test_stats_collector_loss_rate() {
        let mut stats = NetworkStatsCollector::new();
        for _ in 0..90 {
            stats.record_packet_received(100, 20.0);
        }
        for _ in 0..10 {
            stats.record_packet_lost();
        }
        let loss = stats.current_loss_rate();
        assert!(loss > 0.09 && loss < 0.11);
    }

    #[test]
    fn test_channel_stats() {
        let mut stats = NetworkStatsCollector::new();
        stats.record_packet_sent_channel("reliable", 200);
        stats.record_packet_sent_channel("unreliable", 100);
        stats.record_packet_received_channel("reliable", 150, 30.0);
        assert_eq!(stats.channels.len(), 2);
        assert_eq!(stats.channels["reliable"].packets_sent, 1);
        assert_eq!(stats.channels["unreliable"].packets_sent, 1);
    }

    #[test]
    fn test_graph_data() {
        let mut samples = RollingSamples::new(10);
        for i in 0..5 {
            samples.push(i as f64 * 10.0);
        }
        let graph = GraphData::from_samples(&samples, "Test", [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(graph.points.len(), 5);
        assert_eq!(graph.current_value, 40.0);
        assert_eq!(graph.max_value, 40.0);
    }

    #[test]
    fn test_stats_summary() {
        let mut stats = NetworkStatsCollector::new();
        for i in 0..20 {
            stats.record_packet_sent(100);
            stats.record_packet_received(100, 20.0 + i as f64);
        }
        stats.update(1.0);
        let summary = stats.summary();
        assert!(summary.avg_rtt_ms > 0.0);
        assert_eq!(summary.total_packets_sent, 20);
        assert_eq!(summary.total_packets_received, 20);
    }

    #[test]
    fn test_stats_collector_jitter() {
        let mut stats = NetworkStatsCollector::new();
        stats.record_packet_received(100, 20.0);
        stats.record_packet_received(100, 25.0);
        stats.record_packet_received(100, 22.0);
        assert!(stats.smoothed_jitter > 0.0);
    }

    #[test]
    fn test_stats_snapshot() {
        let mut stats = NetworkStatsCollector::new();
        for _ in 0..10 {
            stats.record_packet_received(100, 30.0);
        }
        stats.update(1.5); // Should trigger a snapshot.
        assert!(!stats.snapshots.is_empty());
    }

    #[test]
    fn test_channel_loss_rate() {
        let mut channel = ChannelStats::new("test");
        channel.record_sent(100);
        channel.record_sent(100);
        channel.record_lost();
        assert!(channel.loss_rate() > 0.0);
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = NetworkStatsCollector::new();
        stats.record_packet_sent(100);
        stats.record_packet_received(200, 10.0);
        stats.reset();
        assert_eq!(stats.total_packets_sent, 0);
        assert_eq!(stats.total_packets_received, 0);
        assert_eq!(stats.total_bytes_sent, 0);
    }
}
