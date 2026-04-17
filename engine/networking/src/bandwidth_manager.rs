// engine/networking/src/bandwidth_manager.rs
//
// Bandwidth management for the Genovo engine networking layer.
// Per-channel bandwidth budgets, priority-based allocation,
// congestion detection, adaptive send rate, bandwidth smoothing,
// and burst allowance.

use std::collections::HashMap;

pub const DEFAULT_BANDWIDTH_BPS: u64 = 256_000; // 256 Kbps
pub const DEFAULT_BURST_ALLOWANCE: f64 = 1.5;
pub const CONGESTION_THRESHOLD: f64 = 0.9;
pub const MIN_SEND_RATE: f64 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChannelPriority { Low, Normal, High, Critical, Realtime }

impl ChannelPriority {
    pub fn weight(&self) -> f32 {
        match self { Self::Low => 0.5, Self::Normal => 1.0, Self::High => 2.0, Self::Critical => 4.0, Self::Realtime => 8.0 }
    }
}

#[derive(Debug, Clone)]
pub struct ChannelBudget {
    pub id: ChannelId,
    pub name: String,
    pub priority: ChannelPriority,
    pub max_bandwidth_bps: u64,
    pub allocated_bandwidth_bps: u64,
    pub used_bytes_this_interval: u64,
    pub queued_bytes: u64,
    pub packets_sent: u32,
    pub packets_dropped: u32,
    pub throttled: bool,
    pub guaranteed_minimum_bps: u64,
}

impl ChannelBudget {
    pub fn new(id: ChannelId, name: &str, priority: ChannelPriority) -> Self {
        Self { id, name: name.to_string(), priority, max_bandwidth_bps: DEFAULT_BANDWIDTH_BPS, allocated_bandwidth_bps: 0, used_bytes_this_interval: 0, queued_bytes: 0, packets_sent: 0, packets_dropped: 0, throttled: false, guaranteed_minimum_bps: 1000 }
    }
    pub fn usage_fraction(&self) -> f64 {
        if self.allocated_bandwidth_bps == 0 { return 0.0; }
        self.used_bytes_this_interval as f64 * 8.0 / self.allocated_bandwidth_bps as f64
    }
    pub fn can_send(&self, bytes: u64) -> bool {
        !self.throttled && (self.used_bytes_this_interval + bytes) * 8 <= self.allocated_bandwidth_bps
    }
    pub fn record_send(&mut self, bytes: u64) { self.used_bytes_this_interval += bytes; self.packets_sent += 1; }
    pub fn record_drop(&mut self) { self.packets_dropped += 1; }
    pub fn reset_interval(&mut self) { self.used_bytes_this_interval = 0; }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionState { None, Mild, Moderate, Severe }

#[derive(Debug)]
pub struct CongestionDetector {
    pub state: CongestionState,
    pub rtt_samples: Vec<f64>,
    pub packet_loss_rate: f64,
    pub rtt_threshold_ms: f64,
    pub loss_threshold: f64,
    pub window_size: usize,
    pub rtt_increase_ratio: f64,
    pub baseline_rtt_ms: f64,
}

impl CongestionDetector {
    pub fn new() -> Self {
        Self { state: CongestionState::None, rtt_samples: Vec::new(), packet_loss_rate: 0.0, rtt_threshold_ms: 200.0, loss_threshold: 0.05, window_size: 20, rtt_increase_ratio: 1.5, baseline_rtt_ms: 50.0 }
    }
    pub fn add_rtt_sample(&mut self, rtt_ms: f64) {
        self.rtt_samples.push(rtt_ms);
        if self.rtt_samples.len() > self.window_size { self.rtt_samples.remove(0); }
        self.update_state();
    }
    pub fn set_loss_rate(&mut self, rate: f64) {
        self.packet_loss_rate = rate;
        self.update_state();
    }
    fn avg_rtt(&self) -> f64 {
        if self.rtt_samples.is_empty() { return 0.0; }
        self.rtt_samples.iter().sum::<f64>() / self.rtt_samples.len() as f64
    }
    fn update_state(&mut self) {
        let avg = self.avg_rtt();
        let rtt_ratio = avg / self.baseline_rtt_ms.max(1.0);
        if self.packet_loss_rate > self.loss_threshold * 3.0 || rtt_ratio > 3.0 { self.state = CongestionState::Severe; }
        else if self.packet_loss_rate > self.loss_threshold * 2.0 || rtt_ratio > 2.0 { self.state = CongestionState::Moderate; }
        else if self.packet_loss_rate > self.loss_threshold || rtt_ratio > self.rtt_increase_ratio { self.state = CongestionState::Mild; }
        else { self.state = CongestionState::None; }
    }
}

impl Default for CongestionDetector { fn default() -> Self { Self::new() } }

#[derive(Debug)]
pub struct BandwidthSmoother {
    pub samples: Vec<f64>,
    pub window_size: usize,
    pub smoothed_value: f64,
}

impl BandwidthSmoother {
    pub fn new(window_size: usize) -> Self { Self { samples: Vec::new(), window_size, smoothed_value: 0.0 } }
    pub fn add_sample(&mut self, bps: f64) {
        self.samples.push(bps);
        if self.samples.len() > self.window_size { self.samples.remove(0); }
        self.smoothed_value = self.samples.iter().sum::<f64>() / self.samples.len() as f64;
    }
    pub fn value(&self) -> f64 { self.smoothed_value }
}

#[derive(Debug)]
pub struct BandwidthManager {
    pub channels: HashMap<ChannelId, ChannelBudget>,
    pub total_bandwidth_bps: u64,
    pub congestion: CongestionDetector,
    pub smoother: BandwidthSmoother,
    pub send_rate_multiplier: f64,
    pub burst_allowance: f64,
    pub interval_seconds: f64,
    pub interval_elapsed: f64,
    pub stats: BandwidthStats,
    next_channel_id: u32,
}

#[derive(Debug, Clone, Default)]
pub struct BandwidthStats {
    pub total_sent_bytes: u64,
    pub total_sent_packets: u32,
    pub total_dropped_packets: u32,
    pub current_usage_bps: u64,
    pub congestion_state: CongestionState,
    pub send_rate_multiplier: f64,
    pub channel_count: u32,
}

impl Default for CongestionState { fn default() -> Self { Self::None } }

impl BandwidthManager {
    pub fn new(total_bandwidth_bps: u64) -> Self {
        Self {
            channels: HashMap::new(), total_bandwidth_bps,
            congestion: CongestionDetector::new(), smoother: BandwidthSmoother::new(10),
            send_rate_multiplier: 1.0, burst_allowance: DEFAULT_BURST_ALLOWANCE,
            interval_seconds: 1.0, interval_elapsed: 0.0,
            stats: BandwidthStats::default(), next_channel_id: 0,
        }
    }

    pub fn add_channel(&mut self, name: &str, priority: ChannelPriority) -> ChannelId {
        let id = ChannelId(self.next_channel_id); self.next_channel_id += 1;
        self.channels.insert(id, ChannelBudget::new(id, name, priority));
        self.reallocate_budgets();
        id
    }

    pub fn remove_channel(&mut self, id: ChannelId) {
        self.channels.remove(&id);
        self.reallocate_budgets();
    }

    fn reallocate_budgets(&mut self) {
        let total_weight: f32 = self.channels.values().map(|c| c.priority.weight()).sum();
        if total_weight <= 0.0 { return; }
        let available = (self.total_bandwidth_bps as f64 * self.send_rate_multiplier) as u64;
        for channel in self.channels.values_mut() {
            let fraction = channel.priority.weight() / total_weight;
            channel.allocated_bandwidth_bps = ((available as f64 * fraction as f64) as u64).max(channel.guaranteed_minimum_bps);
        }
    }

    pub fn can_send(&self, channel_id: ChannelId, bytes: u64) -> bool {
        self.channels.get(&channel_id).map_or(false, |c| c.can_send(bytes))
    }

    pub fn record_send(&mut self, channel_id: ChannelId, bytes: u64) {
        if let Some(c) = self.channels.get_mut(&channel_id) { c.record_send(bytes); }
        self.stats.total_sent_bytes += bytes;
        self.stats.total_sent_packets += 1;
    }

    pub fn update(&mut self, dt: f64) {
        self.interval_elapsed += dt;
        if self.interval_elapsed >= self.interval_seconds {
            let total_used: u64 = self.channels.values().map(|c| c.used_bytes_this_interval * 8).sum();
            self.smoother.add_sample(total_used as f64);
            self.stats.current_usage_bps = total_used;

            // Adapt send rate based on congestion
            match self.congestion.state {
                CongestionState::None => { self.send_rate_multiplier = (self.send_rate_multiplier + 0.05).min(1.0); }
                CongestionState::Mild => { self.send_rate_multiplier *= 0.95; }
                CongestionState::Moderate => { self.send_rate_multiplier *= 0.8; }
                CongestionState::Severe => { self.send_rate_multiplier *= 0.5; }
            }
            self.send_rate_multiplier = self.send_rate_multiplier.max(MIN_SEND_RATE);

            for channel in self.channels.values_mut() {
                channel.throttled = channel.usage_fraction() > CONGESTION_THRESHOLD;
                channel.reset_interval();
            }
            self.reallocate_budgets();
            self.interval_elapsed = 0.0;
        }
        self.stats.congestion_state = self.congestion.state;
        self.stats.send_rate_multiplier = self.send_rate_multiplier;
        self.stats.channel_count = self.channels.len() as u32;
    }

    pub fn add_rtt_sample(&mut self, rtt_ms: f64) { self.congestion.add_rtt_sample(rtt_ms); }
    pub fn set_loss_rate(&mut self, rate: f64) { self.congestion.set_loss_rate(rate); }
    pub fn channel_count(&self) -> usize { self.channels.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bandwidth_manager() {
        let mut bm = BandwidthManager::new(100_000);
        let ch = bm.add_channel("game_state", ChannelPriority::High);
        assert!(bm.can_send(ch, 100));
        bm.record_send(ch, 100);
        assert_eq!(bm.stats.total_sent_packets, 1);
    }
    #[test]
    fn test_congestion_detection() {
        let mut cd = CongestionDetector::new();
        cd.baseline_rtt_ms = 50.0;
        for _ in 0..20 { cd.add_rtt_sample(200.0); }
        assert!(cd.state != CongestionState::None);
    }
}
