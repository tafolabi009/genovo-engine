// engine/networking/src/network_simulation.rs
//
// Network condition simulation for testing multiplayer without real network issues.
// Artificial latency, packet loss, jitter, bandwidth limit, out-of-order
// delivery, and duplicate packets.

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct NetworkSimConfig {
    pub enabled: bool,
    pub latency_ms: f64,
    pub latency_jitter_ms: f64,
    pub packet_loss_percent: f64,
    pub packet_duplicate_percent: f64,
    pub out_of_order_percent: f64,
    pub bandwidth_limit_bps: u64,
    pub bandwidth_burst_bytes: u64,
    pub corruption_percent: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub spike_probability: f64,
    pub spike_latency_ms: f64,
    pub spike_duration_ms: f64,
}

impl Default for NetworkSimConfig {
    fn default() -> Self {
        Self {
            enabled: false, latency_ms: 50.0, latency_jitter_ms: 10.0,
            packet_loss_percent: 0.0, packet_duplicate_percent: 0.0,
            out_of_order_percent: 0.0, bandwidth_limit_bps: 0,
            bandwidth_burst_bytes: 4096, corruption_percent: 0.0,
            min_latency_ms: 0.0, max_latency_ms: 500.0,
            spike_probability: 0.0, spike_latency_ms: 300.0, spike_duration_ms: 1000.0,
        }
    }
}

impl NetworkSimConfig {
    pub fn good_connection() -> Self { Self { enabled: true, latency_ms: 20.0, latency_jitter_ms: 5.0, packet_loss_percent: 0.1, ..Default::default() } }
    pub fn average_connection() -> Self { Self { enabled: true, latency_ms: 60.0, latency_jitter_ms: 15.0, packet_loss_percent: 1.0, ..Default::default() } }
    pub fn poor_connection() -> Self { Self { enabled: true, latency_ms: 150.0, latency_jitter_ms: 50.0, packet_loss_percent: 5.0, out_of_order_percent: 2.0, ..Default::default() } }
    pub fn terrible_connection() -> Self { Self { enabled: true, latency_ms: 300.0, latency_jitter_ms: 100.0, packet_loss_percent: 15.0, out_of_order_percent: 10.0, packet_duplicate_percent: 3.0, spike_probability: 0.05, ..Default::default() } }
    pub fn lan() -> Self { Self { enabled: true, latency_ms: 2.0, latency_jitter_ms: 1.0, ..Default::default() } }
}

#[derive(Debug, Clone)]
pub struct SimulatedPacket {
    pub data: Vec<u8>,
    pub delivery_time: f64,
    pub sequence: u64,
    pub is_duplicate: bool,
    pub corrupted: bool,
    pub original_size: usize,
}

#[derive(Debug)]
pub struct NetworkSimulator {
    pub config: NetworkSimConfig,
    pub outgoing_queue: VecDeque<SimulatedPacket>,
    pub incoming_queue: VecDeque<SimulatedPacket>,
    pub current_time: f64,
    pub rng_state: u64,
    pub bytes_sent_this_second: u64,
    pub second_timer: f64,
    pub in_spike: bool,
    pub spike_end_time: f64,
    pub stats: NetworkSimStats,
}

#[derive(Debug, Clone, Default)]
pub struct NetworkSimStats {
    pub packets_sent: u64,
    pub packets_received: u64,
    pub packets_dropped: u64,
    pub packets_duplicated: u64,
    pub packets_reordered: u64,
    pub packets_corrupted: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub avg_latency_ms: f64,
    pub current_latency_ms: f64,
    pub effective_loss_percent: f64,
}

impl NetworkSimulator {
    pub fn new(config: NetworkSimConfig) -> Self {
        Self {
            config, outgoing_queue: VecDeque::new(), incoming_queue: VecDeque::new(),
            current_time: 0.0, rng_state: 42, bytes_sent_this_second: 0,
            second_timer: 0.0, in_spike: false, spike_end_time: 0.0,
            stats: NetworkSimStats::default(),
        }
    }

    fn random_f64(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.rng_state >> 33) as f64 / (u32::MAX as f64)
    }

    fn compute_latency(&mut self) -> f64 {
        let base = if self.in_spike { self.config.spike_latency_ms } else { self.config.latency_ms };
        let jitter = (self.random_f64() * 2.0 - 1.0) * self.config.latency_jitter_ms;
        (base + jitter).clamp(self.config.min_latency_ms, self.config.max_latency_ms)
    }

    pub fn send(&mut self, data: Vec<u8>) {
        if !self.config.enabled {
            self.incoming_queue.push_back(SimulatedPacket {
                original_size: data.len(), data, delivery_time: self.current_time,
                sequence: self.stats.packets_sent, is_duplicate: false, corrupted: false,
            });
            self.stats.packets_sent += 1;
            return;
        }

        self.stats.packets_sent += 1;
        self.stats.bytes_sent += data.len() as u64;

        // Bandwidth limit
        if self.config.bandwidth_limit_bps > 0 {
            let limit_bytes = self.config.bandwidth_limit_bps / 8;
            if self.bytes_sent_this_second + data.len() as u64 > limit_bytes + self.config.bandwidth_burst_bytes {
                self.stats.packets_dropped += 1;
                return;
            }
            self.bytes_sent_this_second += data.len() as u64;
        }

        // Packet loss
        if self.random_f64() * 100.0 < self.config.packet_loss_percent {
            self.stats.packets_dropped += 1;
            return;
        }

        let latency = self.compute_latency();
        self.stats.current_latency_ms = latency;
        let delivery_time = self.current_time + latency / 1000.0;
        let seq = self.stats.packets_sent - 1;

        let mut packet = SimulatedPacket {
            original_size: data.len(), data: data.clone(),
            delivery_time, sequence: seq, is_duplicate: false, corrupted: false,
        };

        // Corruption
        if self.random_f64() * 100.0 < self.config.corruption_percent && !packet.data.is_empty() {
            let idx = (self.random_f64() * packet.data.len() as f64) as usize;
            if idx < packet.data.len() { packet.data[idx] ^= 0xFF; }
            packet.corrupted = true;
            self.stats.packets_corrupted += 1;
        }

        // Out-of-order
        if self.random_f64() * 100.0 < self.config.out_of_order_percent {
            packet.delivery_time += self.random_f64() * 0.05;
            self.stats.packets_reordered += 1;
        }

        self.outgoing_queue.push_back(packet);

        // Duplicate
        if self.random_f64() * 100.0 < self.config.packet_duplicate_percent {
            let dup_latency = self.compute_latency();
            let dup = SimulatedPacket {
                data, delivery_time: self.current_time + dup_latency / 1000.0,
                sequence: seq, is_duplicate: true, corrupted: false,
                original_size: 0,
            };
            self.outgoing_queue.push_back(dup);
            self.stats.packets_duplicated += 1;
        }
    }

    pub fn receive(&mut self) -> Vec<Vec<u8>> {
        let mut delivered = Vec::new();
        while let Some(front) = self.outgoing_queue.front() {
            if front.delivery_time <= self.current_time {
                let packet = self.outgoing_queue.pop_front().unwrap();
                self.stats.packets_received += 1;
                self.stats.bytes_received += packet.data.len() as u64;
                delivered.push(packet.data);
            } else { break; }
        }
        delivered
    }

    pub fn update(&mut self, dt: f64) {
        self.current_time += dt;
        self.second_timer += dt;
        if self.second_timer >= 1.0 { self.bytes_sent_this_second = 0; self.second_timer -= 1.0; }

        // Latency spike handling
        if self.in_spike && self.current_time >= self.spike_end_time { self.in_spike = false; }
        if !self.in_spike && self.config.spike_probability > 0.0 && self.random_f64() < self.config.spike_probability * dt {
            self.in_spike = true;
            self.spike_end_time = self.current_time + self.config.spike_duration_ms / 1000.0;
        }

        // Sort outgoing queue by delivery time for correct ordering
        let mut sorted: Vec<_> = self.outgoing_queue.drain(..).collect();
        sorted.sort_by(|a, b| a.delivery_time.partial_cmp(&b.delivery_time).unwrap_or(std::cmp::Ordering::Equal));
        self.outgoing_queue = sorted.into();

        // Update stats
        if self.stats.packets_sent > 0 {
            self.stats.effective_loss_percent = self.stats.packets_dropped as f64 / self.stats.packets_sent as f64 * 100.0;
        }
    }

    pub fn pending_count(&self) -> usize { self.outgoing_queue.len() }
    pub fn reset_stats(&mut self) { self.stats = NetworkSimStats::default(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_passthrough() {
        let mut sim = NetworkSimulator::new(NetworkSimConfig::default()); // disabled
        sim.send(vec![1, 2, 3]);
        let received = sim.receive();
        assert_eq!(received.len(), 1);
    }
    #[test]
    fn test_latency() {
        let mut sim = NetworkSimulator::new(NetworkSimConfig { enabled: true, latency_ms: 100.0, latency_jitter_ms: 0.0, ..Default::default() });
        sim.send(vec![1, 2, 3]);
        let r1 = sim.receive(); // Not yet delivered
        assert!(r1.is_empty());
        sim.update(0.15);
        let r2 = sim.receive();
        assert_eq!(r2.len(), 1);
    }
    #[test]
    fn test_presets() {
        let good = NetworkSimConfig::good_connection();
        assert!(good.enabled);
        assert!(good.latency_ms < 50.0);
        let terrible = NetworkSimConfig::terrible_connection();
        assert!(terrible.packet_loss_percent > 10.0);
    }
}
