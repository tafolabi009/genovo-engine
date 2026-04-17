// engine/networking/src/state_sync.rs
//
// State synchronization: snapshot interpolation, jitter buffer,
// clock synchronization, latency estimation, and adaptive quality.
// Implements a complete client-side state synchronization pipeline
// for multiplayer games.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32, pub y: f32, pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Self::new(a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t)
    }
    pub fn distance(self, other: Self) -> f32 {
        let dx = self.x-other.x; let dy = self.y-other.y; let dz = self.z-other.z;
        (dx*dx+dy*dy+dz*dz).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Clock synchronization
// ---------------------------------------------------------------------------

/// Synchronizes the local clock with the server's clock.
pub struct ClockSync {
    /// Estimated offset: server_time = local_time + offset.
    offset: f64,
    /// Estimated round-trip time.
    rtt: f64,
    /// Smoothed RTT (EWMA).
    smoothed_rtt: f64,
    /// RTT variance.
    rtt_variance: f64,
    /// Number of samples collected.
    sample_count: u32,
    /// Maximum samples to keep.
    max_samples: usize,
    /// Recent offset samples for averaging.
    offset_samples: VecDeque<f64>,
    /// Recent RTT samples.
    rtt_samples: VecDeque<f64>,
    /// Local time of last sync.
    last_sync_time: f64,
    /// Sync interval.
    sync_interval: f64,
    /// Whether the clock is synchronized.
    synchronized: bool,
}

impl ClockSync {
    pub fn new() -> Self {
        Self {
            offset: 0.0,
            rtt: 0.0,
            smoothed_rtt: 0.1,
            rtt_variance: 0.0,
            sample_count: 0,
            max_samples: 32,
            offset_samples: VecDeque::with_capacity(32),
            rtt_samples: VecDeque::with_capacity(32),
            last_sync_time: 0.0,
            sync_interval: 1.0,
            synchronized: false,
        }
    }

    /// Process a ping/pong response. `send_time` is when we sent the ping (local),
    /// `server_time` is the server's timestamp in the pong.
    pub fn process_pong(&mut self, send_time: f64, receive_time: f64, server_time: f64) {
        let rtt = receive_time - send_time;
        if rtt < 0.0 { return; }

        let one_way = rtt * 0.5;
        let estimated_offset = server_time - (send_time + one_way);

        // Add to samples.
        self.rtt_samples.push_back(rtt);
        self.offset_samples.push_back(estimated_offset);
        if self.rtt_samples.len() > self.max_samples {
            self.rtt_samples.pop_front();
        }
        if self.offset_samples.len() > self.max_samples {
            self.offset_samples.pop_front();
        }

        self.sample_count += 1;

        // Update RTT with EWMA.
        let alpha = 0.125;
        let beta = 0.25;
        let err = rtt - self.smoothed_rtt;
        self.smoothed_rtt += alpha * err;
        self.rtt_variance += beta * (err.abs() - self.rtt_variance);
        self.rtt = rtt;

        // Median filter for offset (reject outliers).
        if self.offset_samples.len() >= 3 {
            let mut sorted: Vec<f64> = self.offset_samples.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.offset = sorted[sorted.len() / 2]; // median
        } else {
            self.offset = estimated_offset;
        }

        self.last_sync_time = receive_time;
        self.synchronized = self.sample_count >= 3;
    }

    /// Convert local time to estimated server time.
    pub fn to_server_time(&self, local_time: f64) -> f64 {
        local_time + self.offset
    }

    /// Convert server time to local time.
    pub fn to_local_time(&self, server_time: f64) -> f64 {
        server_time - self.offset
    }

    /// Should we send another sync ping?
    pub fn needs_sync(&self, local_time: f64) -> bool {
        !self.synchronized || (local_time - self.last_sync_time) > self.sync_interval
    }

    pub fn offset(&self) -> f64 { self.offset }
    pub fn rtt(&self) -> f64 { self.rtt }
    pub fn smoothed_rtt(&self) -> f64 { self.smoothed_rtt }
    pub fn rtt_jitter(&self) -> f64 { self.rtt_variance }
    pub fn is_synchronized(&self) -> bool { self.synchronized }
    pub fn sample_count(&self) -> u32 { self.sample_count }
    pub fn one_way_latency(&self) -> f64 { self.smoothed_rtt * 0.5 }
}

impl Default for ClockSync {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Jitter buffer
// ---------------------------------------------------------------------------

/// A timestamped state snapshot.
#[derive(Debug, Clone)]
pub struct StateSnapshot<T: Clone> {
    pub server_tick: u64,
    pub server_time: f64,
    pub receive_time: f64,
    pub data: T,
}

/// A jitter buffer that holds snapshots and delivers them at a consistent rate.
pub struct JitterBuffer<T: Clone> {
    buffer: VecDeque<StateSnapshot<T>>,
    max_size: usize,
    /// How far behind real-time we render (in seconds).
    interpolation_delay: f64,
    /// Minimum number of snapshots before we start rendering.
    min_buffered: usize,
    /// Whether we have enough data to interpolate.
    ready: bool,
    /// Adaptive delay parameters.
    adaptive: bool,
    target_buffer_size: f32,
    dropped_count: u64,
    late_count: u64,
    out_of_order_count: u64,
}

impl<T: Clone> JitterBuffer<T> {
    pub fn new(max_size: usize, interpolation_delay: f64) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_size),
            max_size,
            interpolation_delay,
            min_buffered: 2,
            ready: false,
            adaptive: true,
            target_buffer_size: 3.0,
            dropped_count: 0,
            late_count: 0,
            out_of_order_count: 0,
        }
    }

    /// Insert a snapshot into the buffer.
    pub fn push(&mut self, snapshot: StateSnapshot<T>) {
        // Check for out-of-order.
        if let Some(last) = self.buffer.back() {
            if snapshot.server_tick <= last.server_tick {
                self.out_of_order_count += 1;
                // Insert in order.
                let pos = self.buffer.iter()
                    .position(|s| s.server_tick > snapshot.server_tick)
                    .unwrap_or(self.buffer.len());
                self.buffer.insert(pos, snapshot);
                return;
            }
        }

        self.buffer.push_back(snapshot);

        // Drop old snapshots if buffer is too large.
        while self.buffer.len() > self.max_size {
            self.buffer.pop_front();
            self.dropped_count += 1;
        }

        if self.buffer.len() >= self.min_buffered {
            self.ready = true;
        }
    }

    /// Get two snapshots for interpolation at the given render time.
    /// Returns (before, after, interpolation_factor) or None if not ready.
    pub fn sample(&self, render_time: f64) -> Option<(&StateSnapshot<T>, &StateSnapshot<T>, f64)> {
        if !self.ready || self.buffer.len() < 2 {
            return None;
        }

        let target_time = render_time - self.interpolation_delay;

        // Find the two snapshots bracketing the target time.
        let mut before_idx = 0;
        for (i, snap) in self.buffer.iter().enumerate() {
            if snap.server_time <= target_time {
                before_idx = i;
            } else {
                break;
            }
        }

        let after_idx = (before_idx + 1).min(self.buffer.len() - 1);
        if before_idx == after_idx {
            return None;
        }

        let before = &self.buffer[before_idx];
        let after = &self.buffer[after_idx];

        let dt = after.server_time - before.server_time;
        let t = if dt > 0.0 {
            ((target_time - before.server_time) / dt).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Some((before, after, t))
    }

    /// Remove snapshots older than the given server time.
    pub fn cleanup(&mut self, min_server_time: f64) {
        while self.buffer.len() > 2 {
            if self.buffer.front().map(|s| s.server_time < min_server_time).unwrap_or(false) {
                self.buffer.pop_front();
            } else {
                break;
            }
        }
    }

    /// Adapt the interpolation delay based on jitter.
    pub fn adapt_delay(&mut self, rtt_jitter: f64) {
        if !self.adaptive { return; }
        // Set delay to 2x jitter, clamped.
        let new_delay = (rtt_jitter * 2.0).clamp(0.05, 0.5);
        self.interpolation_delay = self.interpolation_delay * 0.9 + new_delay * 0.1;
    }

    pub fn is_ready(&self) -> bool { self.ready }
    pub fn len(&self) -> usize { self.buffer.len() }
    pub fn is_empty(&self) -> bool { self.buffer.is_empty() }
    pub fn interpolation_delay(&self) -> f64 { self.interpolation_delay }
    pub fn dropped_count(&self) -> u64 { self.dropped_count }
    pub fn out_of_order_count(&self) -> u64 { self.out_of_order_count }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.ready = false;
    }
}

// ---------------------------------------------------------------------------
// Snapshot interpolation
// ---------------------------------------------------------------------------

/// An entity state for synchronization.
#[derive(Debug, Clone)]
pub struct EntityState {
    pub entity_id: u64,
    pub position: Vec3,
    pub rotation: [f32; 4], // quaternion
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub custom_data: Vec<u8>,
}

/// Interpolate between two entity states.
pub fn interpolate_entity_state(a: &EntityState, b: &EntityState, t: f32) -> EntityState {
    EntityState {
        entity_id: a.entity_id,
        position: Vec3::lerp(a.position, b.position, t),
        rotation: slerp_quat(&a.rotation, &b.rotation, t),
        velocity: Vec3::lerp(a.velocity, b.velocity, t),
        angular_velocity: Vec3::lerp(a.angular_velocity, b.angular_velocity, t),
        custom_data: if t < 0.5 { a.custom_data.clone() } else { b.custom_data.clone() },
    }
}

fn slerp_quat(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
    let mut dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    let mut b = *b;
    if dot < 0.0 {
        dot = -dot;
        b = [-b[0], -b[1], -b[2], -b[3]];
    }
    if dot > 0.9995 {
        let result = [
            a[0] + (b[0]-a[0])*t,
            a[1] + (b[1]-a[1])*t,
            a[2] + (b[2]-a[2])*t,
            a[3] + (b[3]-a[3])*t,
        ];
        let len = (result[0]*result[0] + result[1]*result[1] + result[2]*result[2] + result[3]*result[3]).sqrt();
        if len > 0.0 {
            return [result[0]/len, result[1]/len, result[2]/len, result[3]/len];
        }
        return *a;
    }
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let wa = ((1.0-t)*theta).sin() / sin_theta;
    let wb = (t*theta).sin() / sin_theta;
    [
        a[0]*wa + b[0]*wb,
        a[1]*wa + b[1]*wb,
        a[2]*wa + b[2]*wb,
        a[3]*wa + b[3]*wb,
    ]
}

/// A world snapshot containing multiple entity states.
#[derive(Debug, Clone)]
pub struct WorldStateSnapshot {
    pub server_tick: u64,
    pub server_time: f64,
    pub entities: Vec<EntityState>,
}

/// Interpolate between two world snapshots.
pub fn interpolate_world(a: &WorldStateSnapshot, b: &WorldStateSnapshot, t: f32) -> WorldStateSnapshot {
    let mut entities = Vec::with_capacity(a.entities.len());
    for ea in &a.entities {
        if let Some(eb) = b.entities.iter().find(|e| e.entity_id == ea.entity_id) {
            entities.push(interpolate_entity_state(ea, eb, t));
        } else {
            entities.push(ea.clone());
        }
    }
    // Add entities only in b.
    for eb in &b.entities {
        if !a.entities.iter().any(|e| e.entity_id == eb.entity_id) {
            if t > 0.5 {
                entities.push(eb.clone());
            }
        }
    }
    WorldStateSnapshot {
        server_tick: if t < 0.5 { a.server_tick } else { b.server_tick },
        server_time: a.server_time + (b.server_time - a.server_time) * t as f64,
        entities,
    }
}

// ---------------------------------------------------------------------------
// Latency estimator
// ---------------------------------------------------------------------------

/// Estimates network latency with statistical analysis.
pub struct LatencyEstimator {
    samples: VecDeque<f64>,
    max_samples: usize,
    min_rtt: f64,
    max_rtt: f64,
    avg_rtt: f64,
    jitter: f64,
    percentile_95: f64,
}

impl LatencyEstimator {
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            min_rtt: f64::MAX,
            max_rtt: 0.0,
            avg_rtt: 0.0,
            jitter: 0.0,
            percentile_95: 0.0,
        }
    }

    pub fn add_sample(&mut self, rtt: f64) {
        self.samples.push_back(rtt);
        if self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
        self.recalculate();
    }

    fn recalculate(&mut self) {
        if self.samples.is_empty() { return; }
        let n = self.samples.len();
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        self.min_rtt = sorted[0];
        self.max_rtt = sorted[n - 1];
        self.avg_rtt = sorted.iter().sum::<f64>() / n as f64;

        // Jitter = average absolute difference between consecutive samples.
        if n > 1 {
            let mut jitter_sum = 0.0;
            for i in 1..self.samples.len() {
                jitter_sum += (self.samples[i] - self.samples[i-1]).abs();
            }
            self.jitter = jitter_sum / (n - 1) as f64;
        }

        // 95th percentile.
        let p95_idx = ((n as f64 * 0.95) as usize).min(n - 1);
        self.percentile_95 = sorted[p95_idx];
    }

    pub fn min_rtt(&self) -> f64 { self.min_rtt }
    pub fn max_rtt(&self) -> f64 { self.max_rtt }
    pub fn avg_rtt(&self) -> f64 { self.avg_rtt }
    pub fn jitter(&self) -> f64 { self.jitter }
    pub fn percentile_95(&self) -> f64 { self.percentile_95 }
    pub fn one_way_latency(&self) -> f64 { self.avg_rtt * 0.5 }
    pub fn sample_count(&self) -> usize { self.samples.len() }
}

impl Default for LatencyEstimator {
    fn default() -> Self { Self::new(64) }
}

// ---------------------------------------------------------------------------
// Adaptive quality
// ---------------------------------------------------------------------------

/// Network quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NetworkQuality {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Adaptive quality controller based on network conditions.
pub struct AdaptiveQuality {
    quality: NetworkQuality,
    send_rate_hz: f32,
    interpolation_delay_ms: f32,
    snapshot_compression: bool,
    delta_encoding: bool,
    priority_culling: bool,
    min_send_rate: f32,
    max_send_rate: f32,
    quality_history: VecDeque<NetworkQuality>,
}

impl AdaptiveQuality {
    pub fn new() -> Self {
        Self {
            quality: NetworkQuality::Good,
            send_rate_hz: 20.0,
            interpolation_delay_ms: 100.0,
            snapshot_compression: true,
            delta_encoding: true,
            priority_culling: false,
            min_send_rate: 5.0,
            max_send_rate: 60.0,
            quality_history: VecDeque::with_capacity(30),
        }
    }

    /// Update quality based on network conditions.
    pub fn update(&mut self, rtt_ms: f64, jitter_ms: f64, packet_loss: f32) {
        let new_quality = if rtt_ms < 50.0 && jitter_ms < 10.0 && packet_loss < 0.01 {
            NetworkQuality::Excellent
        } else if rtt_ms < 100.0 && jitter_ms < 30.0 && packet_loss < 0.03 {
            NetworkQuality::Good
        } else if rtt_ms < 200.0 && jitter_ms < 50.0 && packet_loss < 0.05 {
            NetworkQuality::Fair
        } else if rtt_ms < 400.0 && packet_loss < 0.10 {
            NetworkQuality::Poor
        } else {
            NetworkQuality::Critical
        };

        self.quality_history.push_back(new_quality);
        if self.quality_history.len() > 30 {
            self.quality_history.pop_front();
        }

        // Use most common recent quality to avoid flapping.
        let mut counts = [0u32; 5];
        for &q in &self.quality_history {
            counts[q as usize] += 1;
        }
        let most_common = counts.iter().enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(1);
        self.quality = match most_common {
            0 => NetworkQuality::Excellent,
            1 => NetworkQuality::Good,
            2 => NetworkQuality::Fair,
            3 => NetworkQuality::Poor,
            _ => NetworkQuality::Critical,
        };

        // Adjust parameters.
        match self.quality {
            NetworkQuality::Excellent => {
                self.send_rate_hz = self.max_send_rate;
                self.interpolation_delay_ms = 50.0;
                self.snapshot_compression = false;
                self.priority_culling = false;
            }
            NetworkQuality::Good => {
                self.send_rate_hz = 20.0;
                self.interpolation_delay_ms = 100.0;
                self.snapshot_compression = true;
                self.priority_culling = false;
            }
            NetworkQuality::Fair => {
                self.send_rate_hz = 15.0;
                self.interpolation_delay_ms = 150.0;
                self.snapshot_compression = true;
                self.priority_culling = true;
            }
            NetworkQuality::Poor => {
                self.send_rate_hz = 10.0;
                self.interpolation_delay_ms = 200.0;
                self.snapshot_compression = true;
                self.delta_encoding = true;
                self.priority_culling = true;
            }
            NetworkQuality::Critical => {
                self.send_rate_hz = self.min_send_rate;
                self.interpolation_delay_ms = 300.0;
                self.snapshot_compression = true;
                self.delta_encoding = true;
                self.priority_culling = true;
            }
        }
    }

    pub fn quality(&self) -> NetworkQuality { self.quality }
    pub fn send_rate_hz(&self) -> f32 { self.send_rate_hz }
    pub fn interpolation_delay_ms(&self) -> f32 { self.interpolation_delay_ms }
    pub fn should_compress(&self) -> bool { self.snapshot_compression }
    pub fn should_delta_encode(&self) -> bool { self.delta_encoding }
    pub fn should_priority_cull(&self) -> bool { self.priority_culling }
}

impl Default for AdaptiveQuality {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_sync() {
        let mut cs = ClockSync::new();
        // Simulate: local=1.0, server=1.05, receive=1.1 (RTT=0.1).
        cs.process_pong(1.0, 1.1, 1.05);
        cs.process_pong(1.2, 1.3, 1.25);
        cs.process_pong(1.4, 1.5, 1.45);
        assert!(cs.is_synchronized());
        assert!(cs.rtt() > 0.0);
    }

    #[test]
    fn test_jitter_buffer() {
        let mut jb = JitterBuffer::new(32, 0.1);

        for i in 0..5 {
            jb.push(StateSnapshot {
                server_tick: i,
                server_time: i as f64 * 0.05,
                receive_time: i as f64 * 0.05 + 0.01,
                data: i as f32,
            });
        }

        assert!(jb.is_ready());
        let result = jb.sample(0.2);
        assert!(result.is_some());
    }

    #[test]
    fn test_entity_interpolation() {
        let a = EntityState {
            entity_id: 1,
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: [0.0, 0.0, 0.0, 1.0],
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            custom_data: Vec::new(),
        };
        let b = EntityState {
            entity_id: 1,
            position: Vec3::new(10.0, 0.0, 0.0),
            rotation: [0.0, 0.0, 0.0, 1.0],
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            custom_data: Vec::new(),
        };

        let mid = interpolate_entity_state(&a, &b, 0.5);
        assert!((mid.position.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_latency_estimator() {
        let mut le = LatencyEstimator::new(32);
        for i in 0..20 {
            le.add_sample(0.05 + 0.01 * (i as f64 % 3.0));
        }
        assert!(le.avg_rtt() > 0.0);
        assert!(le.jitter() >= 0.0);
        assert!(le.percentile_95() >= le.avg_rtt());
    }

    #[test]
    fn test_adaptive_quality() {
        let mut aq = AdaptiveQuality::new();
        aq.update(30.0, 5.0, 0.0);
        assert_eq!(aq.quality(), NetworkQuality::Excellent);

        // Simulate poor conditions for enough frames.
        for _ in 0..35 {
            aq.update(350.0, 80.0, 0.08);
        }
        assert!(aq.quality() >= NetworkQuality::Poor);
    }
}
