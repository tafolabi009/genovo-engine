// engine/core/src/profiling_v2.rs
//
// Enhanced profiling system with GPU timestamp queries, nested scope tree,
// flame graph export, per-system averages, automatic hotspot alerts,
// and memory allocation tracking per scope.

use std::collections::HashMap;

pub const MAX_PROFILE_DEPTH: usize = 32;
pub const DEFAULT_HISTORY_FRAMES: usize = 120;
pub const HOTSPOT_THRESHOLD_MS: f64 = 8.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(pub u32);

#[derive(Debug, Clone)]
pub struct ProfileScope {
    pub id: ScopeId,
    pub name: String,
    pub parent: Option<ScopeId>,
    pub children: Vec<ScopeId>,
    pub depth: u32,
    pub cpu_start_us: f64,
    pub cpu_end_us: f64,
    pub gpu_start_us: f64,
    pub gpu_end_us: f64,
    pub cpu_duration_us: f64,
    pub gpu_duration_us: f64,
    pub call_count: u32,
    pub memory_allocated: u64,
    pub memory_freed: u64,
    pub color: u32,
}

impl ProfileScope {
    pub fn new(id: ScopeId, name: &str, parent: Option<ScopeId>, depth: u32) -> Self {
        Self { id, name: name.to_string(), parent, children: Vec::new(), depth, cpu_start_us: 0.0, cpu_end_us: 0.0, gpu_start_us: 0.0, gpu_end_us: 0.0, cpu_duration_us: 0.0, gpu_duration_us: 0.0, call_count: 0, memory_allocated: 0, memory_freed: 0, color: 0xFF00FF00 }
    }
    pub fn cpu_ms(&self) -> f64 { self.cpu_duration_us / 1000.0 }
    pub fn gpu_ms(&self) -> f64 { self.gpu_duration_us / 1000.0 }
    pub fn net_memory(&self) -> i64 { self.memory_allocated as i64 - self.memory_freed as i64 }
}

#[derive(Debug, Clone)]
pub struct FrameProfile {
    pub frame_index: u64,
    pub scopes: Vec<ProfileScope>,
    pub total_cpu_us: f64,
    pub total_gpu_us: f64,
    pub frame_time_ms: f64,
    pub total_allocations: u64,
    pub total_frees: u64,
    pub peak_memory: u64,
}

impl FrameProfile {
    pub fn new(frame_index: u64) -> Self {
        Self { frame_index, scopes: Vec::new(), total_cpu_us: 0.0, total_gpu_us: 0.0, frame_time_ms: 0.0, total_allocations: 0, total_frees: 0, peak_memory: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct ScopeAverage {
    pub name: String,
    pub avg_cpu_ms: f64,
    pub avg_gpu_ms: f64,
    pub min_cpu_ms: f64,
    pub max_cpu_ms: f64,
    pub avg_call_count: f32,
    pub avg_memory_bytes: f64,
    pub sample_count: u32,
}

impl ScopeAverage {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), avg_cpu_ms: 0.0, avg_gpu_ms: 0.0, min_cpu_ms: f64::MAX, max_cpu_ms: 0.0, avg_call_count: 0.0, avg_memory_bytes: 0.0, sample_count: 0 }
    }
    pub fn add_sample(&mut self, cpu_ms: f64, gpu_ms: f64, calls: u32, memory: i64) {
        self.sample_count += 1;
        let n = self.sample_count as f64;
        self.avg_cpu_ms = self.avg_cpu_ms * (n - 1.0) / n + cpu_ms / n;
        self.avg_gpu_ms = self.avg_gpu_ms * (n - 1.0) / n + gpu_ms / n;
        self.avg_call_count = self.avg_call_count * (n as f32 - 1.0) / n as f32 + calls as f32 / n as f32;
        self.avg_memory_bytes = self.avg_memory_bytes * (n - 1.0) / n + memory as f64 / n;
        if cpu_ms < self.min_cpu_ms { self.min_cpu_ms = cpu_ms; }
        if cpu_ms > self.max_cpu_ms { self.max_cpu_ms = cpu_ms; }
    }
}

#[derive(Debug, Clone)]
pub struct HotspotAlert {
    pub scope_name: String,
    pub cpu_ms: f64,
    pub threshold_ms: f64,
    pub frame_index: u64,
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity { Warning, Critical }

#[derive(Debug, Clone)]
pub struct FlameGraphEntry {
    pub name: String,
    pub start_us: f64,
    pub duration_us: f64,
    pub depth: u32,
    pub color: u32,
}

#[derive(Debug)]
pub struct ProfilerV2 {
    pub enabled: bool,
    pub current_frame: FrameProfile,
    pub history: Vec<FrameProfile>,
    pub max_history: usize,
    pub scope_averages: HashMap<String, ScopeAverage>,
    pub alerts: Vec<HotspotAlert>,
    pub hotspot_threshold_ms: f64,
    pub scope_stack: Vec<ScopeId>,
    next_scope_id: u32,
    pub frame_count: u64,
    pub gpu_profiling_enabled: bool,
    pub memory_tracking_enabled: bool,
    pub auto_alert_enabled: bool,
}

impl ProfilerV2 {
    pub fn new() -> Self {
        Self {
            enabled: true, current_frame: FrameProfile::new(0),
            history: Vec::new(), max_history: DEFAULT_HISTORY_FRAMES,
            scope_averages: HashMap::new(), alerts: Vec::new(),
            hotspot_threshold_ms: HOTSPOT_THRESHOLD_MS,
            scope_stack: Vec::new(), next_scope_id: 0, frame_count: 0,
            gpu_profiling_enabled: false, memory_tracking_enabled: true,
            auto_alert_enabled: true,
        }
    }

    pub fn begin_frame(&mut self) {
        if !self.enabled { return; }
        self.frame_count += 1;
        self.current_frame = FrameProfile::new(self.frame_count);
        self.scope_stack.clear();
        self.next_scope_id = 0;
    }

    pub fn begin_scope(&mut self, name: &str) -> ScopeId {
        let id = ScopeId(self.next_scope_id);
        self.next_scope_id += 1;
        let parent = self.scope_stack.last().copied();
        let depth = self.scope_stack.len() as u32;
        let scope = ProfileScope::new(id, name, parent, depth);
        self.current_frame.scopes.push(scope);
        self.scope_stack.push(id);
        id
    }

    pub fn end_scope(&mut self, duration_us: f64) {
        if let Some(id) = self.scope_stack.pop() {
            if let Some(scope) = self.current_frame.scopes.iter_mut().find(|s| s.id == id) {
                scope.cpu_duration_us = duration_us;
                scope.call_count += 1;
            }
        }
    }

    pub fn end_frame(&mut self, frame_time_ms: f64) {
        if !self.enabled { return; }
        self.current_frame.frame_time_ms = frame_time_ms;
        self.current_frame.total_cpu_us = self.current_frame.scopes.iter().filter(|s| s.depth == 0).map(|s| s.cpu_duration_us).sum();

        // Update averages
        for scope in &self.current_frame.scopes {
            let avg = self.scope_averages.entry(scope.name.clone()).or_insert_with(|| ScopeAverage::new(&scope.name));
            avg.add_sample(scope.cpu_ms(), scope.gpu_ms(), scope.call_count, scope.net_memory());
        }

        // Check hotspots
        if self.auto_alert_enabled {
            for scope in &self.current_frame.scopes {
                if scope.cpu_ms() > self.hotspot_threshold_ms {
                    let severity = if scope.cpu_ms() > self.hotspot_threshold_ms * 2.0 { AlertSeverity::Critical } else { AlertSeverity::Warning };
                    self.alerts.push(HotspotAlert { scope_name: scope.name.clone(), cpu_ms: scope.cpu_ms(), threshold_ms: self.hotspot_threshold_ms, frame_index: self.frame_count, severity });
                }
            }
        }

        // Store in history
        let frame = std::mem::replace(&mut self.current_frame, FrameProfile::new(0));
        self.history.push(frame);
        if self.history.len() > self.max_history { self.history.remove(0); }
    }

    pub fn export_flame_graph(&self) -> Vec<FlameGraphEntry> {
        if let Some(frame) = self.history.last() {
            frame.scopes.iter().map(|s| FlameGraphEntry {
                name: s.name.clone(), start_us: s.cpu_start_us,
                duration_us: s.cpu_duration_us, depth: s.depth, color: s.color,
            }).collect()
        } else { Vec::new() }
    }

    pub fn record_allocation(&mut self, scope_id: ScopeId, bytes: u64) {
        if !self.memory_tracking_enabled { return; }
        if let Some(scope) = self.current_frame.scopes.iter_mut().find(|s| s.id == scope_id) {
            scope.memory_allocated += bytes;
        }
        self.current_frame.total_allocations += bytes;
    }

    pub fn record_free(&mut self, scope_id: ScopeId, bytes: u64) {
        if !self.memory_tracking_enabled { return; }
        if let Some(scope) = self.current_frame.scopes.iter_mut().find(|s| s.id == scope_id) {
            scope.memory_freed += bytes;
        }
        self.current_frame.total_frees += bytes;
    }

    pub fn avg_frame_time_ms(&self) -> f64 {
        if self.history.is_empty() { return 0.0; }
        let sum: f64 = self.history.iter().map(|f| f.frame_time_ms).sum();
        sum / self.history.len() as f64
    }

    pub fn top_scopes_by_cpu(&self, count: usize) -> Vec<(&str, f64)> {
        let mut avgs: Vec<_> = self.scope_averages.iter().map(|(name, avg)| (name.as_str(), avg.avg_cpu_ms)).collect();
        avgs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        avgs.truncate(count);
        avgs
    }

    pub fn recent_alerts(&self, count: usize) -> &[HotspotAlert] {
        let start = self.alerts.len().saturating_sub(count);
        &self.alerts[start..]
    }

    pub fn clear_history(&mut self) { self.history.clear(); self.scope_averages.clear(); self.alerts.clear(); }
}

impl Default for ProfilerV2 { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_profiler_lifecycle() {
        let mut p = ProfilerV2::new();
        p.begin_frame();
        let id = p.begin_scope("test");
        p.end_scope(1000.0);
        p.end_frame(16.0);
        assert_eq!(p.history.len(), 1);
        assert!(p.scope_averages.contains_key("test"));
    }
    #[test]
    fn test_scope_average() {
        let mut avg = ScopeAverage::new("test");
        avg.add_sample(5.0, 0.0, 1, 100);
        avg.add_sample(10.0, 0.0, 1, 200);
        assert!((avg.avg_cpu_ms - 7.5).abs() < 0.01);
        assert_eq!(avg.sample_count, 2);
    }
}
