// engine/core/src/performance_counters.rs
// Performance tracking: CPU/GPU frame time, per-system timing, memory, allocation rate, budgets.
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy)]
pub struct FrameTiming { pub frame: u64, pub cpu_ms: f32, pub gpu_ms: f32, pub total_ms: f32, pub fps: f32, pub timestamp: f64 }

#[derive(Debug, Clone)]
pub struct SystemTiming { pub name: String, pub cpu_ms: f32, pub samples: VecDeque<f32>, pub avg_ms: f32, pub max_ms: f32, pub min_ms: f32, pub budget_ms: f32, pub over_budget: bool }

impl SystemTiming {
    pub fn new(name: &str, budget: f32) -> Self {
        Self { name: name.to_string(), cpu_ms: 0.0, samples: VecDeque::with_capacity(120), avg_ms: 0.0, max_ms: 0.0, min_ms: f32::MAX, budget_ms: budget, over_budget: false }
    }
    pub fn record(&mut self, ms: f32) {
        self.cpu_ms = ms;
        self.samples.push_back(ms);
        if self.samples.len() > 120 { self.samples.pop_front(); }
        self.avg_ms = self.samples.iter().sum::<f32>() / self.samples.len() as f32;
        self.max_ms = self.samples.iter().cloned().fold(0.0_f32, f32::max);
        self.min_ms = self.samples.iter().cloned().fold(f32::MAX, f32::min);
        self.over_budget = self.avg_ms > self.budget_ms;
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats { pub total_allocated_bytes: u64, pub peak_bytes: u64, pub allocation_count: u64, pub deallocation_count: u64, pub current_bytes: u64, pub frame_allocations: u32, pub frame_deallocations: u32, pub categories: HashMap<String, u64> }
impl Default for MemoryStats { fn default() -> Self { Self { total_allocated_bytes: 0, peak_bytes: 0, allocation_count: 0, deallocation_count: 0, current_bytes: 0, frame_allocations: 0, frame_deallocations: 0, categories: HashMap::new() } } }

#[derive(Debug, Clone)]
pub struct PerformanceBudget { pub name: String, pub budget_ms: f32, pub current_ms: f32, pub utilization: f32, pub is_over: bool }

pub struct PerformanceCounters {
    pub frame_history: VecDeque<FrameTiming>,
    pub system_timings: HashMap<String, SystemTiming>,
    pub memory: MemoryStats,
    pub budgets: Vec<PerformanceBudget>,
    pub frame_count: u64,
    pub total_time: f64,
    history_max: usize,
    pub target_fps: f32,
    pub target_frame_ms: f32,
    pub cpu_frame_ms: f32,
    pub gpu_frame_ms: f32,
    pub current_fps: f32,
    pub avg_fps: f32,
    pub one_percent_low_fps: f32,
}

impl PerformanceCounters {
    pub fn new(target_fps: f32) -> Self {
        Self {
            frame_history: VecDeque::with_capacity(300), system_timings: HashMap::new(),
            memory: MemoryStats::default(), budgets: Vec::new(), frame_count: 0, total_time: 0.0,
            history_max: 300, target_fps, target_frame_ms: 1000.0 / target_fps,
            cpu_frame_ms: 0.0, gpu_frame_ms: 0.0, current_fps: 0.0, avg_fps: 0.0, one_percent_low_fps: 0.0,
        }
    }

    pub fn begin_frame(&mut self) { self.memory.frame_allocations = 0; self.memory.frame_deallocations = 0; }

    pub fn end_frame(&mut self, cpu_ms: f32, gpu_ms: f32, timestamp: f64) {
        self.frame_count += 1;
        self.cpu_frame_ms = cpu_ms;
        self.gpu_frame_ms = gpu_ms;
        let total = cpu_ms.max(gpu_ms);
        self.current_fps = if total > 0.0 { 1000.0 / total } else { 0.0 };
        self.total_time = timestamp;
        let timing = FrameTiming { frame: self.frame_count, cpu_ms, gpu_ms, total_ms: total, fps: self.current_fps, timestamp };
        self.frame_history.push_back(timing);
        if self.frame_history.len() > self.history_max { self.frame_history.pop_front(); }
        // Compute averages
        if !self.frame_history.is_empty() {
            self.avg_fps = self.frame_history.iter().map(|f| f.fps).sum::<f32>() / self.frame_history.len() as f32;
            let mut fps_sorted: Vec<f32> = self.frame_history.iter().map(|f| f.fps).collect();
            fps_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let one_pct = (fps_sorted.len() as f32 * 0.01).ceil() as usize;
            self.one_percent_low_fps = if one_pct > 0 { fps_sorted[..one_pct].iter().sum::<f32>() / one_pct as f32 } else { 0.0 };
        }
        // Update budgets
        for budget in &mut self.budgets {
            if let Some(sys) = self.system_timings.get(&budget.name) {
                budget.current_ms = sys.avg_ms;
                budget.utilization = sys.avg_ms / budget.budget_ms.max(0.001);
                budget.is_over = budget.utilization > 1.0;
            }
        }
    }

    pub fn record_system(&mut self, name: &str, ms: f32) {
        self.system_timings.entry(name.to_string()).or_insert_with(|| SystemTiming::new(name, self.target_frame_ms * 0.2)).record(ms);
    }

    pub fn record_allocation(&mut self, bytes: u64, category: &str) {
        self.memory.total_allocated_bytes += bytes; self.memory.allocation_count += 1;
        self.memory.current_bytes += bytes; self.memory.frame_allocations += 1;
        self.memory.peak_bytes = self.memory.peak_bytes.max(self.memory.current_bytes);
        *self.memory.categories.entry(category.to_string()).or_insert(0) += bytes;
    }

    pub fn record_deallocation(&mut self, bytes: u64) {
        self.memory.current_bytes = self.memory.current_bytes.saturating_sub(bytes);
        self.memory.deallocation_count += 1; self.memory.frame_deallocations += 1;
    }

    pub fn add_budget(&mut self, name: &str, budget_ms: f32) {
        self.budgets.push(PerformanceBudget { name: name.to_string(), budget_ms, current_ms: 0.0, utilization: 0.0, is_over: false });
    }

    pub fn is_meeting_target(&self) -> bool { self.avg_fps >= self.target_fps * 0.95 }
    pub fn bottleneck(&self) -> &str { if self.cpu_frame_ms > self.gpu_frame_ms { "CPU" } else { "GPU" } }
    pub fn over_budget_systems(&self) -> Vec<&str> { self.system_timings.values().filter(|s| s.over_budget).map(|s| s.name.as_str()).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_perf_counters() {
        let mut pc = PerformanceCounters::new(60.0);
        pc.begin_frame();
        pc.record_system("Render", 8.0);
        pc.record_system("Physics", 3.0);
        pc.end_frame(12.0, 10.0, 0.016);
        assert!(pc.current_fps > 0.0);
    }
    #[test]
    fn test_memory_tracking() {
        let mut pc = PerformanceCounters::new(60.0);
        pc.record_allocation(1024, "textures");
        assert_eq!(pc.memory.current_bytes, 1024);
        pc.record_deallocation(512);
        assert_eq!(pc.memory.current_bytes, 512);
    }
}
