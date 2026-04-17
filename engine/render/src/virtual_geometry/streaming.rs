// engine/render/src/virtual_geometry/streaming.rs
//
// Virtual geometry streaming system. Manages loading and unloading of mesh
// clusters based on camera position, screen-space error, and a memory budget.
// Uses page-based GPU buffer management.

use super::cluster::{ClusterDAG, MeshCluster, ClusterBounds};
use glam::Vec3;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default GPU memory budget for virtual geometry (256 MB).
pub const DEFAULT_MEMORY_BUDGET: u64 = 256 * 1024 * 1024;

/// Number of clusters per page.
pub const CLUSTERS_PER_PAGE: usize = 32;

/// Maximum number of in-flight load requests.
pub const MAX_INFLIGHT_REQUESTS: usize = 64;

/// Number of frames a cluster must be unused before eviction.
pub const EVICTION_FRAME_DELAY: u64 = 30;

// ---------------------------------------------------------------------------
// StreamPriority
// ---------------------------------------------------------------------------

/// Priority for a streaming request. Higher values load first.
#[derive(Debug, Clone, Copy)]
pub struct StreamPriority {
    /// Screen-space error (higher error = higher priority to load).
    pub screen_error: f32,
    /// Distance from camera (closer = higher priority).
    pub distance: f32,
    /// Whether this cluster is needed to prevent popping.
    pub is_critical: bool,
}

impl StreamPriority {
    /// Compute a scalar priority value. Higher = more important.
    pub fn priority_value(&self) -> f32 {
        let base = self.screen_error / (self.distance + 1.0);
        if self.is_critical { base * 10.0 } else { base }
    }
}

impl PartialEq for StreamPriority {
    fn eq(&self, other: &Self) -> bool {
        self.priority_value().total_cmp(&other.priority_value()) == Ordering::Equal
    }
}

impl Eq for StreamPriority {}

impl PartialOrd for StreamPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StreamPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_value().total_cmp(&other.priority_value())
    }
}

// ---------------------------------------------------------------------------
// StreamRequest
// ---------------------------------------------------------------------------

/// A request to load or unload a cluster.
#[derive(Debug, Clone)]
pub struct StreamRequest {
    /// Cluster index in the DAG.
    pub cluster_index: u32,
    /// Priority of this request.
    pub priority: StreamPriority,
    /// Whether this is a load (true) or unload (false) request.
    pub is_load: bool,
    /// Page to load into (for load requests).
    pub target_page: Option<u32>,
    /// Byte size of the cluster data.
    pub byte_size: u64,
}

impl PartialEq for StreamRequest {
    fn eq(&self, other: &Self) -> bool {
        self.cluster_index == other.cluster_index
    }
}

impl Eq for StreamRequest {}

impl PartialOrd for StreamRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StreamRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

// ---------------------------------------------------------------------------
// PageAllocation
// ---------------------------------------------------------------------------

/// Represents a page in the GPU virtual geometry buffer.
#[derive(Debug, Clone)]
pub struct PageAllocation {
    /// Page index.
    pub page_id: u32,
    /// Byte offset in the GPU buffer.
    pub byte_offset: u64,
    /// Total byte capacity of this page.
    pub byte_capacity: u64,
    /// Current byte usage.
    pub byte_used: u64,
    /// Cluster indices stored in this page.
    pub cluster_indices: Vec<u32>,
    /// Last frame this page was accessed.
    pub last_access_frame: u64,
    /// Whether this page is currently allocated.
    pub is_allocated: bool,
}

impl PageAllocation {
    /// Remaining capacity in bytes.
    pub fn remaining_capacity(&self) -> u64 {
        self.byte_capacity.saturating_sub(self.byte_used)
    }

    /// Whether the page has room for a cluster of the given size.
    pub fn can_fit(&self, byte_size: u64) -> bool {
        self.byte_used + byte_size <= self.byte_capacity
    }

    /// Whether the page is empty (no clusters).
    pub fn is_empty(&self) -> bool {
        self.cluster_indices.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ResidentClusterInfo
// ---------------------------------------------------------------------------

/// Tracking information for a cluster that is currently resident on the GPU.
#[derive(Debug, Clone)]
struct ResidentClusterInfo {
    /// Page this cluster lives in.
    page_id: u32,
    /// Byte offset within the page.
    offset_in_page: u64,
    /// Byte size of the cluster data.
    byte_size: u64,
    /// Last frame this cluster was rendered.
    last_rendered_frame: u64,
    /// Screen-space error at last evaluation.
    last_screen_error: f32,
}

// ---------------------------------------------------------------------------
// VirtualGeometryStreaming
// ---------------------------------------------------------------------------

/// Manages streaming of virtual geometry clusters to and from GPU memory.
///
/// The streaming system maintains a memory budget and uses a priority queue
/// to decide which clusters to load or evict. Clusters are organised into
/// pages for efficient GPU buffer management.
pub struct VirtualGeometryStreaming {
    /// Memory budget in bytes.
    pub memory_budget: u64,
    /// Current memory usage in bytes.
    pub memory_used: u64,
    /// Page allocations.
    pages: Vec<PageAllocation>,
    /// Map from cluster index to resident info.
    resident_clusters: HashMap<u32, ResidentClusterInfo>,
    /// Pending load requests (priority queue).
    load_queue: BinaryHeap<StreamRequest>,
    /// Pending unload requests.
    unload_queue: VecDeque<u32>,
    /// Set of cluster indices currently being loaded (in-flight).
    inflight: HashSet<u32>,
    /// Current frame number.
    current_frame: u64,
    /// Per-page byte capacity.
    page_capacity: u64,
    /// Next page ID to allocate.
    next_page_id: u32,
    /// Free page IDs (recycled).
    free_pages: Vec<u32>,
    /// Viewport height for error computation.
    pub viewport_height: f32,
    /// Vertical FOV for error computation.
    pub fov_y: f32,
    /// Error threshold in pixels.
    pub error_threshold: f32,
}

impl VirtualGeometryStreaming {
    /// Create a new streaming manager with default settings.
    pub fn new(memory_budget: u64) -> Self {
        let page_capacity = (CLUSTERS_PER_PAGE * 128 * 3 * (48 + 4)) as u64;

        Self {
            memory_budget,
            memory_used: 0,
            pages: Vec::new(),
            resident_clusters: HashMap::new(),
            load_queue: BinaryHeap::new(),
            unload_queue: VecDeque::new(),
            inflight: HashSet::new(),
            current_frame: 0,
            page_capacity,
            next_page_id: 0,
            free_pages: Vec::new(),
            viewport_height: 1080.0,
            fov_y: std::f32::consts::FRAC_PI_4,
            error_threshold: 1.0,
        }
    }

    /// Update the streaming system. Evaluates which clusters should be loaded
    /// or unloaded based on camera position and screen-space error.
    ///
    /// Returns a tuple of (load_requests, unload_requests).
    pub fn update(
        &mut self,
        dag: &ClusterDAG,
        camera_pos: Vec3,
    ) -> (Vec<StreamRequest>, Vec<StreamRequest>) {
        self.current_frame += 1;
        self.load_queue.clear();

        // Determine which clusters are needed.
        let needed_clusters = dag.select_clusters(
            camera_pos,
            self.viewport_height,
            self.fov_y,
            self.error_threshold,
        );

        let needed_set: HashSet<u32> = needed_clusters.iter().copied().collect();

        // Queue loads for needed clusters that are not resident.
        for &cluster_idx in &needed_clusters {
            if self.resident_clusters.contains_key(&cluster_idx)
                || self.inflight.contains(&cluster_idx)
            {
                // Already resident or loading -- just update access time.
                if let Some(info) = self.resident_clusters.get_mut(&cluster_idx) {
                    info.last_rendered_frame = self.current_frame;
                }
                continue;
            }

            let cluster = &dag.clusters[cluster_idx as usize];
            let distance = (cluster.bounds.center - camera_pos).length();
            let screen_error = cluster.bounds.screen_space_error(
                cluster.error_metric,
                distance,
                self.viewport_height,
                self.fov_y,
            );

            self.load_queue.push(StreamRequest {
                cluster_index: cluster_idx,
                priority: StreamPriority {
                    screen_error,
                    distance,
                    is_critical: cluster.is_root(),
                },
                is_load: true,
                target_page: None,
                byte_size: cluster.gpu_byte_size() as u64,
            });
        }

        // Find clusters to evict (not needed and old enough).
        let mut eviction_candidates: Vec<(u32, f32)> = Vec::new();

        for (&cluster_idx, info) in &self.resident_clusters {
            if !needed_set.contains(&cluster_idx) {
                let frames_unused = self.current_frame.saturating_sub(info.last_rendered_frame);
                if frames_unused >= EVICTION_FRAME_DELAY {
                    eviction_candidates.push((cluster_idx, info.last_screen_error));
                }
            }
        }

        // Sort eviction candidates by lowest priority (lowest screen error first).
        eviction_candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        });

        // Process unloads to make room for loads.
        let mut unload_requests = Vec::new();
        let mut load_requests = Vec::new();

        // Evict if we are over budget or need room.
        while self.memory_used > self.memory_budget && !eviction_candidates.is_empty() {
            let (cluster_idx, _) = eviction_candidates.remove(0);
            if let Some(request) = self.create_unload_request(cluster_idx) {
                unload_requests.push(request);
            }
        }

        // Process load queue.
        let mut loads_this_frame = 0;
        while let Some(mut request) = self.load_queue.pop() {
            if loads_this_frame >= MAX_INFLIGHT_REQUESTS {
                break;
            }

            if self.resident_clusters.contains_key(&request.cluster_index) {
                continue;
            }

            // Check memory budget.
            if self.memory_used + request.byte_size > self.memory_budget {
                // Try to evict something.
                if let Some((evict_idx, _)) = eviction_candidates.first().copied() {
                    eviction_candidates.remove(0);
                    if let Some(unload) = self.create_unload_request(evict_idx) {
                        unload_requests.push(unload);
                    }
                } else {
                    // No more room -- skip this request.
                    continue;
                }
            }

            // Find or allocate a page.
            let page_id = self.find_or_allocate_page(request.byte_size);
            request.target_page = Some(page_id);

            // Register as in-flight.
            self.inflight.insert(request.cluster_index);
            self.memory_used += request.byte_size;

            load_requests.push(request);
            loads_this_frame += 1;
        }

        (load_requests, unload_requests)
    }

    /// Called when a load request has completed. Marks the cluster as resident.
    pub fn on_load_complete(&mut self, cluster_index: u32, page_id: u32, byte_size: u64) {
        self.inflight.remove(&cluster_index);

        self.resident_clusters.insert(cluster_index, ResidentClusterInfo {
            page_id,
            offset_in_page: 0,
            byte_size,
            last_rendered_frame: self.current_frame,
            last_screen_error: 0.0,
        });

        // Update page.
        if let Some(page) = self.pages.iter_mut().find(|p| p.page_id == page_id) {
            page.cluster_indices.push(cluster_index);
            page.byte_used += byte_size;
            page.last_access_frame = self.current_frame;
        }
    }

    /// Called when an unload has completed.
    pub fn on_unload_complete(&mut self, cluster_index: u32) {
        if let Some(info) = self.resident_clusters.remove(&cluster_index) {
            self.memory_used = self.memory_used.saturating_sub(info.byte_size);

            // Update page.
            if let Some(page) = self.pages.iter_mut().find(|p| p.page_id == info.page_id) {
                page.cluster_indices.retain(|&idx| idx != cluster_index);
                page.byte_used = page.byte_used.saturating_sub(info.byte_size);

                // If page is empty, recycle it.
                if page.is_empty() {
                    page.is_allocated = false;
                    self.free_pages.push(page.page_id);
                }
            }
        }
    }

    /// Check if a cluster is currently resident on the GPU.
    pub fn is_resident(&self, cluster_index: u32) -> bool {
        self.resident_clusters.contains_key(&cluster_index)
    }

    /// Get the number of resident clusters.
    pub fn resident_count(&self) -> usize {
        self.resident_clusters.len()
    }

    /// Get memory usage as a fraction of the budget.
    pub fn memory_utilization(&self) -> f32 {
        if self.memory_budget == 0 {
            return 0.0;
        }
        self.memory_used as f32 / self.memory_budget as f32
    }

    /// Get the number of allocated pages.
    pub fn allocated_page_count(&self) -> usize {
        self.pages.iter().filter(|p| p.is_allocated).count()
    }

    /// Get the number of in-flight load requests.
    pub fn inflight_count(&self) -> usize {
        self.inflight.len()
    }

    /// Reset the streaming state (e.g. on teleport).
    pub fn reset(&mut self) {
        self.resident_clusters.clear();
        self.inflight.clear();
        self.load_queue.clear();
        self.unload_queue.clear();
        self.memory_used = 0;

        for page in &mut self.pages {
            page.is_allocated = false;
            page.byte_used = 0;
            page.cluster_indices.clear();
        }

        self.free_pages = self.pages.iter().map(|p| p.page_id).collect();
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Create an unload request for a cluster.
    fn create_unload_request(&mut self, cluster_index: u32) -> Option<StreamRequest> {
        let info = self.resident_clusters.get(&cluster_index)?;

        Some(StreamRequest {
            cluster_index,
            priority: StreamPriority {
                screen_error: info.last_screen_error,
                distance: 0.0,
                is_critical: false,
            },
            is_load: false,
            target_page: Some(info.page_id),
            byte_size: info.byte_size,
        })
    }

    /// Find a page with enough room or allocate a new one.
    fn find_or_allocate_page(&mut self, byte_size: u64) -> u32 {
        // Try to find an existing page with room.
        for page in &self.pages {
            if page.is_allocated && page.can_fit(byte_size) {
                return page.page_id;
            }
        }

        // Allocate a new page.
        let page_id = if let Some(id) = self.free_pages.pop() {
            // Reuse a freed page.
            if let Some(page) = self.pages.iter_mut().find(|p| p.page_id == id) {
                page.is_allocated = true;
                page.byte_used = 0;
                page.cluster_indices.clear();
                page.last_access_frame = self.current_frame;
            }
            id
        } else {
            let id = self.next_page_id;
            self.next_page_id += 1;

            let offset = id as u64 * self.page_capacity;
            self.pages.push(PageAllocation {
                page_id: id,
                byte_offset: offset,
                byte_capacity: self.page_capacity,
                byte_used: 0,
                cluster_indices: Vec::new(),
                last_access_frame: self.current_frame,
                is_allocated: true,
            });
            id
        };

        page_id
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_priority_ordering() {
        let high = StreamPriority {
            screen_error: 10.0,
            distance: 1.0,
            is_critical: false,
        };
        let low = StreamPriority {
            screen_error: 1.0,
            distance: 10.0,
            is_critical: false,
        };
        assert!(high.priority_value() > low.priority_value());
    }

    #[test]
    fn test_critical_priority() {
        let normal = StreamPriority {
            screen_error: 5.0,
            distance: 5.0,
            is_critical: false,
        };
        let critical = StreamPriority {
            screen_error: 5.0,
            distance: 5.0,
            is_critical: true,
        };
        assert!(critical.priority_value() > normal.priority_value());
    }

    #[test]
    fn test_page_allocation() {
        let mut page = PageAllocation {
            page_id: 0,
            byte_offset: 0,
            byte_capacity: 1024,
            byte_used: 0,
            cluster_indices: Vec::new(),
            last_access_frame: 0,
            is_allocated: true,
        };

        assert!(page.can_fit(512));
        assert!(page.can_fit(1024));
        assert!(!page.can_fit(1025));

        page.byte_used = 512;
        assert!(page.can_fit(512));
        assert!(!page.can_fit(513));
        assert_eq!(page.remaining_capacity(), 512);
    }

    #[test]
    fn test_streaming_manager_creation() {
        let manager = VirtualGeometryStreaming::new(DEFAULT_MEMORY_BUDGET);
        assert_eq!(manager.memory_budget, DEFAULT_MEMORY_BUDGET);
        assert_eq!(manager.memory_used, 0);
        assert_eq!(manager.resident_count(), 0);
    }

    #[test]
    fn test_streaming_load_complete() {
        let mut manager = VirtualGeometryStreaming::new(DEFAULT_MEMORY_BUDGET);
        let page_id = manager.find_or_allocate_page(1024);
        manager.on_load_complete(0, page_id, 1024);

        assert!(manager.is_resident(0));
        assert_eq!(manager.resident_count(), 1);
        assert_eq!(manager.memory_used, 1024);
    }

    #[test]
    fn test_streaming_unload() {
        let mut manager = VirtualGeometryStreaming::new(DEFAULT_MEMORY_BUDGET);
        let page_id = manager.find_or_allocate_page(1024);
        manager.on_load_complete(42, page_id, 1024);
        assert!(manager.is_resident(42));

        manager.on_unload_complete(42);
        assert!(!manager.is_resident(42));
        assert_eq!(manager.memory_used, 0);
    }

    #[test]
    fn test_streaming_reset() {
        let mut manager = VirtualGeometryStreaming::new(DEFAULT_MEMORY_BUDGET);
        let page_id = manager.find_or_allocate_page(1024);
        manager.on_load_complete(0, page_id, 1024);
        manager.on_load_complete(1, page_id, 512);

        manager.reset();
        assert_eq!(manager.resident_count(), 0);
        assert_eq!(manager.memory_used, 0);
    }

    #[test]
    fn test_memory_utilization() {
        let mut manager = VirtualGeometryStreaming::new(1000);
        assert_eq!(manager.memory_utilization(), 0.0);

        manager.memory_used = 500;
        assert!((manager.memory_utilization() - 0.5).abs() < 0.001);
    }
}
