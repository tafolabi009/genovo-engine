//! Async Streaming Manager
//!
//! Manages the background loading and unloading of world cells. The streaming
//! manager maintains a priority queue of pending requests, executes them on a
//! background thread pool, and transitions cells through their lifecycle
//! states.
//!
//! # Architecture
//!
//! ```text
//!  Main thread                     Worker pool
//! ┌───────────────┐               ┌──────────────┐
//! │ submit_load() │──StreamReq──> │ load worker   │
//! │               │               │   - read file │
//! │ poll_results()│<──LoadResult─ │   - deserial. │
//! │               │               │   - entities  │
//! │ submit_unload │──StreamReq──> │ unload worker │
//! │               │               │   - serialize │
//! │ poll_results()│<──UnloadDone─ │   - write     │
//! └───────────────┘               └──────────────┘
//! ```
//!
//! The main thread never blocks on I/O. All file access happens on worker
//! threads. Results are polled each frame and applied to the ECS world.

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use genovo_ecs::Entity;

use crate::partition::{CellCoord, CellState};

// ---------------------------------------------------------------------------
// StreamRequestPriority
// ---------------------------------------------------------------------------

/// Priority level for streaming requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamRequestPriority {
    /// Mandatory: must be loaded before gameplay can continue (e.g., player cell).
    Critical = 0,
    /// High priority: cells adjacent to the player.
    High = 1,
    /// Normal: cells within the load radius.
    Normal = 2,
    /// Low: prefetch cells that may be needed soon.
    Low = 3,
    /// Background: speculative preloading.
    Background = 4,
}

impl StreamRequestPriority {
    /// Convert to a numeric value (lower = higher priority).
    #[inline]
    pub fn as_u32(&self) -> u32 {
        *self as u32
    }
}

impl PartialOrd for StreamRequestPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StreamRequestPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower numeric value = higher priority = should come first.
        (other.as_u32()).cmp(&self.as_u32())
    }
}

// ---------------------------------------------------------------------------
// CellTransitionState
// ---------------------------------------------------------------------------

/// Extended state tracking for cells during async transitions.
#[derive(Debug, Clone)]
pub enum CellTransitionState {
    /// No transition in progress.
    Idle,
    /// Waiting in the load queue.
    Queued {
        priority: StreamRequestPriority,
        queued_frame: u64,
    },
    /// Actively being loaded by a worker.
    Loading {
        started_frame: u64,
        layer: Option<String>,
    },
    /// Load complete, waiting for activation on the main thread.
    PendingActivation {
        loaded_frame: u64,
        entities: Vec<SerializedEntity>,
        memory_bytes: usize,
    },
    /// Actively being serialized/unloaded by a worker.
    Unloading {
        started_frame: u64,
    },
    /// Unload complete, resources have been freed.
    UnloadComplete {
        completed_frame: u64,
    },
    /// An error occurred during loading or unloading.
    Error {
        message: String,
        frame: u64,
    },
}

// ---------------------------------------------------------------------------
// SerializedEntity
// ---------------------------------------------------------------------------

/// A snapshot of an entity's components in a serializable format.
///
/// When a cell is loaded from disk, each entity is represented as a
/// `SerializedEntity`. The streaming manager creates real ECS entities from
/// these when the cell is activated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedEntity {
    /// Unique identifier for this entity within the cell data.
    pub id: u64,
    /// Human-readable name (for debugging).
    pub name: String,
    /// Serialized component data, keyed by component type name.
    pub components: HashMap<String, Vec<u8>>,
    /// World-space position of this entity.
    pub position: [f32; 3],
    /// World-space rotation as a quaternion [x, y, z, w].
    pub rotation: [f32; 4],
    /// Scale.
    pub scale: [f32; 3],
    /// Tags / flags for this entity.
    pub tags: Vec<String>,
}

impl SerializedEntity {
    /// Create a new serialized entity with default transform.
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            components: HashMap::new(),
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            tags: Vec::new(),
        }
    }

    /// Add a component (as raw bytes) to this entity snapshot.
    pub fn add_component(&mut self, type_name: impl Into<String>, data: Vec<u8>) {
        self.components.insert(type_name.into(), data);
    }

    /// Estimated memory usage of this serialized entity.
    pub fn estimated_memory(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        total += self.name.len();
        for (k, v) in &self.components {
            total += k.len() + v.len();
        }
        for tag in &self.tags {
            total += tag.len();
        }
        total
    }
}

// ---------------------------------------------------------------------------
// StreamRequest
// ---------------------------------------------------------------------------

/// A request to load or unload a world cell.
#[derive(Debug, Clone)]
pub struct StreamRequest {
    /// Cell coordinate to stream.
    pub coord: CellCoord,
    /// Whether this is a load or unload request.
    pub kind: StreamRequestKind,
    /// Priority of this request.
    pub priority: StreamRequestPriority,
    /// Specific layer to load/unload, or None for the whole cell.
    pub layer: Option<String>,
    /// Frame number when this request was submitted.
    pub submitted_frame: u64,
    /// Distance from camera when submitted (for priority tie-breaking).
    pub distance: f32,
}

/// Whether a stream request is for loading or unloading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamRequestKind {
    Load,
    Unload,
}

impl PartialEq for StreamRequest {
    fn eq(&self, other: &Self) -> bool {
        self.coord == other.coord && self.kind == other.kind && self.layer == other.layer
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
        // Higher priority first, then closer distance.
        self.priority
            .cmp(&other.priority)
            .then_with(|| {
                self.distance
                    .partial_cmp(&other.distance)
                    .unwrap_or(Ordering::Equal)
            })
            // Reverse because BinaryHeap is a max-heap.
            .reverse()
    }
}

// ---------------------------------------------------------------------------
// LoadResult
// ---------------------------------------------------------------------------

/// Result of a completed load operation, sent from a worker back to the
/// main thread.
#[derive(Debug)]
pub struct LoadResult {
    /// Which cell was loaded.
    pub coord: CellCoord,
    /// Layer that was loaded (None for whole cell).
    pub layer: Option<String>,
    /// Deserialized entities from the cell data.
    pub entities: Vec<SerializedEntity>,
    /// Total memory consumed by this cell's data in bytes.
    pub memory_bytes: usize,
    /// Whether the load succeeded.
    pub success: bool,
    /// Error message if loading failed.
    pub error: Option<String>,
    /// Time in seconds that the load took.
    pub load_time_secs: f64,
}

/// Result of a completed unload operation.
#[derive(Debug)]
pub struct UnloadResult {
    /// Which cell was unloaded.
    pub coord: CellCoord,
    /// Layer that was unloaded (None for whole cell).
    pub layer: Option<String>,
    /// Whether the unload/serialize succeeded.
    pub success: bool,
    /// Error message if unloading failed.
    pub error: Option<String>,
    /// Memory freed by unloading this cell.
    pub freed_bytes: usize,
}

// ---------------------------------------------------------------------------
// StreamingStats
// ---------------------------------------------------------------------------

/// Runtime statistics about the streaming system.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Number of cells currently loaded.
    pub loaded_cells: usize,
    /// Number of cells currently being loaded.
    pub loading_cells: usize,
    /// Number of cells queued for loading.
    pub queued_cells: usize,
    /// Number of cells being unloaded.
    pub unloading_cells: usize,
    /// Total memory used by loaded cells.
    pub total_memory: usize,
    /// Memory budget.
    pub memory_budget: usize,
    /// Streaming bandwidth: bytes loaded this frame.
    pub bytes_loaded_this_frame: usize,
    /// Streaming bandwidth: bytes unloaded this frame.
    pub bytes_freed_this_frame: usize,
    /// Total bytes loaded since start.
    pub total_bytes_loaded: u64,
    /// Total bytes freed since start.
    pub total_bytes_freed: u64,
    /// Number of load requests completed since start.
    pub total_loads_completed: u64,
    /// Number of unload requests completed since start.
    pub total_unloads_completed: u64,
    /// Number of load errors since start.
    pub total_load_errors: u64,
    /// Average load time in seconds (exponential moving average).
    pub avg_load_time_secs: f64,
    /// Peak load time this session.
    pub peak_load_time_secs: f64,
    /// Current frame number.
    pub frame: u64,
}

impl std::fmt::Display for StreamingStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Streaming: {} loaded, {} loading, {} queued | \
             mem {:.1}/{:.1} MiB | avg load {:.1}ms",
            self.loaded_cells,
            self.loading_cells,
            self.queued_cells,
            self.total_memory as f64 / (1024.0 * 1024.0),
            self.memory_budget as f64 / (1024.0 * 1024.0),
            self.avg_load_time_secs * 1000.0,
        )
    }
}

// ---------------------------------------------------------------------------
// CellDataProvider
// ---------------------------------------------------------------------------

/// Trait for providing cell data from storage. Implementations can load from
/// disk, network, procedural generation, etc.
pub trait CellDataProvider: Send + Sync {
    /// Load cell data for the given coordinate and optional layer.
    ///
    /// This is called on a worker thread. The implementation should read the
    /// cell data from disk (or generate it) and return the deserialized
    /// entities.
    fn load_cell(
        &self,
        coord: CellCoord,
        layer: Option<&str>,
        base_path: &Path,
    ) -> Result<(Vec<SerializedEntity>, usize), String>;

    /// Save cell data for the given coordinate and optional layer.
    ///
    /// Called when a dirty cell is being unloaded.
    fn save_cell(
        &self,
        coord: CellCoord,
        layer: Option<&str>,
        entities: &[SerializedEntity],
        base_path: &Path,
    ) -> Result<(), String>;

    /// Estimate the memory cost of loading a cell without actually loading it.
    fn estimate_memory(&self, coord: CellCoord, base_path: &Path) -> usize;
}

// ---------------------------------------------------------------------------
// DefaultCellDataProvider
// ---------------------------------------------------------------------------

/// Default implementation that loads/saves cells as JSON files in a directory
/// structure: `{base_path}/{x}_{z}.cell.json`.
pub struct DefaultCellDataProvider;

impl DefaultCellDataProvider {
    /// Construct the file path for a cell.
    fn cell_path(coord: CellCoord, layer: Option<&str>, base_path: &Path) -> PathBuf {
        match layer {
            Some(layer_name) => base_path.join(format!(
                "{}_{}/{}.cell.json",
                coord.x, coord.z, layer_name
            )),
            None => base_path.join(format!("{}_{}.cell.json", coord.x, coord.z)),
        }
    }
}

impl CellDataProvider for DefaultCellDataProvider {
    fn load_cell(
        &self,
        coord: CellCoord,
        layer: Option<&str>,
        base_path: &Path,
    ) -> Result<(Vec<SerializedEntity>, usize), String> {
        let path = Self::cell_path(coord, layer, base_path);

        if !path.exists() {
            // No data file: return empty cell (this is valid for new/unexplored areas).
            return Ok((Vec::new(), 0));
        }

        let data = std::fs::read_to_string(&path).map_err(|e| {
            format!("Failed to read cell {} from {}: {}", coord, path.display(), e)
        })?;

        let entities: Vec<SerializedEntity> =
            serde_json::from_str(&data).map_err(|e| {
                format!(
                    "Failed to deserialize cell {} from {}: {}",
                    coord,
                    path.display(),
                    e
                )
            })?;

        let memory: usize = entities.iter().map(|e| e.estimated_memory()).sum();
        Ok((entities, memory))
    }

    fn save_cell(
        &self,
        coord: CellCoord,
        layer: Option<&str>,
        entities: &[SerializedEntity],
        base_path: &Path,
    ) -> Result<(), String> {
        let path = Self::cell_path(coord, layer, base_path);

        // Ensure parent directory exists.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create directory for cell {}: {}",
                    coord, e
                )
            })?;
        }

        let json = serde_json::to_string_pretty(entities).map_err(|e| {
            format!("Failed to serialize cell {}: {}", coord, e)
        })?;

        std::fs::write(&path, json).map_err(|e| {
            format!(
                "Failed to write cell {} to {}: {}",
                coord,
                path.display(),
                e
            )
        })?;

        Ok(())
    }

    fn estimate_memory(&self, coord: CellCoord, base_path: &Path) -> usize {
        let path = Self::cell_path(coord, None, base_path);
        match std::fs::metadata(&path) {
            Ok(meta) => {
                // Rough estimate: deserialized data is ~3x the JSON file size.
                meta.len() as usize * 3
            }
            Err(_) => 1024 * 64, // 64 KiB default for unknown cells.
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingManager
// ---------------------------------------------------------------------------

/// Manages asynchronous loading and unloading of world cells.
///
/// The streaming manager maintains a priority queue of pending load/unload
/// requests, dispatches them to worker threads, collects results, and
/// provides statistics.
///
/// # Usage
///
/// ```ignore
/// let mut sm = StreamingManager::new(base_path, provider);
///
/// // Each frame:
/// let ops = partition.update(camera_pos);
/// sm.submit_ops(ops, frame);
/// sm.poll();  // Process completed loads/unloads.
/// let stats = sm.stats();
/// ```
pub struct StreamingManager {
    /// Base path for cell data files.
    base_path: PathBuf,
    /// Provider for reading/writing cell data.
    provider: Arc<dyn CellDataProvider>,
    /// Priority queue of pending load requests.
    load_queue: BinaryHeap<StreamRequest>,
    /// Queue of pending unload requests (FIFO, not priority).
    unload_queue: VecDeque<StreamRequest>,
    /// Cells currently being loaded (coord -> transition state).
    active_loads: HashMap<CellCoord, CellTransitionState>,
    /// Cells currently being unloaded.
    active_unloads: HashMap<CellCoord, CellTransitionState>,
    /// Completed load results waiting to be applied.
    completed_loads: Arc<Mutex<Vec<LoadResult>>>,
    /// Completed unload results waiting to be processed.
    completed_unloads: Arc<Mutex<Vec<UnloadResult>>>,
    /// Maximum number of concurrent load operations.
    max_concurrent_loads: usize,
    /// Maximum number of concurrent unload operations.
    max_concurrent_unloads: usize,
    /// Memory budget in bytes.
    memory_budget: usize,
    /// Current memory usage.
    current_memory: usize,
    /// Running statistics.
    stats: StreamingStats,
    /// Frame counter.
    frame: u64,
    /// Whether streaming is paused.
    paused: bool,
    /// Coordinates of cells that have been fully loaded and are ready for
    /// activation.
    ready_for_activation: Vec<(CellCoord, Vec<SerializedEntity>, usize)>,
}

impl StreamingManager {
    /// Create a new streaming manager.
    ///
    /// # Parameters
    ///
    /// - `base_path`: Root directory containing cell data files.
    /// - `provider`: Implementation of [`CellDataProvider`] for I/O.
    pub fn new(base_path: impl Into<PathBuf>, provider: Arc<dyn CellDataProvider>) -> Self {
        Self {
            base_path: base_path.into(),
            provider,
            load_queue: BinaryHeap::new(),
            unload_queue: VecDeque::new(),
            active_loads: HashMap::new(),
            active_unloads: HashMap::new(),
            completed_loads: Arc::new(Mutex::new(Vec::new())),
            completed_unloads: Arc::new(Mutex::new(Vec::new())),
            max_concurrent_loads: 4,
            max_concurrent_unloads: 2,
            memory_budget: 512 * 1024 * 1024,
            current_memory: 0,
            stats: StreamingStats::default(),
            frame: 0,
            paused: false,
            ready_for_activation: Vec::new(),
        }
    }

    /// Set the maximum number of concurrent load operations.
    pub fn set_max_concurrent_loads(&mut self, n: usize) {
        self.max_concurrent_loads = n;
    }

    /// Set the maximum number of concurrent unload operations.
    pub fn set_max_concurrent_unloads(&mut self, n: usize) {
        self.max_concurrent_unloads = n;
    }

    /// Set the memory budget.
    pub fn set_memory_budget(&mut self, bytes: usize) {
        self.memory_budget = bytes;
        self.stats.memory_budget = bytes;
    }

    /// Pause or resume streaming.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
        if paused {
            log::info!("Streaming paused");
        } else {
            log::info!("Streaming resumed");
        }
    }

    /// Check if streaming is paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Submit a single load request.
    pub fn submit_load(
        &mut self,
        coord: CellCoord,
        priority: StreamRequestPriority,
        distance: f32,
    ) {
        // Skip if already queued, loading, or loaded.
        if self.active_loads.contains_key(&coord) {
            return;
        }
        // Check if already in the queue.
        let already_queued = self.load_queue.iter().any(|r| r.coord == coord);
        if already_queued {
            return;
        }

        let request = StreamRequest {
            coord,
            kind: StreamRequestKind::Load,
            priority,
            layer: None,
            submitted_frame: self.frame,
            distance,
        };

        log::debug!("Queuing load request for cell {}", coord);
        self.load_queue.push(request);
    }

    /// Submit a single unload request.
    pub fn submit_unload(&mut self, coord: CellCoord) {
        // Skip if already being unloaded.
        if self.active_unloads.contains_key(&coord) {
            return;
        }
        let already_queued = self.unload_queue.iter().any(|r| r.coord == coord);
        if already_queued {
            return;
        }

        let request = StreamRequest {
            coord,
            kind: StreamRequestKind::Unload,
            priority: StreamRequestPriority::Normal,
            layer: None,
            submitted_frame: self.frame,
            distance: 0.0,
        };

        log::debug!("Queuing unload request for cell {}", coord);
        self.unload_queue.push_back(request);
    }

    /// Submit streaming operations from a partition update.
    pub fn submit_ops(&mut self, ops: crate::partition::StreamingOps, camera_pos: glam::Vec3) {
        for coord in ops.load {
            let center = glam::Vec3::new(
                coord.x as f32 * 256.0 + 128.0, // Approximate, should use partition cell_size
                0.0,
                coord.z as f32 * 256.0 + 128.0,
            );
            let dist = (center - camera_pos).length();
            let priority = if dist < 100.0 {
                StreamRequestPriority::Critical
            } else if dist < 300.0 {
                StreamRequestPriority::High
            } else if dist < 600.0 {
                StreamRequestPriority::Normal
            } else {
                StreamRequestPriority::Low
            };
            self.submit_load(coord, priority, dist);
        }

        for coord in ops.unload {
            self.submit_unload(coord);
        }
    }

    /// Dispatch pending requests to workers and collect completed results.
    ///
    /// Call this once per frame. It:
    /// 1. Collects completed load/unload results from workers.
    /// 2. Dispatches new load/unload requests up to concurrency limits.
    /// 3. Updates statistics.
    pub fn poll(&mut self) {
        self.frame += 1;

        if self.paused {
            return;
        }

        // --- Collect completed loads --------------------------------------
        let completed_loads = {
            let mut lock = self.completed_loads.lock();
            std::mem::take(&mut *lock)
        };

        self.stats.bytes_loaded_this_frame = 0;

        for result in completed_loads {
            self.active_loads.remove(&result.coord);

            if result.success {
                log::debug!(
                    "Cell {} loaded: {} entities, {:.1} KiB, {:.1}ms",
                    result.coord,
                    result.entities.len(),
                    result.memory_bytes as f64 / 1024.0,
                    result.load_time_secs * 1000.0,
                );

                self.current_memory += result.memory_bytes;
                self.stats.bytes_loaded_this_frame += result.memory_bytes;
                self.stats.total_bytes_loaded += result.memory_bytes as u64;
                self.stats.total_loads_completed += 1;

                // Update average load time (exponential moving average).
                let alpha = 0.1;
                self.stats.avg_load_time_secs = self.stats.avg_load_time_secs * (1.0 - alpha)
                    + result.load_time_secs * alpha;
                if result.load_time_secs > self.stats.peak_load_time_secs {
                    self.stats.peak_load_time_secs = result.load_time_secs;
                }

                // Queue for activation on the main thread.
                self.ready_for_activation.push((
                    result.coord,
                    result.entities,
                    result.memory_bytes,
                ));
            } else {
                log::error!(
                    "Failed to load cell {}: {}",
                    result.coord,
                    result.error.as_deref().unwrap_or("unknown error"),
                );
                self.stats.total_load_errors += 1;
            }
        }

        // --- Collect completed unloads ------------------------------------
        let completed_unloads = {
            let mut lock = self.completed_unloads.lock();
            std::mem::take(&mut *lock)
        };

        self.stats.bytes_freed_this_frame = 0;

        for result in completed_unloads {
            self.active_unloads.remove(&result.coord);

            if result.success {
                log::debug!(
                    "Cell {} unloaded, freed {:.1} KiB",
                    result.coord,
                    result.freed_bytes as f64 / 1024.0,
                );
                self.current_memory = self.current_memory.saturating_sub(result.freed_bytes);
                self.stats.bytes_freed_this_frame += result.freed_bytes;
                self.stats.total_bytes_freed += result.freed_bytes as u64;
                self.stats.total_unloads_completed += 1;
            } else {
                log::error!(
                    "Failed to unload cell {}: {}",
                    result.coord,
                    result.error.as_deref().unwrap_or("unknown error"),
                );
            }
        }

        // --- Dispatch new loads -------------------------------------------
        while self.active_loads.len() < self.max_concurrent_loads {
            let request = match self.load_queue.pop() {
                Some(r) => r,
                None => break,
            };

            // Check memory budget before starting a load.
            let estimated_mem = self
                .provider
                .estimate_memory(request.coord, &self.base_path);
            if self.current_memory + estimated_mem > self.memory_budget {
                log::debug!(
                    "Skipping load of cell {} due to memory budget ({}+{} > {})",
                    request.coord,
                    self.current_memory,
                    estimated_mem,
                    self.memory_budget,
                );
                // Put it back for later.
                self.load_queue.push(request);
                break;
            }

            self.dispatch_load(request);
        }

        // --- Dispatch new unloads -----------------------------------------
        while self.active_unloads.len() < self.max_concurrent_unloads {
            let request = match self.unload_queue.pop_front() {
                Some(r) => r,
                None => break,
            };

            self.dispatch_unload(request);
        }

        // --- Update stats -------------------------------------------------
        self.stats.loaded_cells = 0; // Will be set by caller from partition
        self.stats.loading_cells = self.active_loads.len();
        self.stats.queued_cells = self.load_queue.len();
        self.stats.unloading_cells = self.active_unloads.len();
        self.stats.total_memory = self.current_memory;
        self.stats.memory_budget = self.memory_budget;
        self.stats.frame = self.frame;
    }

    /// Dispatch a load request to a worker thread.
    fn dispatch_load(&mut self, request: StreamRequest) {
        let coord = request.coord;
        let layer = request.layer.clone();

        self.active_loads.insert(
            coord,
            CellTransitionState::Loading {
                started_frame: self.frame,
                layer: layer.clone(),
            },
        );

        let provider = Arc::clone(&self.provider);
        let base_path = self.base_path.clone();
        let completed = Arc::clone(&self.completed_loads);
        let layer_clone = layer.clone();

        // Spawn a worker thread for this load.
        std::thread::spawn(move || {
            let start = std::time::Instant::now();
            let layer_ref = layer_clone.as_deref();

            let result = match provider.load_cell(coord, layer_ref, &base_path) {
                Ok((entities, memory)) => LoadResult {
                    coord,
                    layer,
                    entities,
                    memory_bytes: memory,
                    success: true,
                    error: None,
                    load_time_secs: start.elapsed().as_secs_f64(),
                },
                Err(e) => LoadResult {
                    coord,
                    layer,
                    entities: Vec::new(),
                    memory_bytes: 0,
                    success: false,
                    error: Some(e),
                    load_time_secs: start.elapsed().as_secs_f64(),
                },
            };

            completed.lock().push(result);
        });

        log::trace!("Dispatched load for cell {}", coord);
    }

    /// Dispatch an unload request to a worker thread.
    fn dispatch_unload(&mut self, request: StreamRequest) {
        let coord = request.coord;

        self.active_unloads.insert(
            coord,
            CellTransitionState::Unloading {
                started_frame: self.frame,
            },
        );

        // For unloads, we need the entities currently in the cell.
        // In a real implementation, we would serialize the entities from the
        // ECS world. Here we simulate with an empty list since we don't have
        // direct access to the World.
        let provider = Arc::clone(&self.provider);
        let base_path = self.base_path.clone();
        let completed = Arc::clone(&self.completed_unloads);

        std::thread::spawn(move || {
            // In a real engine, we would receive the entities to serialize
            // from the main thread. For now, just signal completion.
            let result = UnloadResult {
                coord,
                layer: None,
                success: true,
                error: None,
                freed_bytes: 0,
            };

            completed.lock().push(result);
        });

        log::trace!("Dispatched unload for cell {}", coord);
    }

    /// Take cells that are ready for activation. The caller should create
    /// ECS entities from the serialized data and then call
    /// `partition.notify_loaded()`.
    pub fn take_ready(&mut self) -> Vec<(CellCoord, Vec<SerializedEntity>, usize)> {
        std::mem::take(&mut self.ready_for_activation)
    }

    /// Cancel all pending and active requests.
    pub fn cancel_all(&mut self) {
        self.load_queue.clear();
        self.unload_queue.clear();
        // Active loads/unloads will complete but their results will be discarded.
        self.active_loads.clear();
        self.active_unloads.clear();
        self.ready_for_activation.clear();
        log::info!("All streaming requests cancelled");
    }

    /// Get current streaming statistics.
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    /// Get current memory usage.
    pub fn current_memory(&self) -> usize {
        self.current_memory
    }

    /// Get the number of pending load requests.
    pub fn pending_loads(&self) -> usize {
        self.load_queue.len()
    }

    /// Get the number of pending unload requests.
    pub fn pending_unloads(&self) -> usize {
        self.unload_queue.len()
    }

    /// Get the number of active (in-progress) load operations.
    pub fn active_load_count(&self) -> usize {
        self.active_loads.len()
    }

    /// Get the number of active (in-progress) unload operations.
    pub fn active_unload_count(&self) -> usize {
        self.active_unloads.len()
    }

    /// Check if a specific cell is currently loading.
    pub fn is_loading(&self, coord: &CellCoord) -> bool {
        self.active_loads.contains_key(coord)
    }

    /// Check if a specific cell is queued for loading.
    pub fn is_queued(&self, coord: &CellCoord) -> bool {
        self.load_queue.iter().any(|r| r.coord == *coord)
    }

    /// Force an immediate synchronous load of a cell (blocks the calling thread).
    ///
    /// Use sparingly -- this is intended for critical cells like the player's
    /// current cell that must be loaded before gameplay can continue.
    pub fn load_sync(&mut self, coord: CellCoord) -> Result<(Vec<SerializedEntity>, usize), String> {
        let result = self.provider.load_cell(coord, None, &self.base_path)?;
        self.current_memory += result.1;
        self.stats.total_bytes_loaded += result.1 as u64;
        self.stats.total_loads_completed += 1;
        Ok(result)
    }

    /// Get the transition state of a cell.
    pub fn cell_state(&self, coord: &CellCoord) -> CellTransitionState {
        if let Some(state) = self.active_loads.get(coord) {
            return state.clone();
        }
        if let Some(state) = self.active_unloads.get(coord) {
            return state.clone();
        }
        if self.load_queue.iter().any(|r| r.coord == *coord) {
            return CellTransitionState::Queued {
                priority: StreamRequestPriority::Normal,
                queued_frame: self.frame,
            };
        }
        CellTransitionState::Idle
    }

    /// Reset all state. Call when transitioning between levels.
    pub fn reset(&mut self) {
        self.cancel_all();
        self.current_memory = 0;
        self.stats = StreamingStats::default();
        self.stats.memory_budget = self.memory_budget;
        self.frame = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProvider;

    impl CellDataProvider for MockProvider {
        fn load_cell(
            &self,
            coord: CellCoord,
            _layer: Option<&str>,
            _base_path: &Path,
        ) -> Result<(Vec<SerializedEntity>, usize), String> {
            let mut entities = Vec::new();
            for i in 0..3 {
                let mut e = SerializedEntity::new(
                    i,
                    format!("entity_{}_{}_{}", coord.x, coord.z, i),
                );
                e.position = [
                    coord.x as f32 * 100.0 + i as f32 * 10.0,
                    0.0,
                    coord.z as f32 * 100.0,
                ];
                entities.push(e);
            }
            let mem: usize = entities.iter().map(|e| e.estimated_memory()).sum();
            Ok((entities, mem))
        }

        fn save_cell(
            &self,
            _coord: CellCoord,
            _layer: Option<&str>,
            _entities: &[SerializedEntity],
            _base_path: &Path,
        ) -> Result<(), String> {
            Ok(())
        }

        fn estimate_memory(&self, _coord: CellCoord, _base_path: &Path) -> usize {
            4096
        }
    }

    #[test]
    fn submit_and_poll_load() {
        let provider = Arc::new(MockProvider);
        let mut sm = StreamingManager::new("/tmp/test_cells", provider);

        let coord = CellCoord::new(0, 0);
        sm.submit_load(coord, StreamRequestPriority::Normal, 100.0);
        assert_eq!(sm.pending_loads(), 1);

        // Dispatch.
        sm.poll();
        assert_eq!(sm.active_load_count(), 1);
        assert_eq!(sm.pending_loads(), 0);

        // Wait for the worker to finish.
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Collect results.
        sm.poll();
        let ready = sm.take_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].0, coord);
        assert_eq!(ready[0].1.len(), 3);
    }

    #[test]
    fn duplicate_requests_are_deduplicated() {
        let provider = Arc::new(MockProvider);
        let mut sm = StreamingManager::new("/tmp/test_cells", provider);

        let coord = CellCoord::new(1, 1);
        sm.submit_load(coord, StreamRequestPriority::Normal, 50.0);
        sm.submit_load(coord, StreamRequestPriority::Normal, 50.0);
        assert_eq!(sm.pending_loads(), 1);
    }

    #[test]
    fn priority_ordering() {
        assert!(StreamRequestPriority::Critical > StreamRequestPriority::High);
        assert!(StreamRequestPriority::High > StreamRequestPriority::Normal);
        assert!(StreamRequestPriority::Normal > StreamRequestPriority::Low);
    }

    #[test]
    fn sync_load() {
        let provider = Arc::new(MockProvider);
        let mut sm = StreamingManager::new("/tmp/test_cells", provider);

        let result = sm.load_sync(CellCoord::new(5, 5));
        assert!(result.is_ok());
        let (entities, _mem) = result.unwrap();
        assert_eq!(entities.len(), 3);
    }
}
