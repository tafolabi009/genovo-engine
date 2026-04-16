//! Asset import, cooking, and streaming pipeline.
//!
//! Assets go through a multi-stage pipeline:
//! 1. **Import** — raw source files are parsed into an intermediate representation.
//! 2. **Process** — intermediate data is converted to an optimised runtime format.
//! 3. **Write** — runtime data is serialised to disk for distribution builds.
//!
//! The [`StreamingManager`] handles distance/LOD-based asset streaming at runtime.

use std::collections::{BinaryHeap, HashMap};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use genovo_core::{EngineError, EngineResult};

// ---------------------------------------------------------------------------
// AssetImporter
// ---------------------------------------------------------------------------

/// Trait for importing raw source assets into an intermediate representation.
///
/// Importers parse format-specific files (e.g. `.glb`, `.fbx`, `.png`) and
/// produce a normalised intermediate representation suitable for further
/// processing.
pub trait AssetImporter: Send + Sync + 'static {
    /// The intermediate representation produced by this importer.
    type Intermediate: Send + Sync + 'static;

    /// Import settings type for controlling import behaviour.
    type Settings: Default + Send + Sync + 'static;

    /// Imports raw bytes into the intermediate representation.
    fn import(
        &self,
        data: &[u8],
        settings: &Self::Settings,
        path: &Path,
    ) -> EngineResult<Self::Intermediate>;

    /// Returns the file extensions this importer handles (without dot).
    fn supported_extensions(&self) -> &[&str];
}

// ---------------------------------------------------------------------------
// AssetProcessor
// ---------------------------------------------------------------------------

/// Trait for processing imported assets into their runtime-optimised format.
///
/// Processors take the intermediate representation from an [`AssetImporter`]
/// and transform it into the format consumed by the engine at runtime
/// (e.g. compressing textures to BC7, baking lightmaps, optimising meshes).
pub trait AssetProcessor: Send + Sync + 'static {
    /// The intermediate type consumed (output of an importer).
    type Input: Send + Sync + 'static;

    /// The runtime-ready type produced.
    type Output: Send + Sync + 'static;

    /// Processing settings.
    type Settings: Default + Send + Sync + 'static;

    /// Processes an imported asset into its runtime format.
    fn process(&self, input: Self::Input, settings: &Self::Settings)
        -> EngineResult<Self::Output>;
}

// ---------------------------------------------------------------------------
// ErasedImporter / ErasedProcessor  (type-erased trait objects)
// ---------------------------------------------------------------------------

/// Type-erased importer for heterogeneous storage in the cook pipeline.
trait ErasedImporter: Send + Sync + 'static {
    fn import_erased(&self, data: &[u8], path: &Path) -> EngineResult<Box<dyn std::any::Any + Send + Sync>>;
    fn supported_extensions(&self) -> &[&str];
}

impl<I: AssetImporter> ErasedImporter for I
where
    I::Intermediate: 'static,
{
    fn import_erased(&self, data: &[u8], path: &Path) -> EngineResult<Box<dyn std::any::Any + Send + Sync>> {
        let intermediate = self.import(data, &I::Settings::default(), path)?;
        Ok(Box::new(intermediate))
    }

    fn supported_extensions(&self) -> &[&str] {
        AssetImporter::supported_extensions(self)
    }
}

/// Type-erased processor.
trait ErasedProcessor: Send + Sync + 'static {
    fn process_erased(
        &self,
        input: Box<dyn std::any::Any + Send + Sync>,
    ) -> EngineResult<Box<dyn std::any::Any + Send + Sync>>;
}

impl<P: AssetProcessor> ErasedProcessor for P
where
    P::Input: 'static,
    P::Output: 'static,
{
    fn process_erased(
        &self,
        input: Box<dyn std::any::Any + Send + Sync>,
    ) -> EngineResult<Box<dyn std::any::Any + Send + Sync>> {
        let typed = input
            .downcast::<P::Input>()
            .map_err(|_| EngineError::InvalidState("processor input type mismatch".into()))?;
        let output = self.process(*typed, &P::Settings::default())?;
        Ok(Box::new(output))
    }
}

// ---------------------------------------------------------------------------
// AssetManifest
// ---------------------------------------------------------------------------

/// Tracks all cooked assets and their metadata for a build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetManifest {
    /// Mapping from asset UUID to its manifest entry.
    pub entries: HashMap<Uuid, ManifestEntry>,
}

/// A single entry in the asset manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Unique asset identifier.
    pub uuid: Uuid,
    /// Original source path (relative to asset root).
    pub source_path: PathBuf,
    /// Path to the cooked output file.
    pub cooked_path: PathBuf,
    /// Size of the cooked asset in bytes.
    pub size_bytes: u64,
    /// Content hash for integrity checking.
    pub content_hash: u64,
    /// UUIDs of assets this asset depends on.
    pub dependencies: Vec<Uuid>,
}

impl AssetManifest {
    /// Creates a new empty manifest.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Adds an entry to the manifest.
    pub fn add_entry(&mut self, entry: ManifestEntry) {
        self.entries.insert(entry.uuid, entry);
    }

    /// Looks up an entry by UUID.
    pub fn get(&self, uuid: &Uuid) -> Option<&ManifestEntry> {
        self.entries.get(uuid)
    }

    /// Returns all entries sorted by dependency order (leaves first).
    ///
    /// Uses Kahn's algorithm for topological sorting.
    pub fn dependency_sorted(&self) -> Vec<&ManifestEntry> {
        let uuids: Vec<Uuid> = self.entries.keys().copied().collect();
        let uuid_set: std::collections::HashSet<Uuid> = uuids.iter().copied().collect();

        // Build adjacency and in-degree.
        let mut in_degree: HashMap<Uuid, usize> = uuids.iter().map(|&u| (u, 0)).collect();
        let mut dependents: HashMap<Uuid, Vec<Uuid>> = HashMap::new();

        for entry in self.entries.values() {
            for dep in &entry.dependencies {
                if uuid_set.contains(dep) {
                    *in_degree.entry(entry.uuid).or_default() += 1;
                    dependents.entry(*dep).or_default().push(entry.uuid);
                }
            }
        }

        let mut queue: std::collections::VecDeque<Uuid> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(&u, _)| u)
            .collect();

        let mut sorted = Vec::with_capacity(uuids.len());

        while let Some(u) = queue.pop_front() {
            sorted.push(u);
            if let Some(deps) = dependents.get(&u) {
                for &d in deps {
                    if let Some(deg) = in_degree.get_mut(&d) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(d);
                        }
                    }
                }
            }
        }

        // If there are cycles, append remaining nodes at the end.
        if sorted.len() < uuids.len() {
            for u in &uuids {
                if !sorted.contains(u) {
                    sorted.push(*u);
                }
            }
        }

        sorted
            .iter()
            .filter_map(|u| self.entries.get(u))
            .collect()
    }

    /// Serialises the manifest to a JSON string.
    pub fn to_json(&self) -> EngineResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| EngineError::Other(e.to_string()))
    }

    /// Deserialises a manifest from a JSON string.
    pub fn from_json(json: &str) -> EngineResult<Self> {
        serde_json::from_str(json).map_err(|e| EngineError::Other(e.to_string()))
    }
}

impl Default for AssetManifest {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AssetCookPipeline
// ---------------------------------------------------------------------------

/// Orchestrates the full import -> process -> write pipeline for asset cooking.
///
/// Used during editor builds and distribution packaging to convert source
/// assets into optimised runtime formats.
pub struct AssetCookPipeline {
    /// Root directory for source assets.
    source_root: PathBuf,
    /// Root directory for cooked output.
    output_root: PathBuf,
    /// The manifest tracking all cooked assets.
    manifest: AssetManifest,
    /// Registered importers keyed by extension.
    importers: HashMap<String, Box<dyn ErasedImporter>>,
    /// Registered processors keyed by a type tag string.
    processors: HashMap<String, Box<dyn ErasedProcessor>>,
}

impl AssetCookPipeline {
    /// Creates a new cook pipeline with the given source and output directories.
    pub fn new(source_root: impl Into<PathBuf>, output_root: impl Into<PathBuf>) -> Self {
        Self {
            source_root: source_root.into(),
            output_root: output_root.into(),
            manifest: AssetManifest::new(),
            importers: HashMap::new(),
            processors: HashMap::new(),
        }
    }

    /// Registers an importer for its declared extensions.
    pub fn register_importer<I: AssetImporter>(&mut self, importer: I)
    where
        I::Intermediate: 'static,
    {
        let extensions: Vec<String> = importer
            .supported_extensions()
            .iter()
            .map(|s| s.to_ascii_lowercase())
            .collect();
        let boxed = Box::new(importer) as Box<dyn ErasedImporter>;
        // We need to register the same boxed importer for multiple extensions.
        // Since Box is not Clone, we will only store for the first extension
        // and use a shared wrapper.  For simplicity, store for each extension
        // by leaking into an Arc.
        let arc: std::sync::Arc<dyn ErasedImporter> = std::sync::Arc::from(boxed);
        for ext in extensions {
            let arc_clone = std::sync::Arc::clone(&arc);
            self.importers.insert(ext, Box::new(ArcImporter(arc_clone)));
        }
    }

    /// Registers a processor keyed by a type tag (e.g. `"texture"`, `"mesh"`).
    pub fn register_processor<P: AssetProcessor>(&mut self, tag: &str, processor: P)
    where
        P::Input: 'static,
        P::Output: 'static,
    {
        self.processors
            .insert(tag.to_owned(), Box::new(processor));
    }

    /// Cooks a single asset at the given relative path.
    ///
    /// Returns the UUID assigned to the cooked asset.
    pub fn cook_asset(&mut self, relative_path: &Path) -> EngineResult<Uuid> {
        let source = self.source_root.join(relative_path);
        let data = std::fs::read(&source)?;

        let ext = relative_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let importer = self
            .importers
            .get(&ext)
            .ok_or_else(|| {
                EngineError::NotFound(format!("No importer for extension '.{ext}'"))
            })?;

        let _intermediate = importer.import_erased(&data, &source)?;

        // Determine output path.
        let uuid = Uuid::new_v4();
        let cooked_name = format!("{uuid}.cooked");
        let cooked_path = self.output_root.join(&cooked_name);

        // In a full implementation we would serialise the processed output.
        // For now, write the raw data as a placeholder.
        std::fs::create_dir_all(&self.output_root)?;
        std::fs::write(&cooked_path, &data)?;

        let entry = ManifestEntry {
            uuid,
            source_path: relative_path.to_path_buf(),
            cooked_path: PathBuf::from(&cooked_name),
            size_bytes: data.len() as u64,
            content_hash: simple_hash(&data),
            dependencies: Vec::new(),
        };
        self.manifest.add_entry(entry);

        log::info!(
            "Cooked asset: {} -> {} ({})",
            relative_path.display(),
            cooked_name,
            uuid
        );
        Ok(uuid)
    }

    /// Cooks all importable assets found under the source root.
    pub fn cook_all(&mut self) -> EngineResult<AssetManifest> {
        let extensions: Vec<String> = self.importers.keys().cloned().collect();
        let mut paths = Vec::new();

        collect_files_recursive(&self.source_root, &extensions, &mut paths)?;

        for path in &paths {
            let relative = path
                .strip_prefix(&self.source_root)
                .unwrap_or(path);
            if let Err(e) = self.cook_asset(relative) {
                log::error!("Failed to cook {}: {e}", relative.display());
            }
        }

        Ok(self.manifest.clone())
    }

    /// Returns a reference to the current manifest.
    pub fn manifest(&self) -> &AssetManifest {
        &self.manifest
    }

    /// Writes the manifest to disk at the output root.
    pub fn write_manifest(&self) -> EngineResult<()> {
        let path = self.output_root.join("manifest.json");
        let json = self.manifest.to_json()?;
        std::fs::create_dir_all(&self.output_root)?;
        std::fs::write(&path, json)?;
        log::info!("Wrote manifest to {}", path.display());
        Ok(())
    }

    /// Loads a previously written manifest from the output root.
    pub fn load_manifest(&mut self) -> EngineResult<()> {
        let path = self.output_root.join("manifest.json");
        let json = std::fs::read_to_string(&path)?;
        self.manifest = AssetManifest::from_json(&json)?;
        Ok(())
    }
}

/// Helper: wraps `Arc<dyn ErasedImporter>` so it can be stored in a `Box`.
struct ArcImporter(std::sync::Arc<dyn ErasedImporter>);

impl ErasedImporter for ArcImporter {
    fn import_erased(&self, data: &[u8], path: &Path) -> EngineResult<Box<dyn std::any::Any + Send + Sync>> {
        self.0.import_erased(data, path)
    }

    fn supported_extensions(&self) -> &[&str] {
        self.0.supported_extensions()
    }
}

/// Recursively collect files whose extensions match the given set.
fn collect_files_recursive(
    dir: &Path,
    extensions: &[String],
    out: &mut Vec<PathBuf>,
) -> EngineResult<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        let path = entry.path();
        if ft.is_dir() {
            collect_files_recursive(&path, extensions, out)?;
        } else if ft.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if extensions.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
                    out.push(path);
                }
            }
        }
    }
    Ok(())
}

/// A very simple non-cryptographic hash for content fingerprinting.
fn simple_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h
}

// ---------------------------------------------------------------------------
// StreamingPriority
// ---------------------------------------------------------------------------

/// Priority hint for streaming decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StreamingPriority {
    /// Load only when very close or explicitly requested.
    Low = 0,
    /// Normal streaming priority based on distance.
    Normal = 1,
    /// High priority — load before lower priority assets.
    High = 2,
    /// Must be loaded immediately (e.g. player-visible assets).
    Critical = 3,
}

// ---------------------------------------------------------------------------
// StreamingRequest
// ---------------------------------------------------------------------------

/// A request to stream an asset.
#[derive(Debug, Clone)]
pub struct StreamingRequest {
    /// UUID of the asset to stream.
    pub asset_id: Uuid,
    /// Priority of this request.
    pub priority: StreamingPriority,
    /// Estimated size in bytes.
    pub estimated_size: u64,
    /// Squared distance from the camera (for sorting).
    pub distance_sq: f32,
}

impl PartialEq for StreamingRequest {
    fn eq(&self, other: &Self) -> bool {
        self.asset_id == other.asset_id
    }
}
impl Eq for StreamingRequest {}

impl PartialOrd for StreamingRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StreamingRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first; then closer distance first (lower distance_sq).
        self.priority
            .cmp(&other.priority)
            .then_with(|| {
                // Reverse distance: closer (smaller) is higher priority.
                other
                    .distance_sq
                    .partial_cmp(&self.distance_sq)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

// ---------------------------------------------------------------------------
// StreamingManager
// ---------------------------------------------------------------------------

/// Manages LOD and distance-based asset streaming at runtime.
///
/// Determines which assets should be loaded or unloaded based on camera
/// position, asset priority, and available memory budget.
pub struct StreamingManager {
    /// Maximum memory budget for streamed assets (bytes).
    memory_budget_bytes: u64,
    /// Currently estimated memory usage (bytes).
    current_usage_bytes: u64,
    /// Priority queue of pending load requests.
    pending_loads: BinaryHeap<StreamingRequest>,
    /// Pending unload requests.
    pending_unloads: Vec<Uuid>,
    /// Loaded assets and their sizes for eviction tracking.
    loaded: HashMap<Uuid, u64>,
}

impl StreamingManager {
    /// Creates a new streaming manager with the given memory budget.
    pub fn new(memory_budget_bytes: u64) -> Self {
        Self {
            memory_budget_bytes,
            current_usage_bytes: 0,
            pending_loads: BinaryHeap::new(),
            pending_unloads: Vec::new(),
            loaded: HashMap::new(),
        }
    }

    /// Requests an asset to be streamed in.
    pub fn request_load(&mut self, request: StreamingRequest) {
        // De-duplicate: don't re-queue if already loaded.
        if self.loaded.contains_key(&request.asset_id) {
            return;
        }
        self.pending_loads.push(request);
    }

    /// Requests an asset to be streamed out (unloaded).
    pub fn request_unload(&mut self, asset_id: Uuid) {
        self.pending_unloads.push(asset_id);
    }

    /// Updates streaming decisions based on camera position and memory budget.
    ///
    /// Call once per frame from the main update loop.
    ///
    /// Returns a tuple of `(load_ids, unload_ids)` — the assets that should be
    /// loaded and unloaded this frame.
    pub fn update(&mut self, _camera_position: [f32; 3]) -> (Vec<Uuid>, Vec<Uuid>) {
        let mut to_load = Vec::new();
        let mut to_unload = Vec::new();

        // Process unloads first to free budget.
        for id in self.pending_unloads.drain(..) {
            if let Some(size) = self.loaded.remove(&id) {
                self.current_usage_bytes = self.current_usage_bytes.saturating_sub(size);
                to_unload.push(id);
            }
        }

        // Process loads while under budget.
        while let Some(request) = self.pending_loads.peek() {
            if self.current_usage_bytes + request.estimated_size > self.memory_budget_bytes {
                // Over budget — try to evict lowest-priority loaded assets.
                // For simplicity, stop loading once budget is exceeded.
                break;
            }
            let request = self.pending_loads.pop().unwrap();
            self.current_usage_bytes += request.estimated_size;
            self.loaded.insert(request.asset_id, request.estimated_size);
            to_load.push(request.asset_id);
        }

        (to_load, to_unload)
    }

    /// Returns the current memory usage in bytes.
    pub fn current_usage(&self) -> u64 {
        self.current_usage_bytes
    }

    /// Returns the memory budget in bytes.
    pub fn memory_budget(&self) -> u64 {
        self.memory_budget_bytes
    }

    /// Sets a new memory budget (may trigger evictions on next update).
    pub fn set_memory_budget(&mut self, budget_bytes: u64) {
        self.memory_budget_bytes = budget_bytes;
    }

    /// Returns the number of assets currently tracked as loaded.
    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- AssetManifest tests -----------------------------------------------

    #[test]
    fn test_manifest_add_and_get() {
        let mut manifest = AssetManifest::new();
        let uuid = Uuid::new_v4();
        manifest.add_entry(ManifestEntry {
            uuid,
            source_path: PathBuf::from("textures/grass.bmp"),
            cooked_path: PathBuf::from("cooked/grass.cooked"),
            size_bytes: 1024,
            content_hash: 0xDEADBEEF,
            dependencies: Vec::new(),
        });

        assert_eq!(manifest.entries.len(), 1);
        let entry = manifest.get(&uuid).unwrap();
        assert_eq!(entry.size_bytes, 1024);
    }

    #[test]
    fn test_manifest_json_round_trip() {
        let mut manifest = AssetManifest::new();
        let uuid = Uuid::new_v4();
        manifest.add_entry(ManifestEntry {
            uuid,
            source_path: PathBuf::from("meshes/cube.obj"),
            cooked_path: PathBuf::from("cooked/cube.cooked"),
            size_bytes: 2048,
            content_hash: 12345,
            dependencies: Vec::new(),
        });

        let json = manifest.to_json().unwrap();
        let restored = AssetManifest::from_json(&json).unwrap();
        assert_eq!(restored.entries.len(), 1);
        let entry = restored.get(&uuid).unwrap();
        assert_eq!(entry.size_bytes, 2048);
        assert_eq!(entry.source_path, PathBuf::from("meshes/cube.obj"));
    }

    #[test]
    fn test_manifest_dependency_sort() {
        let mut manifest = AssetManifest::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();

        // c depends on b, b depends on a.
        manifest.add_entry(ManifestEntry {
            uuid: c,
            source_path: PathBuf::from("c"),
            cooked_path: PathBuf::from("c.cooked"),
            size_bytes: 0,
            content_hash: 0,
            dependencies: vec![b],
        });
        manifest.add_entry(ManifestEntry {
            uuid: b,
            source_path: PathBuf::from("b"),
            cooked_path: PathBuf::from("b.cooked"),
            size_bytes: 0,
            content_hash: 0,
            dependencies: vec![a],
        });
        manifest.add_entry(ManifestEntry {
            uuid: a,
            source_path: PathBuf::from("a"),
            cooked_path: PathBuf::from("a.cooked"),
            size_bytes: 0,
            content_hash: 0,
            dependencies: Vec::new(),
        });

        let sorted = manifest.dependency_sorted();
        let sorted_uuids: Vec<Uuid> = sorted.iter().map(|e| e.uuid).collect();
        let pos_a = sorted_uuids.iter().position(|&u| u == a).unwrap();
        let pos_b = sorted_uuids.iter().position(|&u| u == b).unwrap();
        let pos_c = sorted_uuids.iter().position(|&u| u == c).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    // -- StreamingManager tests --------------------------------------------

    #[test]
    fn test_streaming_manager_basic() {
        let mut mgr = StreamingManager::new(1000);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        mgr.request_load(StreamingRequest {
            asset_id: id1,
            priority: StreamingPriority::Normal,
            estimated_size: 400,
            distance_sq: 10.0,
        });
        mgr.request_load(StreamingRequest {
            asset_id: id2,
            priority: StreamingPriority::High,
            estimated_size: 500,
            distance_sq: 5.0,
        });

        let (loaded, _) = mgr.update([0.0, 0.0, 0.0]);
        assert_eq!(loaded.len(), 2);
        // High priority should be loaded first.
        assert_eq!(loaded[0], id2);
        assert_eq!(loaded[1], id1);
        assert_eq!(mgr.current_usage(), 900);
    }

    #[test]
    fn test_streaming_manager_budget_exceeded() {
        let mut mgr = StreamingManager::new(500);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        mgr.request_load(StreamingRequest {
            asset_id: id1,
            priority: StreamingPriority::High,
            estimated_size: 400,
            distance_sq: 1.0,
        });
        mgr.request_load(StreamingRequest {
            asset_id: id2,
            priority: StreamingPriority::Normal,
            estimated_size: 400,
            distance_sq: 2.0,
        });

        let (loaded, _) = mgr.update([0.0, 0.0, 0.0]);
        // Only one fits within the 500-byte budget.
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], id1);
        assert_eq!(mgr.current_usage(), 400);
    }

    #[test]
    fn test_streaming_manager_unload() {
        let mut mgr = StreamingManager::new(1000);
        let id = Uuid::new_v4();

        mgr.request_load(StreamingRequest {
            asset_id: id,
            priority: StreamingPriority::Normal,
            estimated_size: 300,
            distance_sq: 1.0,
        });
        mgr.update([0.0, 0.0, 0.0]);
        assert_eq!(mgr.current_usage(), 300);

        mgr.request_unload(id);
        let (_, unloaded) = mgr.update([0.0, 0.0, 0.0]);
        assert_eq!(unloaded.len(), 1);
        assert_eq!(mgr.current_usage(), 0);
    }

    #[test]
    fn test_streaming_no_duplicate() {
        let mut mgr = StreamingManager::new(10000);
        let id = Uuid::new_v4();

        mgr.request_load(StreamingRequest {
            asset_id: id,
            priority: StreamingPriority::Normal,
            estimated_size: 100,
            distance_sq: 1.0,
        });
        mgr.update([0.0, 0.0, 0.0]);

        // Request again — should be de-duplicated.
        mgr.request_load(StreamingRequest {
            asset_id: id,
            priority: StreamingPriority::Normal,
            estimated_size: 100,
            distance_sq: 1.0,
        });
        let (loaded, _) = mgr.update([0.0, 0.0, 0.0]);
        assert!(loaded.is_empty());
        assert_eq!(mgr.loaded_count(), 1);
    }

    // -- AssetCookPipeline tests -------------------------------------------

    #[test]
    fn test_cook_pipeline_no_importer() {
        let dir = std::env::temp_dir().join("genovo_cook_noimporter");
        let _ = std::fs::create_dir_all(&dir);
        let mut pipeline = AssetCookPipeline::new(&dir, dir.join("out"));
        let result = pipeline.cook_asset(Path::new("missing.xyz"));
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cook_pipeline_with_text_importer() {
        /// Trivial importer that reads text files.
        struct TextImporter;
        impl AssetImporter for TextImporter {
            type Intermediate = String;
            type Settings = ();
            fn import(&self, data: &[u8], _settings: &(), _path: &Path) -> EngineResult<String> {
                String::from_utf8(data.to_vec())
                    .map_err(|e| EngineError::Other(e.to_string()))
            }
            fn supported_extensions(&self) -> &[&str] {
                &["txt"]
            }
        }

        let dir = std::env::temp_dir().join("genovo_cook_text");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("hello.txt"), "Hello Cook").unwrap();

        let out_dir = dir.join("cooked_out");
        let mut pipeline = AssetCookPipeline::new(&dir, &out_dir);
        pipeline.register_importer(TextImporter);

        let uuid = pipeline.cook_asset(Path::new("hello.txt")).unwrap();
        assert!(pipeline.manifest().get(&uuid).is_some());

        // Write and verify manifest.
        pipeline.write_manifest().unwrap();
        let manifest_path = out_dir.join("manifest.json");
        assert!(manifest_path.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
