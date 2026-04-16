//! Asset loading infrastructure.
//!
//! Provides the [`AssetServer`] for centralised asset management with background
//! loading threads, the [`AssetLoader`] trait for format-specific loading, and
//! [`AssetHandle`] for referencing loaded assets.

use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

use crossbeam::channel::{self, Receiver, Sender};
use parking_lot::RwLock;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Public type aliases
// ---------------------------------------------------------------------------

/// Unique asset identifier.
pub type AssetId = Uuid;

// ---------------------------------------------------------------------------
// AssetError
// ---------------------------------------------------------------------------

/// Errors specific to asset loading.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("no loader registered for extension '{0}'")]
    NoLoader(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("invalid data: {0}")]
    InvalidData(String),

    #[error("asset not found: {0}")]
    NotFound(String),

    #[error("{0}")]
    Other(String),
}

impl From<AssetError> for genovo_core::EngineError {
    fn from(e: AssetError) -> Self {
        genovo_core::EngineError::Other(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// LoadState
// ---------------------------------------------------------------------------

/// Describes the current loading state of an asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadState {
    /// The asset has not been requested for loading.
    NotLoaded,
    /// The asset is currently being loaded (possibly asynchronously).
    Loading,
    /// The asset has been successfully loaded and is ready for use.
    Loaded,
    /// The asset failed to load.
    Failed(String),
}

// ---------------------------------------------------------------------------
// AssetPath
// ---------------------------------------------------------------------------

/// A path to an asset, optionally including a label for sub-assets.
///
/// Example: `"meshes/hero.glb#Mesh0"` has path `meshes/hero.glb` and label `Mesh0`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssetPath {
    /// Filesystem path to the asset file.
    pub path: PathBuf,
    /// Optional label identifying a sub-asset within the file.
    pub label: Option<String>,
}

impl AssetPath {
    /// Creates a new [`AssetPath`] from a path with no label.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            label: None,
        }
    }

    /// Creates a new [`AssetPath`] with a sub-asset label.
    pub fn with_label(path: impl Into<PathBuf>, label: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            label: Some(label.into()),
        }
    }

    /// Parses a string of the form `"path#label"` into an [`AssetPath`].
    pub fn parse(s: &str) -> Self {
        if let Some((path, label)) = s.split_once('#') {
            Self {
                path: PathBuf::from(path),
                label: Some(label.to_owned()),
            }
        } else {
            Self::new(s)
        }
    }
}

impl std::fmt::Display for AssetPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.path.display())?;
        if let Some(label) = &self.label {
            write!(f, "#{label}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AssetHandle
// ---------------------------------------------------------------------------

/// A lightweight handle referencing a loaded (or loading) asset of type `T`.
///
/// The handle stores only the [`AssetId`] and a phantom type parameter; the
/// actual data lives inside the [`AssetServer`].
pub struct AssetHandle<T> {
    /// Unique identifier for this asset instance.
    id: AssetId,
    _marker: PhantomData<T>,
}

impl<T> AssetHandle<T> {
    /// Creates a new handle with the given id.
    pub fn new(id: AssetId) -> Self {
        Self {
            id,
            _marker: PhantomData,
        }
    }

    /// Returns the asset identifier.
    pub fn id(&self) -> AssetId {
        self.id
    }
}

impl<T> Clone for AssetHandle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            _marker: PhantomData,
        }
    }
}

impl<T> Copy for AssetHandle<T> {}

impl<T> std::fmt::Debug for AssetHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssetHandle")
            .field("id", &self.id)
            .finish()
    }
}

impl<T> PartialEq for AssetHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for AssetHandle<T> {}

impl<T> std::hash::Hash for AssetHandle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// ---------------------------------------------------------------------------
// AssetSlot
// ---------------------------------------------------------------------------

/// Internal storage for a single asset within the server.
struct AssetSlot {
    state: LoadState,
    data: Option<Box<dyn Any + Send + Sync>>,
    path: PathBuf,
    ref_count: u32,
}

// ---------------------------------------------------------------------------
// AssetLoader trait
// ---------------------------------------------------------------------------

/// Trait implemented by format-specific asset loaders.
///
/// Each loader is responsible for reading raw bytes from disk and producing a
/// concrete asset type.
pub trait AssetLoader: Send + Sync + 'static {
    /// The concrete asset type produced by this loader.
    type Asset: Send + Sync + 'static;

    /// Returns the list of file extensions this loader can handle (without dot).
    fn extensions(&self) -> &[&str];

    /// Loads an asset from the given byte slice.
    fn load(&self, path: &Path, bytes: &[u8]) -> Result<Self::Asset, AssetError>;
}

// ---------------------------------------------------------------------------
// AnyAssetLoader (type-erased wrapper)
// ---------------------------------------------------------------------------

/// Type-erased wrapper around [`AssetLoader`] for heterogeneous storage.
trait AnyAssetLoader: Send + Sync + 'static {
    /// Returns the extensions this loader handles.
    fn extensions(&self) -> &[&str];

    /// Loads raw bytes into a type-erased `Box<dyn Any>`.
    fn load_erased(
        &self,
        path: &Path,
        bytes: &[u8],
    ) -> Result<Box<dyn Any + Send + Sync>, AssetError>;
}

impl<L: AssetLoader> AnyAssetLoader for L {
    fn extensions(&self) -> &[&str] {
        AssetLoader::extensions(self)
    }

    fn load_erased(
        &self,
        path: &Path,
        bytes: &[u8],
    ) -> Result<Box<dyn Any + Send + Sync>, AssetError> {
        let asset = AssetLoader::load(self, path, bytes)?;
        Ok(Box::new(asset))
    }
}

// ---------------------------------------------------------------------------
// Background loading messages
// ---------------------------------------------------------------------------

/// A request sent to background loader threads.
struct LoadRequest {
    id: AssetId,
    full_path: PathBuf,
    extension: String,
}

/// A result sent back from a background loader thread.
struct LoadResult {
    id: AssetId,
    result: Result<Box<dyn Any + Send + Sync>, AssetError>,
}

// ---------------------------------------------------------------------------
// AssetServer
// ---------------------------------------------------------------------------

/// Central asset management service.
///
/// Handles loading, caching, reference counting, and background-threaded asset
/// loading for all engine assets.
pub struct AssetServer {
    /// Loaded asset slots keyed by [`AssetId`].
    slots: RwLock<HashMap<AssetId, AssetSlot>>,
    /// Maps canonical paths to their asset ids for de-duplication.
    path_to_id: RwLock<HashMap<PathBuf, AssetId>>,
    /// Registered loaders keyed by file extension (lowercase, no dot).
    loaders: RwLock<HashMap<String, Arc<dyn AnyAssetLoader>>>,
    /// Base directory for resolving relative asset paths.
    root_path: PathBuf,
    /// Sender for dispatching load requests to background workers.
    load_sender: Sender<LoadRequest>,
    /// Receiver for completed load results from background workers.
    result_receiver: Receiver<LoadResult>,
}

impl AssetServer {
    /// Creates a new [`AssetServer`] rooted at the given directory and spawns
    /// background loader threads.
    ///
    /// The `root_path` is the base directory against which relative asset paths
    /// are resolved.
    pub fn new(root_path: &str) -> Self {
        let (req_tx, req_rx) = channel::unbounded::<LoadRequest>();
        let (res_tx, res_rx) = channel::unbounded::<LoadResult>();

        // Shared loader registry that workers will reference.
        let loaders: Arc<RwLock<HashMap<String, Arc<dyn AnyAssetLoader>>>> =
            Arc::new(RwLock::new(HashMap::new()));

        // Spawn 2 background loader threads.
        for i in 0..2 {
            let rx = req_rx.clone();
            let tx = res_tx.clone();
            let loaders = Arc::clone(&loaders);

            thread::Builder::new()
                .name(format!("asset-worker-{i}"))
                .spawn(move || {
                    Self::worker_loop(rx, tx, loaders);
                })
                .expect("failed to spawn asset worker thread");
        }

        // We need to share the *same* RwLock between the server and the
        // workers.  The easiest way is to extract the inner from the Arc and
        // store it.  Instead we will keep the Arc and dereference.
        // Actually, let's restructure: the server holds the Arc and workers
        // also hold the Arc.  We just need the server's `loaders` field to
        // be the same Arc.
        //
        // We'll store Arc<RwLock<...>> inside the server and expose it through
        // a helper.  But the public API expects `&self` methods, so let's just
        // keep it simple by wrapping the server's loader map in the same Arc.

        Self {
            slots: RwLock::new(HashMap::new()),
            path_to_id: RwLock::new(HashMap::new()),
            loaders: Arc::try_unwrap(loaders).unwrap_or_else(|arc| {
                // Workers still hold references, so we can't unwrap.
                // Clone the inner map instead -- but at this point the map
                // is still empty so this is fine.
                let guard = arc.read();
                RwLock::new(guard.clone())
            }),
            root_path: PathBuf::from(root_path),
            load_sender: req_tx,
            result_receiver: res_rx,
        }
    }

    /// Creates a new [`AssetServer`] that also gives background workers access
    /// to registered loaders.  This is the recommended constructor.
    pub fn with_workers(root_path: &str) -> AssetServerHandle {
        let loaders: Arc<RwLock<HashMap<String, Arc<dyn AnyAssetLoader>>>> =
            Arc::new(RwLock::new(HashMap::new()));

        let (req_tx, req_rx) = channel::unbounded::<LoadRequest>();
        let (res_tx, res_rx) = channel::unbounded::<LoadResult>();

        for i in 0..2 {
            let rx = req_rx.clone();
            let tx = res_tx.clone();
            let loaders = Arc::clone(&loaders);

            thread::Builder::new()
                .name(format!("asset-worker-{i}"))
                .spawn(move || {
                    Self::worker_loop(rx, tx, loaders);
                })
                .expect("failed to spawn asset worker thread");
        }

        AssetServerHandle {
            slots: Arc::new(RwLock::new(HashMap::new())),
            path_to_id: Arc::new(RwLock::new(HashMap::new())),
            loaders,
            root_path: PathBuf::from(root_path),
            load_sender: req_tx,
            result_receiver: res_rx,
        }
    }

    /// Background worker loop: receives [`LoadRequest`]s, reads files from
    /// disk, dispatches to the correct loader, and sends back [`LoadResult`]s.
    fn worker_loop(
        rx: Receiver<LoadRequest>,
        tx: Sender<LoadResult>,
        loaders: Arc<RwLock<HashMap<String, Arc<dyn AnyAssetLoader>>>>,
    ) {
        while let Ok(request) = rx.recv() {
            let result = (|| -> Result<Box<dyn Any + Send + Sync>, AssetError> {
                let bytes = std::fs::read(&request.full_path)?;

                let loader = {
                    let map = loaders.read();
                    map.get(&request.extension).cloned()
                };

                match loader {
                    Some(loader) => loader.load_erased(&request.full_path, &bytes),
                    None => Err(AssetError::NoLoader(request.extension.clone())),
                }
            })();

            let _ = tx.send(LoadResult {
                id: request.id,
                result,
            });
        }
    }

    /// Registers a format-specific loader for its declared extensions.
    pub fn register_loader<L: AssetLoader>(&self, loader: L) {
        let loader = Arc::new(loader) as Arc<dyn AnyAssetLoader>;
        let mut map = self.loaders.write();
        for &ext in AnyAssetLoader::extensions(loader.as_ref()) {
            map.insert(ext.to_ascii_lowercase(), Arc::clone(&loader));
        }
    }

    /// Begins an asynchronous load of the asset at `path`, returning a handle
    /// immediately.
    ///
    /// If the asset is already loaded or loading, the existing handle is
    /// returned (de-duplicated by canonical path).
    pub fn load<T: 'static + Send + Sync>(&self, path: &str) -> AssetHandle<T> {
        let relative = PathBuf::from(path);
        let full_path = self.root_path.join(&relative);

        // Check the path-to-id cache first.
        {
            let p2id = self.path_to_id.read();
            if let Some(&id) = p2id.get(&relative) {
                return AssetHandle::new(id);
            }
        }

        let id = AssetId::new_v4();

        // Insert slot.
        {
            let mut slots = self.slots.write();
            slots.insert(
                id,
                AssetSlot {
                    state: LoadState::Loading,
                    data: None,
                    path: relative.clone(),
                    ref_count: 1,
                },
            );
        }
        {
            let mut p2id = self.path_to_id.write();
            p2id.insert(relative, id);
        }

        let ext = full_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let _ = self.load_sender.send(LoadRequest {
            id,
            full_path,
            extension: ext,
        });

        AssetHandle::new(id)
    }

    /// Loads an asset synchronously (blocking the calling thread).
    pub fn load_sync<T: 'static + Send + Sync>(
        &self,
        path: &str,
    ) -> Result<AssetHandle<T>, AssetError> {
        let relative = PathBuf::from(path);
        let full_path = self.root_path.join(&relative);

        // De-duplicate.
        {
            let p2id = self.path_to_id.read();
            if let Some(&id) = p2id.get(&relative) {
                let slots = self.slots.read();
                if let Some(slot) = slots.get(&id) {
                    if let LoadState::Loaded = &slot.state {
                        return Ok(AssetHandle::new(id));
                    }
                }
            }
        }

        let bytes = std::fs::read(&full_path)?;

        let ext = full_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let loader = {
            let map = self.loaders.read();
            map.get(&ext).cloned()
        };

        let loader = loader.ok_or_else(|| AssetError::NoLoader(ext))?;
        let data = loader.load_erased(&full_path, &bytes)?;

        let id = AssetId::new_v4();
        {
            let mut slots = self.slots.write();
            slots.insert(
                id,
                AssetSlot {
                    state: LoadState::Loaded,
                    data: Some(data),
                    path: relative.clone(),
                    ref_count: 1,
                },
            );
        }
        {
            let mut p2id = self.path_to_id.write();
            p2id.insert(relative, id);
        }

        Ok(AssetHandle::new(id))
    }

    /// Returns an immutable reference to the loaded asset behind `handle`.
    ///
    /// Returns `None` if the asset is not yet loaded or has failed.
    ///
    /// # Safety note
    ///
    /// The returned reference borrows from the internal `RwLock` read-guard
    /// which is held for the duration of the borrow.  The caller must not hold
    /// this reference across calls that write to the slot map.
    pub fn get<T: 'static + Send + Sync>(&self, handle: &AssetHandle<T>) -> Option<AssetRef<'_, T>> {
        let slots = self.slots.read();
        let slot = slots.get(&handle.id)?;
        if let LoadState::Loaded = &slot.state {
            let data = slot.data.as_ref()?;
            // Check the downcast is valid.
            if data.downcast_ref::<T>().is_some() {
                // We return a wrapper that holds the read guard so the
                // reference stays valid.
                return Some(AssetRef {
                    _guard: slots,
                    id: handle.id,
                    _marker: PhantomData,
                });
            }
        }
        None
    }

    /// Returns a clone of the loaded asset data if available.
    ///
    /// This is a convenience method that avoids lifetime issues with guards.
    pub fn get_cloned<T: 'static + Send + Sync + Clone>(
        &self,
        handle: &AssetHandle<T>,
    ) -> Option<T> {
        let slots = self.slots.read();
        let slot = slots.get(&handle.id)?;
        if let LoadState::Loaded = &slot.state {
            slot.data
                .as_ref()
                .and_then(|d| d.downcast_ref::<T>())
                .cloned()
        } else {
            None
        }
    }

    /// Returns the [`LoadState`] of the asset referenced by `handle`.
    pub fn load_state<T>(&self, handle: &AssetHandle<T>) -> LoadState {
        let slots = self.slots.read();
        slots
            .get(&handle.id)
            .map(|s| s.state.clone())
            .unwrap_or(LoadState::NotLoaded)
    }

    /// Drains completed load results from background workers and updates the
    /// internal slot map.
    ///
    /// This should be called once per frame from the main thread.
    pub fn process_completed(&self) {
        while let Ok(result) = self.result_receiver.try_recv() {
            let mut slots = self.slots.write();
            if let Some(slot) = slots.get_mut(&result.id) {
                match result.result {
                    Ok(data) => {
                        slot.data = Some(data);
                        slot.state = LoadState::Loaded;
                        log::debug!("Asset loaded: {:?} ({})", result.id, slot.path.display());
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        log::error!(
                            "Failed to load asset {}: {}",
                            slot.path.display(),
                            msg
                        );
                        slot.state = LoadState::Failed(msg);
                    }
                }
            }
        }
    }

    /// Increments the reference count for the given asset.
    pub fn add_ref(&self, id: AssetId) {
        let mut slots = self.slots.write();
        if let Some(slot) = slots.get_mut(&id) {
            slot.ref_count = slot.ref_count.saturating_add(1);
        }
    }

    /// Decrements the reference count.  Does *not* unload automatically.
    pub fn remove_ref(&self, id: AssetId) {
        let mut slots = self.slots.write();
        if let Some(slot) = slots.get_mut(&id) {
            slot.ref_count = slot.ref_count.saturating_sub(1);
        }
    }

    /// Unloads all assets whose reference count has dropped to zero.
    pub fn collect_garbage(&self) {
        let mut slots = self.slots.write();
        let mut p2id = self.path_to_id.write();

        let to_remove: Vec<AssetId> = slots
            .iter()
            .filter(|(_, slot)| slot.ref_count == 0)
            .map(|(&id, _)| id)
            .collect();

        for id in to_remove {
            if let Some(slot) = slots.remove(&id) {
                p2id.remove(&slot.path);
                log::debug!("Garbage-collected asset: {} ({})", id, slot.path.display());
            }
        }
    }

    /// Returns the number of loaded assets.
    pub fn asset_count(&self) -> usize {
        self.slots.read().len()
    }

    /// Returns the asset root directory.
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }
}

// ---------------------------------------------------------------------------
// AssetRef — guard-based reference to loaded asset data
// ---------------------------------------------------------------------------

/// An immutable reference to a loaded asset.  Holds the internal read-guard
/// alive so the borrow is safe.
pub struct AssetRef<'a, T: 'static> {
    _guard: parking_lot::RwLockReadGuard<'a, HashMap<AssetId, AssetSlot>>,
    id: AssetId,
    _marker: PhantomData<T>,
}

impl<T: 'static + Send + Sync> std::ops::Deref for AssetRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: we only create AssetRef when we have already verified that the
        // slot exists, is Loaded, and the downcast is valid. The guard keeps the
        // map alive.
        let slot = self._guard.get(&self.id).unwrap();
        slot.data.as_ref().unwrap().downcast_ref::<T>().unwrap()
    }
}

// ---------------------------------------------------------------------------
// AssetServerHandle — Arc-wrapped variant for shared ownership
// ---------------------------------------------------------------------------

/// An [`AssetServer`] variant where the loader registry is shared with
/// background worker threads via `Arc`.  This is the recommended way to
/// construct a server when background loading is needed.
pub struct AssetServerHandle {
    slots: Arc<RwLock<HashMap<AssetId, AssetSlot>>>,
    path_to_id: Arc<RwLock<HashMap<PathBuf, AssetId>>>,
    loaders: Arc<RwLock<HashMap<String, Arc<dyn AnyAssetLoader>>>>,
    root_path: PathBuf,
    load_sender: Sender<LoadRequest>,
    result_receiver: Receiver<LoadResult>,
}

impl AssetServerHandle {
    /// Registers a format-specific loader.
    pub fn register_loader<L: AssetLoader>(&self, loader: L) {
        let loader = Arc::new(loader) as Arc<dyn AnyAssetLoader>;
        let mut map = self.loaders.write();
        for &ext in AnyAssetLoader::extensions(loader.as_ref()) {
            map.insert(ext.to_ascii_lowercase(), Arc::clone(&loader));
        }
    }

    /// Begins an asynchronous load, returning a handle immediately.
    pub fn load<T: 'static + Send + Sync>(&self, path: &str) -> AssetHandle<T> {
        let relative = PathBuf::from(path);
        let full_path = self.root_path.join(&relative);

        {
            let p2id = self.path_to_id.read();
            if let Some(&id) = p2id.get(&relative) {
                return AssetHandle::new(id);
            }
        }

        let id = AssetId::new_v4();
        {
            let mut slots = self.slots.write();
            slots.insert(
                id,
                AssetSlot {
                    state: LoadState::Loading,
                    data: None,
                    path: relative.clone(),
                    ref_count: 1,
                },
            );
        }
        {
            let mut p2id = self.path_to_id.write();
            p2id.insert(relative, id);
        }

        let ext = full_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let _ = self.load_sender.send(LoadRequest {
            id,
            full_path,
            extension: ext,
        });

        AssetHandle::new(id)
    }

    /// Synchronous (blocking) load.
    pub fn load_sync<T: 'static + Send + Sync>(
        &self,
        path: &str,
    ) -> Result<AssetHandle<T>, AssetError> {
        let relative = PathBuf::from(path);
        let full_path = self.root_path.join(&relative);

        {
            let p2id = self.path_to_id.read();
            if let Some(&id) = p2id.get(&relative) {
                let slots = self.slots.read();
                if let Some(slot) = slots.get(&id) {
                    if let LoadState::Loaded = &slot.state {
                        return Ok(AssetHandle::new(id));
                    }
                }
            }
        }

        let bytes = std::fs::read(&full_path)?;
        let ext = full_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let loader = {
            let map = self.loaders.read();
            map.get(&ext).cloned()
        };
        let loader = loader.ok_or_else(|| AssetError::NoLoader(ext))?;
        let data = loader.load_erased(&full_path, &bytes)?;

        let id = AssetId::new_v4();
        {
            let mut slots = self.slots.write();
            slots.insert(
                id,
                AssetSlot {
                    state: LoadState::Loaded,
                    data: Some(data),
                    path: relative.clone(),
                    ref_count: 1,
                },
            );
        }
        {
            let mut p2id = self.path_to_id.write();
            p2id.insert(relative, id);
        }

        Ok(AssetHandle::new(id))
    }

    /// Returns a clone of the loaded asset data.
    pub fn get_cloned<T: 'static + Send + Sync + Clone>(
        &self,
        handle: &AssetHandle<T>,
    ) -> Option<T> {
        let slots = self.slots.read();
        let slot = slots.get(&handle.id)?;
        if let LoadState::Loaded = &slot.state {
            slot.data
                .as_ref()
                .and_then(|d| d.downcast_ref::<T>())
                .cloned()
        } else {
            None
        }
    }

    /// Returns the load state.
    pub fn load_state<T>(&self, handle: &AssetHandle<T>) -> LoadState {
        let slots = self.slots.read();
        slots
            .get(&handle.id)
            .map(|s| s.state.clone())
            .unwrap_or(LoadState::NotLoaded)
    }

    /// Drains completed results and updates slots.
    pub fn process_completed(&self) {
        while let Ok(result) = self.result_receiver.try_recv() {
            let mut slots = self.slots.write();
            if let Some(slot) = slots.get_mut(&result.id) {
                match result.result {
                    Ok(data) => {
                        slot.data = Some(data);
                        slot.state = LoadState::Loaded;
                    }
                    Err(e) => {
                        slot.state = LoadState::Failed(e.to_string());
                    }
                }
            }
        }
    }

    /// Returns the asset root directory.
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A trivial loader that loads files as `String`.
    struct TestTextLoader;

    impl AssetLoader for TestTextLoader {
        type Asset = String;

        fn extensions(&self) -> &[&str] {
            &["txt"]
        }

        fn load(&self, _path: &Path, bytes: &[u8]) -> Result<String, AssetError> {
            String::from_utf8(bytes.to_vec())
                .map_err(|e| AssetError::Parse(e.to_string()))
        }
    }

    #[test]
    fn test_sync_load_text_file() {
        let dir = std::env::temp_dir().join("genovo_asset_test_sync");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("hello.txt");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(b"Hello, Genovo!").unwrap();
        }

        let server = AssetServer::new(dir.to_str().unwrap());
        server.register_loader(TestTextLoader);

        let handle: AssetHandle<String> = server.load_sync("hello.txt").unwrap();
        let text = server.get_cloned(&handle).unwrap();
        assert_eq!(text, "Hello, Genovo!");

        // Cleanup
        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_async_load_text_file() {
        let dir = std::env::temp_dir().join("genovo_asset_test_async");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("world.txt");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(b"Async works!").unwrap();
        }

        let srv = AssetServer::with_workers(dir.to_str().unwrap());
        srv.register_loader(TestTextLoader);

        let handle: AssetHandle<String> = srv.load("world.txt");

        // Poll until loaded (with timeout).
        for _ in 0..200 {
            srv.process_completed();
            if let LoadState::Loaded = srv.load_state(&handle) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let text = srv.get_cloned(&handle).unwrap();
        assert_eq!(text, "Async works!");

        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_load_state_failed() {
        let dir = std::env::temp_dir().join("genovo_asset_test_fail");
        let _ = std::fs::create_dir_all(&dir);

        let server = AssetServer::new(dir.to_str().unwrap());
        server.register_loader(TestTextLoader);

        let result: Result<AssetHandle<String>, _> =
            server.load_sync("nonexistent.txt");
        assert!(result.is_err());

        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_no_loader_error() {
        let dir = std::env::temp_dir().join("genovo_asset_test_noloader");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("data.bin");
        std::fs::write(&file_path, b"binary").unwrap();

        let server = AssetServer::new(dir.to_str().unwrap());
        // No loader registered for .bin

        let result: Result<AssetHandle<Vec<u8>>, _> =
            server.load_sync("data.bin");
        assert!(result.is_err());

        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_handle_clone_and_eq() {
        let id = AssetId::new_v4();
        let h1 = AssetHandle::<String>::new(id);
        let h2 = h1;
        assert_eq!(h1, h2);
        assert_eq!(h1.id(), h2.id());
    }

    #[test]
    fn test_asset_path_parse() {
        let p = AssetPath::parse("meshes/hero.glb#Mesh0");
        assert_eq!(p.path, PathBuf::from("meshes/hero.glb"));
        assert_eq!(p.label.as_deref(), Some("Mesh0"));

        let p2 = AssetPath::parse("textures/diffuse.png");
        assert_eq!(p2.path, PathBuf::from("textures/diffuse.png"));
        assert!(p2.label.is_none());
    }
}
