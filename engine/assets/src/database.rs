//! Asset database for tracking metadata, dependencies, and import settings.
//!
//! The [`AssetDatabase`] serves as the source of truth for all known assets
//! in a project.  It is persisted to disk as JSON and updated incrementally
//! as assets are imported, moved, or deleted.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use genovo_core::{EngineError, EngineResult};

// ---------------------------------------------------------------------------
// AssetMeta
// ---------------------------------------------------------------------------

/// Metadata for a single asset, stored alongside the source file (`.meta`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetMeta {
    /// Stable unique identifier (persists across renames/moves).
    pub uuid: Uuid,
    /// Human-readable type name (e.g. `"Texture"`, `"Mesh"`, `"AudioClip"`).
    pub type_name: String,
    /// Path relative to the asset root.
    pub path: PathBuf,
    /// UUIDs of assets that this asset depends on.
    pub dependencies: Vec<Uuid>,
    /// Importer-specific settings serialised as key-value pairs.
    pub import_settings: HashMap<String, serde_json::Value>,
    /// Timestamp of the last successful import (Unix epoch seconds).
    pub last_imported: u64,
    /// Content hash of the source file at last import time.
    pub source_hash: u64,
}

impl AssetMeta {
    /// Creates new metadata with a fresh UUID.
    pub fn new(type_name: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            type_name: type_name.into(),
            path: path.into(),
            dependencies: Vec::new(),
            import_settings: HashMap::new(),
            last_imported: 0,
            source_hash: 0,
        }
    }

    /// Creates metadata with a specific UUID (for testing or migration).
    pub fn with_uuid(
        uuid: Uuid,
        type_name: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            uuid,
            type_name: type_name.into(),
            path: path.into(),
            dependencies: Vec::new(),
            import_settings: HashMap::new(),
            last_imported: 0,
            source_hash: 0,
        }
    }

    /// Returns `true` if the source file has changed since last import.
    pub fn is_stale(&self, current_source_hash: u64) -> bool {
        self.source_hash != current_source_hash
    }
}

// ---------------------------------------------------------------------------
// Serialised form of the database (for JSON persistence)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct DatabaseFile {
    version: u32,
    assets: Vec<AssetMeta>,
}

// ---------------------------------------------------------------------------
// AssetDatabase
// ---------------------------------------------------------------------------

/// Central registry tracking all known assets, their metadata, and relationships.
///
/// The database supports:
/// - Lookup by UUID, path, or type
/// - Dependency tracking and reverse-dependency queries
/// - Incremental updates when files change on disk
/// - Persistence to / from a JSON file
pub struct AssetDatabase {
    /// All known asset metadata, keyed by UUID.
    assets_by_uuid: HashMap<Uuid, AssetMeta>,
    /// Index from relative path to UUID for fast path-based lookups.
    path_to_uuid: HashMap<PathBuf, Uuid>,
    /// Index from asset type to set of UUIDs.
    type_to_uuids: HashMap<String, Vec<Uuid>>,
    /// Reverse dependency index: asset UUID -> list of assets that depend on it.
    reverse_deps: HashMap<Uuid, Vec<Uuid>>,
    /// Root directory of the asset tree.
    root: PathBuf,
    /// Whether the database has unsaved changes.
    dirty: bool,
}

impl AssetDatabase {
    /// Creates or opens an asset database rooted at the given directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            assets_by_uuid: HashMap::new(),
            path_to_uuid: HashMap::new(),
            type_to_uuids: HashMap::new(),
            reverse_deps: HashMap::new(),
            root: root.into(),
            dirty: false,
        }
    }

    /// Registers a new asset or updates an existing one.
    pub fn register(&mut self, meta: AssetMeta) {
        let uuid = meta.uuid;
        let path = meta.path.clone();
        let type_name = meta.type_name.clone();

        // If we are updating, remove old indices.
        if let Some(old) = self.assets_by_uuid.get(&uuid) {
            self.path_to_uuid.remove(&old.path);
            if let Some(uuids) = self.type_to_uuids.get_mut(&old.type_name) {
                uuids.retain(|id| *id != uuid);
            }
            for dep in &old.dependencies {
                if let Some(rdeps) = self.reverse_deps.get_mut(dep) {
                    rdeps.retain(|id| *id != uuid);
                }
            }
        }

        // Update reverse dependency index.
        for dep in &meta.dependencies {
            self.reverse_deps.entry(*dep).or_default().push(uuid);
        }

        // Update type index.
        self.type_to_uuids
            .entry(type_name)
            .or_default()
            .push(uuid);

        // Update path index.
        self.path_to_uuid.insert(path, uuid);

        // Store metadata.
        self.assets_by_uuid.insert(uuid, meta);
        self.dirty = true;
    }

    /// Removes an asset from the database by UUID.
    pub fn remove(&mut self, uuid: &Uuid) -> Option<AssetMeta> {
        if let Some(meta) = self.assets_by_uuid.remove(uuid) {
            self.path_to_uuid.remove(&meta.path);

            if let Some(uuids) = self.type_to_uuids.get_mut(&meta.type_name) {
                uuids.retain(|id| id != uuid);
            }

            // Clean reverse deps.
            for dep in &meta.dependencies {
                if let Some(rdeps) = self.reverse_deps.get_mut(dep) {
                    rdeps.retain(|id| id != uuid);
                }
            }
            self.reverse_deps.remove(uuid);

            self.dirty = true;
            Some(meta)
        } else {
            None
        }
    }

    /// Looks up an asset by its UUID.
    pub fn get_by_uuid(&self, uuid: &Uuid) -> Option<&AssetMeta> {
        self.assets_by_uuid.get(uuid)
    }

    /// Looks up an asset by its relative path.
    pub fn get_by_path(&self, path: &Path) -> Option<&AssetMeta> {
        self.path_to_uuid
            .get(path)
            .and_then(|uuid| self.assets_by_uuid.get(uuid))
    }

    /// Returns all assets of the given type.
    pub fn get_by_type(&self, type_name: &str) -> Vec<&AssetMeta> {
        self.type_to_uuids
            .get(type_name)
            .map(|uuids| {
                uuids
                    .iter()
                    .filter_map(|uuid| self.assets_by_uuid.get(uuid))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the UUIDs of all assets that depend on the given asset.
    pub fn reverse_dependencies(&self, uuid: &Uuid) -> &[Uuid] {
        self.reverse_deps
            .get(uuid)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns all registered asset metadata.
    pub fn all_assets(&self) -> impl Iterator<Item = &AssetMeta> {
        self.assets_by_uuid.values()
    }

    /// Returns the total number of registered assets.
    pub fn len(&self) -> usize {
        self.assets_by_uuid.len()
    }

    /// Returns `true` if the database contains no assets.
    pub fn is_empty(&self) -> bool {
        self.assets_by_uuid.is_empty()
    }

    /// Returns `true` if the database has unsaved changes.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Returns the root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Scans the asset root and updates the database with any new or changed files.
    ///
    /// New files get a fresh UUID.  Changed files (by hash) are marked stale.
    /// Deleted files are removed from the database.
    pub fn refresh(&mut self) -> EngineResult<()> {
        let root = self.root.clone();
        if !root.is_dir() {
            return Ok(());
        }

        // Collect all files on disk.
        let mut disk_paths: Vec<PathBuf> = Vec::new();
        collect_all_files(&root, &root, &mut disk_paths)?;

        let known_paths: std::collections::HashSet<PathBuf> =
            self.path_to_uuid.keys().cloned().collect();

        // Register new files.
        for path in &disk_paths {
            if !known_paths.contains(path) {
                let type_name = guess_type(path);
                let mut meta = AssetMeta::new(type_name, path.clone());
                // Compute source hash.
                let full = root.join(path);
                if let Ok(data) = std::fs::read(&full) {
                    meta.source_hash = fnv_hash(&data);
                }
                log::info!("New asset discovered: {}", path.display());
                self.register(meta);
            }
        }

        // Remove deleted files.
        let disk_set: std::collections::HashSet<&PathBuf> = disk_paths.iter().collect();
        let removed: Vec<Uuid> = self
            .path_to_uuid
            .iter()
            .filter(|(p, _)| !disk_set.contains(p))
            .map(|(_, &uuid)| uuid)
            .collect();

        for uuid in removed {
            log::info!("Removing deleted asset: {uuid}");
            self.remove(&uuid);
        }

        Ok(())
    }

    /// Persists the database to disk as `asset_db.json` under the root.
    pub fn save(&mut self) -> EngineResult<()> {
        let db_path = self.root.join("asset_db.json");
        let file = DatabaseFile {
            version: 1,
            assets: self.assets_by_uuid.values().cloned().collect(),
        };
        let json =
            serde_json::to_string_pretty(&file).map_err(|e| EngineError::Other(e.to_string()))?;
        std::fs::write(&db_path, json)?;
        self.dirty = false;
        log::info!("Saved asset database to {}", db_path.display());
        Ok(())
    }

    /// Loads the database from disk (`asset_db.json` under `root`).
    pub fn load_from_disk(root: impl Into<PathBuf>) -> EngineResult<Self> {
        let root = root.into();
        let db_path = root.join("asset_db.json");

        if !db_path.exists() {
            log::info!(
                "No asset database found at {}, creating empty",
                db_path.display()
            );
            return Ok(Self::new(root));
        }

        let json = std::fs::read_to_string(&db_path)?;
        let file: DatabaseFile =
            serde_json::from_str(&json).map_err(|e| EngineError::Other(e.to_string()))?;

        let mut db = Self::new(root);
        for meta in file.assets {
            db.register(meta);
        }
        db.dirty = false;

        log::info!(
            "Loaded asset database from {} ({} assets)",
            db_path.display(),
            db.len()
        );
        Ok(db)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Recursively collect file paths relative to `root`.
fn collect_all_files(
    dir: &Path,
    root: &Path,
    out: &mut Vec<PathBuf>,
) -> EngineResult<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            // Skip the asset_db.json and hidden directories.
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with('.') {
                continue;
            }
            collect_all_files(&path, root, out)?;
        } else if ft.is_file() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            // Skip database file and meta files.
            if name_str == "asset_db.json" || name_str.ends_with(".meta") {
                continue;
            }
            if let Ok(relative) = path.strip_prefix(root) {
                out.push(relative.to_path_buf());
            }
        }
    }
    Ok(())
}

/// Guess asset type from file extension.
fn guess_type(path: &Path) -> String {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "bmp" | "png" | "jpg" | "jpeg" | "tga" | "hdr" | "dds" | "ktx2" => "Texture".to_owned(),
        "obj" | "fbx" | "gltf" | "glb" => "Mesh".to_owned(),
        "wav" | "ogg" | "mp3" | "flac" => "AudioClip".to_owned(),
        "spv" | "hlsl" | "glsl" | "wgsl" | "metallib" => "Shader".to_owned(),
        "txt" | "json" | "ron" | "toml" | "yaml" | "yml" | "xml" => "Text".to_owned(),
        _ => "Unknown".to_owned(),
    }
}

/// FNV-1a hash for content fingerprinting.
fn fnv_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_lookup() {
        let mut db = AssetDatabase::new("/tmp");
        let meta = AssetMeta::new("Texture", "textures/grass.bmp");
        let uuid = meta.uuid;
        db.register(meta);

        assert_eq!(db.len(), 1);
        assert!(!db.is_empty());

        let found = db.get_by_uuid(&uuid).unwrap();
        assert_eq!(found.type_name, "Texture");
        assert_eq!(found.path, PathBuf::from("textures/grass.bmp"));

        let by_path = db.get_by_path(Path::new("textures/grass.bmp")).unwrap();
        assert_eq!(by_path.uuid, uuid);

        let by_type = db.get_by_type("Texture");
        assert_eq!(by_type.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut db = AssetDatabase::new("/tmp");
        let meta = AssetMeta::new("Mesh", "meshes/cube.obj");
        let uuid = meta.uuid;
        db.register(meta);
        assert_eq!(db.len(), 1);

        let removed = db.remove(&uuid).unwrap();
        assert_eq!(removed.type_name, "Mesh");
        assert_eq!(db.len(), 0);
        assert!(db.get_by_uuid(&uuid).is_none());
        assert!(db.get_by_path(Path::new("meshes/cube.obj")).is_none());
    }

    #[test]
    fn test_dependencies() {
        let mut db = AssetDatabase::new("/tmp");

        let tex = AssetMeta::new("Texture", "tex.bmp");
        let tex_id = tex.uuid;
        db.register(tex);

        let mut mat = AssetMeta::new("Material", "mat.json");
        mat.dependencies.push(tex_id);
        let mat_id = mat.uuid;
        db.register(mat);

        let rdeps = db.reverse_dependencies(&tex_id);
        assert_eq!(rdeps.len(), 1);
        assert_eq!(rdeps[0], mat_id);
    }

    #[test]
    fn test_save_and_load() {
        let dir = std::env::temp_dir().join("genovo_db_test_save");
        let _ = std::fs::create_dir_all(&dir);

        {
            let mut db = AssetDatabase::new(&dir);
            let m1 = AssetMeta::new("Texture", "a.bmp");
            let m2 = AssetMeta::new("Mesh", "b.obj");
            db.register(m1);
            db.register(m2);
            db.save().unwrap();
        }

        // Reload.
        let db = AssetDatabase::load_from_disk(&dir).unwrap();
        assert_eq!(db.len(), 2);
        assert!(db.get_by_path(Path::new("a.bmp")).is_some());
        assert!(db.get_by_path(Path::new("b.obj")).is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_from_empty() {
        let dir = std::env::temp_dir().join("genovo_db_test_empty");
        let _ = std::fs::create_dir_all(&dir);
        // No asset_db.json present.
        let db = AssetDatabase::load_from_disk(&dir).unwrap();
        assert!(db.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_refresh_discovers_new_files() {
        let dir = std::env::temp_dir().join("genovo_db_test_refresh");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("new_asset.txt"), "hello").unwrap();

        let mut db = AssetDatabase::new(&dir);
        db.refresh().unwrap();

        assert_eq!(db.len(), 1);
        let meta = db.get_by_path(Path::new("new_asset.txt")).unwrap();
        assert_eq!(meta.type_name, "Text");
        assert_ne!(meta.source_hash, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_refresh_removes_deleted_files() {
        let dir = std::env::temp_dir().join("genovo_db_test_refresh_del");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("temp.txt"), "gone soon").unwrap();

        let mut db = AssetDatabase::new(&dir);
        db.refresh().unwrap();
        assert_eq!(db.len(), 1);

        // Delete the file.
        std::fs::remove_file(dir.join("temp.txt")).unwrap();
        db.refresh().unwrap();
        assert_eq!(db.len(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_update_existing() {
        let mut db = AssetDatabase::new("/tmp");
        let uuid = Uuid::new_v4();

        let m1 = AssetMeta::with_uuid(uuid, "Texture", "old.bmp");
        db.register(m1);
        assert_eq!(db.get_by_path(Path::new("old.bmp")).unwrap().uuid, uuid);

        // Update path and type.
        let m2 = AssetMeta::with_uuid(uuid, "HDRTexture", "new.hdr");
        db.register(m2);
        assert!(db.get_by_path(Path::new("old.bmp")).is_none());
        assert_eq!(db.get_by_path(Path::new("new.hdr")).unwrap().uuid, uuid);
        assert_eq!(db.get_by_type("HDRTexture").len(), 1);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_guess_type() {
        assert_eq!(guess_type(Path::new("foo.bmp")), "Texture");
        assert_eq!(guess_type(Path::new("bar.obj")), "Mesh");
        assert_eq!(guess_type(Path::new("sfx.wav")), "AudioClip");
        assert_eq!(guess_type(Path::new("vert.spv")), "Shader");
        assert_eq!(guess_type(Path::new("cfg.toml")), "Text");
        assert_eq!(guess_type(Path::new("data.xyz")), "Unknown");
    }

    #[test]
    fn test_meta_staleness() {
        let mut meta = AssetMeta::new("Texture", "a.bmp");
        meta.source_hash = 12345;
        assert!(!meta.is_stale(12345));
        assert!(meta.is_stale(99999));
    }
}
