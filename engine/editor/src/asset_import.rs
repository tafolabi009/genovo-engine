// engine/editor/src/asset_import.rs
//
// Asset import pipeline UI for the Genovo editor.
// Drag-drop import, import settings dialogs, format conversion,
// thumbnail generation, import progress, import history, batch import.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImportJobId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetFormat { Gltf, Fbx, Obj, Png, Jpg, Tga, Hdr, Exr, Wav, Ogg, Mp3, Flac, Ttf, Otf, Json, Toml, Custom }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportState { Pending, InProgress, Completed, Failed, Cancelled }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureCompression { None, Bc1, Bc3, Bc5, Bc7, Astc4x4, Astc8x8, Etc2 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshImportOption { KeepOriginal, Optimize, Simplify, GenerateLODs }

#[derive(Debug, Clone)]
pub struct TextureImportSettings {
    pub compression: TextureCompression,
    pub generate_mipmaps: bool,
    pub max_resolution: u32,
    pub srgb: bool,
    pub flip_y: bool,
    pub premultiply_alpha: bool,
    pub normal_map: bool,
}

impl Default for TextureImportSettings {
    fn default() -> Self { Self { compression: TextureCompression::Bc7, generate_mipmaps: true, max_resolution: 4096, srgb: true, flip_y: false, premultiply_alpha: false, normal_map: false } }
}

#[derive(Debug, Clone)]
pub struct MeshImportSettings {
    pub import_option: MeshImportOption,
    pub scale_factor: f32,
    pub import_animations: bool,
    pub import_materials: bool,
    pub import_textures: bool,
    pub generate_tangents: bool,
    pub merge_meshes: bool,
    pub lod_levels: u32,
    pub simplification_target: f32,
    pub recalculate_normals: bool,
    pub up_axis: UpAxis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpAxis { Y, Z }

impl Default for MeshImportSettings {
    fn default() -> Self { Self { import_option: MeshImportOption::Optimize, scale_factor: 1.0, import_animations: true, import_materials: true, import_textures: true, generate_tangents: true, merge_meshes: false, lod_levels: 3, simplification_target: 0.5, recalculate_normals: false, up_axis: UpAxis::Y } }
}

#[derive(Debug, Clone)]
pub struct AudioImportSettings {
    pub sample_rate: u32,
    pub channels: u32,
    pub compress: bool,
    pub quality: f32,
    pub streaming: bool,
    pub loop_point: Option<f64>,
}

impl Default for AudioImportSettings {
    fn default() -> Self { Self { sample_rate: 44100, channels: 2, compress: true, quality: 0.7, streaming: false, loop_point: None } }
}

#[derive(Debug, Clone)]
pub enum ImportSettings {
    Texture(TextureImportSettings),
    Mesh(MeshImportSettings),
    Audio(AudioImportSettings),
    Generic,
}

#[derive(Debug, Clone)]
pub struct ImportJob {
    pub id: ImportJobId,
    pub source_path: String,
    pub destination_path: String,
    pub format: AssetFormat,
    pub state: ImportState,
    pub settings: ImportSettings,
    pub progress: f32,
    pub error_message: Option<String>,
    pub file_size_bytes: u64,
    pub import_time_ms: f64,
    pub thumbnail_generated: bool,
    pub timestamp: f64,
}

impl ImportJob {
    pub fn new(id: ImportJobId, source: &str, dest: &str, format: AssetFormat) -> Self {
        let settings = match format {
            AssetFormat::Png | AssetFormat::Jpg | AssetFormat::Tga | AssetFormat::Hdr | AssetFormat::Exr => ImportSettings::Texture(TextureImportSettings::default()),
            AssetFormat::Gltf | AssetFormat::Fbx | AssetFormat::Obj => ImportSettings::Mesh(MeshImportSettings::default()),
            AssetFormat::Wav | AssetFormat::Ogg | AssetFormat::Mp3 | AssetFormat::Flac => ImportSettings::Audio(AudioImportSettings::default()),
            _ => ImportSettings::Generic,
        };
        Self { id, source_path: source.to_string(), destination_path: dest.to_string(), format, state: ImportState::Pending, settings, progress: 0.0, error_message: None, file_size_bytes: 0, import_time_ms: 0.0, thumbnail_generated: false, timestamp: 0.0 }
    }
    pub fn is_complete(&self) -> bool { self.state == ImportState::Completed }
    pub fn is_failed(&self) -> bool { self.state == ImportState::Failed }
    pub fn filename(&self) -> &str { self.source_path.rsplit('/').next().unwrap_or(&self.source_path) }
}

#[derive(Debug, Clone)]
pub struct ImportHistoryEntry {
    pub job_id: ImportJobId,
    pub source_path: String,
    pub destination_path: String,
    pub format: AssetFormat,
    pub success: bool,
    pub import_time_ms: f64,
    pub timestamp: f64,
    pub file_size: u64,
}

#[derive(Debug)]
pub struct AssetImportPipeline {
    pub active_jobs: HashMap<ImportJobId, ImportJob>,
    pub pending_jobs: Vec<ImportJobId>,
    pub completed_jobs: Vec<ImportJobId>,
    pub history: Vec<ImportHistoryEntry>,
    pub max_concurrent: u32,
    pub current_concurrent: u32,
    pub auto_import_enabled: bool,
    pub default_texture_settings: TextureImportSettings,
    pub default_mesh_settings: MeshImportSettings,
    pub default_audio_settings: AudioImportSettings,
    pub supported_formats: Vec<AssetFormat>,
    next_id: u64,
}

impl AssetImportPipeline {
    pub fn new() -> Self {
        Self {
            active_jobs: HashMap::new(), pending_jobs: Vec::new(),
            completed_jobs: Vec::new(), history: Vec::new(),
            max_concurrent: 4, current_concurrent: 0,
            auto_import_enabled: true,
            default_texture_settings: TextureImportSettings::default(),
            default_mesh_settings: MeshImportSettings::default(),
            default_audio_settings: AudioImportSettings::default(),
            supported_formats: vec![AssetFormat::Gltf, AssetFormat::Fbx, AssetFormat::Obj, AssetFormat::Png, AssetFormat::Jpg, AssetFormat::Tga, AssetFormat::Hdr, AssetFormat::Wav, AssetFormat::Ogg],
            next_id: 1,
        }
    }

    pub fn import_file(&mut self, source: &str, dest: &str, format: AssetFormat) -> ImportJobId {
        let id = ImportJobId(self.next_id); self.next_id += 1;
        let job = ImportJob::new(id, source, dest, format);
        self.active_jobs.insert(id, job);
        self.pending_jobs.push(id);
        id
    }

    pub fn batch_import(&mut self, files: &[(&str, &str, AssetFormat)]) -> Vec<ImportJobId> {
        files.iter().map(|(src, dst, fmt)| self.import_file(src, dst, *fmt)).collect()
    }

    pub fn cancel_job(&mut self, id: ImportJobId) {
        if let Some(job) = self.active_jobs.get_mut(&id) { job.state = ImportState::Cancelled; }
        self.pending_jobs.retain(|j| *j != id);
    }

    pub fn update(&mut self) {
        // Process completed jobs
        let completed: Vec<_> = self.active_jobs.iter().filter(|(_, j)| j.is_complete() || j.is_failed()).map(|(id, _)| *id).collect();
        for id in completed {
            if let Some(job) = self.active_jobs.remove(&id) {
                self.history.push(ImportHistoryEntry {
                    job_id: id, source_path: job.source_path.clone(),
                    destination_path: job.destination_path.clone(),
                    format: job.format, success: job.is_complete(),
                    import_time_ms: job.import_time_ms,
                    timestamp: job.timestamp, file_size: job.file_size_bytes,
                });
                self.completed_jobs.push(id);
                self.current_concurrent = self.current_concurrent.saturating_sub(1);
            }
        }
    }

    pub fn get_job(&self, id: ImportJobId) -> Option<&ImportJob> { self.active_jobs.get(&id) }
    pub fn pending_count(&self) -> usize { self.pending_jobs.len() }
    pub fn active_count(&self) -> usize { self.active_jobs.len() }
    pub fn history_count(&self) -> usize { self.history.len() }
    pub fn total_imported(&self) -> usize { self.history.iter().filter(|h| h.success).count() }

    pub fn detect_format(path: &str) -> Option<AssetFormat> {
        let ext = path.rsplit('.').next()?.to_lowercase();
        match ext.as_str() {
            "gltf" | "glb" => Some(AssetFormat::Gltf), "fbx" => Some(AssetFormat::Fbx), "obj" => Some(AssetFormat::Obj),
            "png" => Some(AssetFormat::Png), "jpg" | "jpeg" => Some(AssetFormat::Jpg), "tga" => Some(AssetFormat::Tga),
            "hdr" => Some(AssetFormat::Hdr), "exr" => Some(AssetFormat::Exr),
            "wav" => Some(AssetFormat::Wav), "ogg" => Some(AssetFormat::Ogg), "mp3" => Some(AssetFormat::Mp3),
            "ttf" => Some(AssetFormat::Ttf), "otf" => Some(AssetFormat::Otf),
            "json" => Some(AssetFormat::Json), "toml" => Some(AssetFormat::Toml),
            _ => None,
        }
    }
}

impl Default for AssetImportPipeline { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_import_pipeline() {
        let mut pipeline = AssetImportPipeline::new();
        let id = pipeline.import_file("/models/char.gltf", "/assets/char.bin", AssetFormat::Gltf);
        assert_eq!(pipeline.active_count(), 1);
        assert!(pipeline.get_job(id).is_some());
    }
    #[test]
    fn test_format_detection() {
        assert_eq!(AssetImportPipeline::detect_format("model.gltf"), Some(AssetFormat::Gltf));
        assert_eq!(AssetImportPipeline::detect_format("texture.png"), Some(AssetFormat::Png));
        assert_eq!(AssetImportPipeline::detect_format("sound.wav"), Some(AssetFormat::Wav));
    }
    #[test]
    fn test_batch_import() {
        let mut pipeline = AssetImportPipeline::new();
        let ids = pipeline.batch_import(&[("a.png", "a.bin", AssetFormat::Png), ("b.obj", "b.bin", AssetFormat::Obj)]);
        assert_eq!(ids.len(), 2);
        assert_eq!(pipeline.pending_count(), 2);
    }
}
