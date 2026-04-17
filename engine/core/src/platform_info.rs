// engine/core/src/platform_info.rs
//
// Platform detection: OS name/version, CPU model/cores/features, GPU info
// (from adapter), RAM total/available, display info (resolution/DPI/refresh
// rate), storage info, build configuration (debug/release).

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// OS information
// ---------------------------------------------------------------------------

/// Operating system family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OsFamily {
    Windows,
    Linux,
    MacOS,
    IOS,
    Android,
    WebAssembly,
    Unknown,
}

impl OsFamily {
    pub fn current() -> Self {
        if cfg!(target_os = "windows") { Self::Windows }
        else if cfg!(target_os = "linux") { Self::Linux }
        else if cfg!(target_os = "macos") { Self::MacOS }
        else if cfg!(target_os = "ios") { Self::IOS }
        else if cfg!(target_os = "android") { Self::Android }
        else if cfg!(target_arch = "wasm32") { Self::WebAssembly }
        else { Self::Unknown }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Windows => "Windows",
            Self::Linux => "Linux",
            Self::MacOS => "macOS",
            Self::IOS => "iOS",
            Self::Android => "Android",
            Self::WebAssembly => "WebAssembly",
            Self::Unknown => "Unknown",
        }
    }

    pub fn is_desktop(&self) -> bool {
        matches!(self, Self::Windows | Self::Linux | Self::MacOS)
    }

    pub fn is_mobile(&self) -> bool {
        matches!(self, Self::IOS | Self::Android)
    }
}

/// Detailed OS information.
#[derive(Debug, Clone)]
pub struct OsInfo {
    pub family: OsFamily,
    pub name: String,
    pub version: String,
    pub build: String,
    pub arch: String,
    pub hostname: String,
    pub username: String,
    pub locale: String,
}

impl OsInfo {
    /// Detect current OS information.
    pub fn detect() -> Self {
        Self {
            family: OsFamily::current(),
            name: std::env::consts::OS.to_string(),
            version: String::new(), // Would use OS-specific APIs.
            build: String::new(),
            arch: std::env::consts::ARCH.to_string(),
            hostname: std::env::var("COMPUTERNAME")
                .or_else(|_| std::env::var("HOSTNAME"))
                .unwrap_or_default(),
            username: std::env::var("USERNAME")
                .or_else(|_| std::env::var("USER"))
                .unwrap_or_default(),
            locale: std::env::var("LANG")
                .or_else(|_| std::env::var("LANGUAGE"))
                .unwrap_or_else(|_| "en_US".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// CPU information
// ---------------------------------------------------------------------------

/// CPU feature flags.
#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    pub neon: bool,
    pub aes: bool,
    pub bmi1: bool,
    pub bmi2: bool,
    pub popcnt: bool,
    pub f16c: bool,
}

impl CpuFeatures {
    /// Detect CPU features for the current platform.
    pub fn detect() -> Self {
        Self {
            sse: cfg!(target_feature = "sse"),
            sse2: cfg!(target_feature = "sse2"),
            sse3: false, // Would use CPUID.
            ssse3: false,
            sse41: false,
            sse42: false,
            avx: false,
            avx2: false,
            avx512f: false,
            fma: false,
            neon: cfg!(target_arch = "aarch64"),
            aes: false,
            bmi1: false,
            bmi2: false,
            popcnt: false,
            f16c: false,
        }
    }

    /// Return a list of supported feature names.
    pub fn feature_list(&self) -> Vec<&'static str> {
        let mut features = Vec::new();
        if self.sse { features.push("SSE"); }
        if self.sse2 { features.push("SSE2"); }
        if self.sse3 { features.push("SSE3"); }
        if self.ssse3 { features.push("SSSE3"); }
        if self.sse41 { features.push("SSE4.1"); }
        if self.sse42 { features.push("SSE4.2"); }
        if self.avx { features.push("AVX"); }
        if self.avx2 { features.push("AVX2"); }
        if self.avx512f { features.push("AVX-512"); }
        if self.fma { features.push("FMA"); }
        if self.neon { features.push("NEON"); }
        if self.aes { features.push("AES-NI"); }
        if self.popcnt { features.push("POPCNT"); }
        features
    }
}

/// CPU information.
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub model: String,
    pub vendor: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub base_frequency_mhz: u32,
    pub max_frequency_mhz: u32,
    pub cache_line_size: u32,
    pub l1_cache_kb: u32,
    pub l2_cache_kb: u32,
    pub l3_cache_mb: u32,
    pub features: CpuFeatures,
    pub arch: CpuArch,
    pub endianness: Endianness,
    pub pointer_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArch {
    X86,
    X86_64,
    Arm,
    Aarch64,
    Wasm32,
    Unknown,
}

impl CpuArch {
    pub fn current() -> Self {
        if cfg!(target_arch = "x86_64") { Self::X86_64 }
        else if cfg!(target_arch = "x86") { Self::X86 }
        else if cfg!(target_arch = "aarch64") { Self::Aarch64 }
        else if cfg!(target_arch = "arm") { Self::Arm }
        else if cfg!(target_arch = "wasm32") { Self::Wasm32 }
        else { Self::Unknown }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Little,
    Big,
}

impl CpuInfo {
    pub fn detect() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(|p| p.get() as u32)
            .unwrap_or(1);

        Self {
            model: "Unknown CPU".to_string(),
            vendor: "Unknown".to_string(),
            physical_cores: logical_cores / 2,
            logical_cores,
            base_frequency_mhz: 0,
            max_frequency_mhz: 0,
            cache_line_size: 64,
            l1_cache_kb: 32,
            l2_cache_kb: 256,
            l3_cache_mb: 0,
            features: CpuFeatures::detect(),
            arch: CpuArch::current(),
            endianness: if cfg!(target_endian = "little") { Endianness::Little } else { Endianness::Big },
            pointer_size: std::mem::size_of::<usize>() as u32 * 8,
        }
    }
}

// ---------------------------------------------------------------------------
// GPU information
// ---------------------------------------------------------------------------

/// GPU vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Qualcomm,
    Arm,
    ImgTec,
    Samsung,
    Unknown,
}

impl GpuVendor {
    pub fn from_vendor_id(id: u32) -> Self {
        match id {
            0x10DE => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            0x106B => Self::Apple,
            0x5143 => Self::Qualcomm,
            0x13B5 => Self::Arm,
            0x1010 => Self::ImgTec,
            0x144D => Self::Samsung,
            _ => Self::Unknown,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Apple => "Apple",
            Self::Qualcomm => "Qualcomm",
            Self::Arm => "ARM",
            Self::ImgTec => "Imagination Technologies",
            Self::Samsung => "Samsung",
            Self::Unknown => "Unknown",
        }
    }
}

/// GPU device type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    Discrete,
    Integrated,
    Virtual,
    Software,
    Unknown,
}

/// GPU information.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: GpuVendor,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: GpuDeviceType,
    pub driver_version: String,
    pub api_version: String,
    pub vram_total_mb: u64,
    pub vram_available_mb: u64,
    pub shared_memory_mb: u64,
    pub max_texture_size: u32,
    pub max_compute_workgroup_size: [u32; 3],
    pub supports_raytracing: bool,
    pub supports_mesh_shaders: bool,
    pub supports_variable_rate_shading: bool,
    pub max_msaa_samples: u32,
    pub supported_apis: Vec<String>,
}

impl GpuInfo {
    pub fn unknown() -> Self {
        Self {
            name: "Unknown GPU".to_string(),
            vendor: GpuVendor::Unknown,
            vendor_id: 0,
            device_id: 0,
            device_type: GpuDeviceType::Unknown,
            driver_version: String::new(),
            api_version: String::new(),
            vram_total_mb: 0,
            vram_available_mb: 0,
            shared_memory_mb: 0,
            max_texture_size: 4096,
            max_compute_workgroup_size: [256, 256, 64],
            supports_raytracing: false,
            supports_mesh_shaders: false,
            supports_variable_rate_shading: false,
            max_msaa_samples: 1,
            supported_apis: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory information
// ---------------------------------------------------------------------------

/// System memory information.
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_physical_mb: u64,
    pub available_physical_mb: u64,
    pub total_virtual_mb: u64,
    pub available_virtual_mb: u64,
    pub page_size: u64,
    pub usage_percent: f32,
}

impl MemoryInfo {
    pub fn detect() -> Self {
        Self {
            total_physical_mb: 0,
            available_physical_mb: 0,
            total_virtual_mb: 0,
            available_virtual_mb: 0,
            page_size: 4096,
            usage_percent: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Display information
// ---------------------------------------------------------------------------

/// Information about a connected display.
#[derive(Debug, Clone)]
pub struct DisplayInfo {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub refresh_rate_hz: f32,
    pub dpi: f32,
    pub scale_factor: f32,
    pub is_primary: bool,
    pub position: (i32, i32),
    pub color_depth: u32,
    pub is_hdr: bool,
    pub max_luminance: f32,
}

impl DisplayInfo {
    pub fn aspect_ratio(&self) -> f32 {
        if self.height > 0 { self.width as f32 / self.height as f32 } else { 1.0 }
    }

    pub fn total_pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

// ---------------------------------------------------------------------------
// Storage information
// ---------------------------------------------------------------------------

/// Information about a storage device.
#[derive(Debug, Clone)]
pub struct StorageInfo {
    pub mount_point: String,
    pub total_gb: f64,
    pub available_gb: f64,
    pub used_gb: f64,
    pub is_ssd: bool,
    pub filesystem: String,
}

// ---------------------------------------------------------------------------
// Build configuration
// ---------------------------------------------------------------------------

/// Build-time configuration information.
#[derive(Debug, Clone)]
pub struct BuildConfig {
    pub profile: BuildProfile,
    pub target_triple: String,
    pub rust_version: String,
    pub engine_version: String,
    pub git_hash: Option<String>,
    pub build_date: String,
    pub optimization_level: u8,
    pub debug_assertions: bool,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildProfile {
    Debug,
    Release,
    RelWithDebInfo,
    MinSizeRel,
}

impl BuildConfig {
    pub fn current() -> Self {
        let profile = if cfg!(debug_assertions) {
            BuildProfile::Debug
        } else {
            BuildProfile::Release
        };

        Self {
            profile,
            target_triple: format!("{}-{}-{}", std::env::consts::ARCH, std::env::consts::OS, std::env::consts::FAMILY),
            rust_version: String::new(),
            engine_version: "0.1.0".to_string(),
            git_hash: None,
            build_date: String::new(),
            optimization_level: if cfg!(debug_assertions) { 0 } else { 3 },
            debug_assertions: cfg!(debug_assertions),
            features: Vec::new(),
        }
    }

    pub fn is_debug(&self) -> bool { self.profile == BuildProfile::Debug }
    pub fn is_release(&self) -> bool { self.profile == BuildProfile::Release }
}

// ---------------------------------------------------------------------------
// Platform info (aggregated)
// ---------------------------------------------------------------------------

/// Complete platform information.
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    pub os: OsInfo,
    pub cpu: CpuInfo,
    pub gpu: GpuInfo,
    pub memory: MemoryInfo,
    pub displays: Vec<DisplayInfo>,
    pub storage: Vec<StorageInfo>,
    pub build: BuildConfig,
}

impl PlatformInfo {
    /// Detect all platform information.
    pub fn detect() -> Self {
        Self {
            os: OsInfo::detect(),
            cpu: CpuInfo::detect(),
            gpu: GpuInfo::unknown(),
            memory: MemoryInfo::detect(),
            displays: Vec::new(),
            storage: Vec::new(),
            build: BuildConfig::current(),
        }
    }

    /// Set the GPU info (typically populated by the render backend).
    pub fn set_gpu_info(&mut self, gpu: GpuInfo) {
        self.gpu = gpu;
    }

    /// Generate a summary string.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("OS: {} ({})\n", self.os.name, self.os.arch));
        s.push_str(&format!("CPU: {} ({} cores)\n", self.cpu.model, self.cpu.logical_cores));
        s.push_str(&format!("GPU: {} ({} MB VRAM)\n", self.gpu.name, self.gpu.vram_total_mb));
        s.push_str(&format!("RAM: {} MB total\n", self.memory.total_physical_mb));
        s.push_str(&format!("Build: {:?} ({})\n", self.build.profile, self.build.target_triple));
        if !self.displays.is_empty() {
            let d = &self.displays[0];
            s.push_str(&format!("Display: {}x{} @{:.0}Hz ({:.0} DPI)\n", d.width, d.height, d.refresh_rate_hz, d.dpi));
        }
        s
    }

    /// Check minimum requirements.
    pub fn check_requirements(&self, requirements: &PlatformRequirements) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.cpu.logical_cores < requirements.min_cpu_cores {
            warnings.push(format!("CPU cores: {} (minimum {})", self.cpu.logical_cores, requirements.min_cpu_cores));
        }
        if self.memory.total_physical_mb < requirements.min_ram_mb {
            warnings.push(format!("RAM: {} MB (minimum {} MB)", self.memory.total_physical_mb, requirements.min_ram_mb));
        }
        if self.gpu.vram_total_mb < requirements.min_vram_mb {
            warnings.push(format!("VRAM: {} MB (minimum {} MB)", self.gpu.vram_total_mb, requirements.min_vram_mb));
        }

        warnings
    }
}

/// Minimum platform requirements.
#[derive(Debug, Clone)]
pub struct PlatformRequirements {
    pub min_cpu_cores: u32,
    pub min_ram_mb: u64,
    pub min_vram_mb: u64,
    pub required_cpu_features: Vec<String>,
    pub required_gpu_features: Vec<String>,
    pub min_os_version: Option<String>,
}

impl Default for PlatformRequirements {
    fn default() -> Self {
        Self {
            min_cpu_cores: 2,
            min_ram_mb: 4096,
            min_vram_mb: 1024,
            required_cpu_features: Vec::new(),
            required_gpu_features: Vec::new(),
            min_os_version: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_os_detection() {
        let os = OsInfo::detect();
        assert!(!os.name.is_empty());
        assert!(os.family != OsFamily::Unknown || cfg!(target_os = "unknown"));
    }

    #[test]
    fn test_cpu_detection() {
        let cpu = CpuInfo::detect();
        assert!(cpu.logical_cores >= 1);
        assert!(cpu.pointer_size == 32 || cpu.pointer_size == 64);
    }

    #[test]
    fn test_build_config() {
        let build = BuildConfig::current();
        assert!(!build.target_triple.is_empty());
        if cfg!(debug_assertions) {
            assert!(build.is_debug());
        } else {
            assert!(build.is_release());
        }
    }

    #[test]
    fn test_gpu_vendor_id() {
        assert_eq!(GpuVendor::from_vendor_id(0x10DE), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_vendor_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_vendor_id(0x8086), GpuVendor::Intel);
    }

    #[test]
    fn test_platform_info_summary() {
        let info = PlatformInfo::detect();
        let summary = info.summary();
        assert!(summary.contains("OS:"));
        assert!(summary.contains("CPU:"));
    }
}
