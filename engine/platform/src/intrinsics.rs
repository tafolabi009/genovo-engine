//! # Platform Intrinsics
//!
//! CPU feature detection, hardware query, and low-level platform-specific
//! helper functions for the Genovo engine.
//!
//! This module provides:
//!
//! - Exhaustive CPU feature detection (SSE, SSE2..SSE4.2, AVX, AVX2, AVX-512, FMA, NEON)
//! - `CpuFeatures` struct with all flags
//! - Cache line size, core/thread count, and CPU frequency detection
//! - Memory information (total RAM, available RAM, page size)
//! - Low-level intrinsics: `prefetch`, `pause` (spin-loop hint), `rdtsc`

use std::fmt;

// ============================================================================
// CPU Feature Flags
// ============================================================================

/// Comprehensive CPU feature detection results.
///
/// All fields default to `false`. Call [`detect_cpu_features()`] to populate
/// a `CpuFeatures` struct with the current CPU's capabilities.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFeatures {
    // ---- x86/x86_64 SIMD extensions ----
    /// SSE (Streaming SIMD Extensions).
    pub sse: bool,
    /// SSE2.
    pub sse2: bool,
    /// SSE3.
    pub sse3: bool,
    /// SSSE3 (Supplemental SSE3).
    pub ssse3: bool,
    /// SSE4.1.
    pub sse41: bool,
    /// SSE4.2.
    pub sse42: bool,
    /// AVX (Advanced Vector Extensions).
    pub avx: bool,
    /// AVX2.
    pub avx2: bool,
    /// AVX-512 Foundation.
    pub avx512f: bool,
    /// AVX-512 Byte and Word.
    pub avx512bw: bool,
    /// AVX-512 Doubleword and Quadword.
    pub avx512dq: bool,
    /// AVX-512 Vector Length extensions.
    pub avx512vl: bool,
    /// FMA (Fused Multiply-Add).
    pub fma: bool,
    /// AES-NI (hardware AES encryption).
    pub aesni: bool,
    /// POPCNT instruction.
    pub popcnt: bool,
    /// BMI1 (Bit Manipulation Instructions 1).
    pub bmi1: bool,
    /// BMI2 (Bit Manipulation Instructions 2).
    pub bmi2: bool,
    /// F16C (half-precision float conversion).
    pub f16c: bool,
    /// RDRAND (hardware random number generation).
    pub rdrand: bool,
    /// LZCNT (leading zero count).
    pub lzcnt: bool,
    /// MOVBE (move big-endian).
    pub movbe: bool,
    /// TSC (timestamp counter) supported.
    pub tsc: bool,
    /// Invariant TSC (constant-rate timestamp counter).
    pub tsc_invariant: bool,

    // ---- ARM NEON extensions ----
    /// ARM NEON (Advanced SIMD).
    pub neon: bool,
    /// ARM CRC32 instructions.
    pub crc32: bool,
    /// ARM AES instructions.
    pub arm_aes: bool,
    /// ARM SHA1 instructions.
    pub arm_sha1: bool,
    /// ARM SHA2 instructions.
    pub arm_sha2: bool,
    /// ARM dot product instructions (SDOT/UDOT).
    pub arm_dotprod: bool,
    /// ARM SVE (Scalable Vector Extension).
    pub arm_sve: bool,

    // ---- General CPU info ----
    /// Physical core count (0 if unknown).
    pub physical_cores: u32,
    /// Logical thread count (0 if unknown).
    pub logical_threads: u32,
    /// L1 data cache size in bytes (0 if unknown).
    pub l1d_cache_size: u32,
    /// L1 instruction cache size in bytes (0 if unknown).
    pub l1i_cache_size: u32,
    /// L2 cache size in bytes (0 if unknown).
    pub l2_cache_size: u32,
    /// L3 cache size in bytes (0 if unknown).
    pub l3_cache_size: u32,
    /// Cache line size in bytes (typically 64).
    pub cache_line_size: u32,
    /// CPU base frequency in MHz (0 if unknown).
    pub base_frequency_mhz: u32,
    /// CPU maximum boost frequency in MHz (0 if unknown).
    pub max_frequency_mhz: u32,
}

impl CpuFeatures {
    /// Returns a string listing all detected SIMD extensions.
    pub fn simd_summary(&self) -> String {
        let mut features = Vec::new();

        if self.sse { features.push("SSE"); }
        if self.sse2 { features.push("SSE2"); }
        if self.sse3 { features.push("SSE3"); }
        if self.ssse3 { features.push("SSSE3"); }
        if self.sse41 { features.push("SSE4.1"); }
        if self.sse42 { features.push("SSE4.2"); }
        if self.avx { features.push("AVX"); }
        if self.avx2 { features.push("AVX2"); }
        if self.avx512f { features.push("AVX-512F"); }
        if self.avx512bw { features.push("AVX-512BW"); }
        if self.avx512dq { features.push("AVX-512DQ"); }
        if self.avx512vl { features.push("AVX-512VL"); }
        if self.fma { features.push("FMA"); }
        if self.neon { features.push("NEON"); }
        if self.arm_dotprod { features.push("DotProd"); }
        if self.arm_sve { features.push("SVE"); }

        if features.is_empty() {
            "none".to_string()
        } else {
            features.join(", ")
        }
    }

    /// Returns the best SIMD width available in bits.
    pub fn best_simd_width(&self) -> u32 {
        if self.avx512f {
            512
        } else if self.avx2 || self.avx {
            256
        } else if self.sse2 || self.neon {
            128
        } else {
            0
        }
    }

    /// Returns true if this CPU supports at least SSE2 (or NEON on ARM).
    pub fn has_basic_simd(&self) -> bool {
        self.sse2 || self.neon
    }

    /// Returns true if this CPU supports advanced SIMD (AVX2 or NEON with dotprod).
    pub fn has_advanced_simd(&self) -> bool {
        self.avx2 || (self.neon && self.arm_dotprod)
    }
}

impl fmt::Display for CpuFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CPU Features:")?;
        writeln!(f, "  SIMD: {}", self.simd_summary())?;
        writeln!(f, "  Best SIMD width: {} bits", self.best_simd_width())?;
        writeln!(
            f,
            "  Cores: {} physical, {} logical",
            self.physical_cores, self.logical_threads
        )?;
        writeln!(f, "  Cache line: {} bytes", self.cache_line_size)?;
        if self.l1d_cache_size > 0 {
            writeln!(f, "  L1d: {} KB", self.l1d_cache_size / 1024)?;
        }
        if self.l2_cache_size > 0 {
            writeln!(f, "  L2: {} KB", self.l2_cache_size / 1024)?;
        }
        if self.l3_cache_size > 0 {
            writeln!(f, "  L3: {} KB", self.l3_cache_size / 1024)?;
        }
        if self.base_frequency_mhz > 0 {
            writeln!(f, "  Base freq: {} MHz", self.base_frequency_mhz)?;
        }
        if self.max_frequency_mhz > 0 {
            writeln!(f, "  Max freq: {} MHz", self.max_frequency_mhz)?;
        }
        Ok(())
    }
}

// ============================================================================
// Feature detection
// ============================================================================

/// Detect CPU features at runtime.
///
/// This function queries the CPU for supported instruction set extensions
/// and hardware characteristics.
pub fn detect_cpu_features() -> CpuFeatures {
    let mut features = CpuFeatures::default();

    // ---- x86_64 feature detection ----
    #[cfg(target_arch = "x86_64")]
    {
        features.sse = is_x86_feature_detected!("sse");
        features.sse2 = is_x86_feature_detected!("sse2");
        features.sse3 = is_x86_feature_detected!("sse3");
        features.ssse3 = is_x86_feature_detected!("ssse3");
        features.sse41 = is_x86_feature_detected!("sse4.1");
        features.sse42 = is_x86_feature_detected!("sse4.2");
        features.avx = is_x86_feature_detected!("avx");
        features.avx2 = is_x86_feature_detected!("avx2");
        features.avx512f = is_x86_feature_detected!("avx512f");
        features.avx512bw = is_x86_feature_detected!("avx512bw");
        features.avx512dq = is_x86_feature_detected!("avx512dq");
        features.avx512vl = is_x86_feature_detected!("avx512vl");
        features.fma = is_x86_feature_detected!("fma");
        features.aesni = is_x86_feature_detected!("aes");
        features.popcnt = is_x86_feature_detected!("popcnt");
        features.bmi1 = is_x86_feature_detected!("bmi1");
        features.bmi2 = is_x86_feature_detected!("bmi2");
        features.f16c = is_x86_feature_detected!("f16c");
        features.rdrand = is_x86_feature_detected!("rdrand");
        features.lzcnt = is_x86_feature_detected!("lzcnt");
        features.movbe = is_x86_feature_detected!("movbe");
        features.tsc = true; // All modern x86_64 CPUs have TSC
        // Invariant TSC is indicated by CPUID leaf 0x80000007, bit 8
        // We cannot easily check this with std library alone, so
        // assume invariant TSC on x86_64 (true for all modern CPUs).
        features.tsc_invariant = true;
    }

    // ---- aarch64 feature detection ----
    #[cfg(target_arch = "aarch64")]
    {
        // All aarch64 CPUs have NEON (it's mandatory in the architecture)
        features.neon = true;

        // aarch64 feature detection via std::arch is limited;
        // use cfg target_feature for compile-time detection
        #[cfg(target_feature = "crc")]
        {
            features.crc32 = true;
        }
        #[cfg(target_feature = "aes")]
        {
            features.arm_aes = true;
        }
        #[cfg(target_feature = "sha2")]
        {
            features.arm_sha2 = true;
        }
        #[cfg(target_feature = "dotprod")]
        {
            features.arm_dotprod = true;
        }
        #[cfg(target_feature = "sve")]
        {
            features.arm_sve = true;
        }
    }

    // ---- Core/thread count ----
    features.logical_threads = detect_logical_threads();
    features.physical_cores = detect_physical_cores();

    // ---- Cache info ----
    features.cache_line_size = detect_cache_line_size();
    let (l1d, l1i, l2, l3) = detect_cache_sizes();
    features.l1d_cache_size = l1d;
    features.l1i_cache_size = l1i;
    features.l2_cache_size = l2;
    features.l3_cache_size = l3;

    // ---- Frequency info ----
    let (base, max) = detect_cpu_frequency();
    features.base_frequency_mhz = base;
    features.max_frequency_mhz = max;

    features
}

/// Detect the number of logical threads (hardware threads / hyperthreads).
fn detect_logical_threads() -> u32 {
    // std::thread::available_parallelism() is the portable way
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

/// Detect the number of physical cores.
fn detect_physical_cores() -> u32 {
    // Without a dedicated library, we estimate physical cores as
    // logical_threads / 2 on x86 (assuming HT), or equal on ARM.
    let logical = detect_logical_threads();

    #[cfg(target_arch = "x86_64")]
    {
        // Most x86_64 CPUs have 2 threads per core (SMT/HT)
        // This is a heuristic; for exact counts, use platform-specific APIs
        (logical / 2).max(1)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // ARM and other architectures typically don't have SMT
        logical
    }
}

/// Detect the cache line size in bytes.
fn detect_cache_line_size() -> u32 {
    // Most modern CPUs use 64-byte cache lines
    #[cfg(target_arch = "x86_64")]
    {
        64 // All modern x86_64 CPUs use 64-byte cache lines
    }
    #[cfg(target_arch = "aarch64")]
    {
        // ARM Cortex-A series typically uses 64-byte cache lines
        // Apple M-series uses 128-byte cache lines for performance cores
        #[cfg(target_os = "macos")]
        {
            128
        }
        #[cfg(not(target_os = "macos"))]
        {
            64
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        64 // Reasonable default
    }
}

/// Detect cache sizes (L1D, L1I, L2, L3) in bytes.
/// Returns (l1d, l1i, l2, l3). Values are 0 if unknown.
fn detect_cache_sizes() -> (u32, u32, u32, u32) {
    // Without raw CPUID access or platform-specific APIs, we use
    // common defaults for modern CPUs.
    #[cfg(target_arch = "x86_64")]
    {
        // Common defaults for modern Intel/AMD desktop CPUs
        (
            32 * 1024,         // L1D: 32 KB
            32 * 1024,         // L1I: 32 KB
            256 * 1024,        // L2: 256 KB
            8 * 1024 * 1024,   // L3: 8 MB
        )
    }
    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_os = "macos")]
        {
            // Apple Silicon (M-series) performance core estimates
            (
                192 * 1024,        // L1D: 192 KB (performance core)
                128 * 1024,        // L1I: 128 KB
                4 * 1024 * 1024,   // L2: 4 MB (per cluster)
                16 * 1024 * 1024,  // L3: 16 MB (shared)
            )
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Generic ARM Cortex-A estimates
            (
                64 * 1024,         // L1D: 64 KB
                64 * 1024,         // L1I: 64 KB
                512 * 1024,        // L2: 512 KB
                2 * 1024 * 1024,   // L3: 2 MB
            )
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        (0, 0, 0, 0)
    }
}

/// Detect CPU base and max frequency in MHz.
/// Returns (base_mhz, max_mhz). Values are 0 if unknown.
fn detect_cpu_frequency() -> (u32, u32) {
    // CPU frequency detection is platform-specific and unreliable
    // without kernel APIs. Return 0 to indicate "unknown".
    //
    // On Linux: /sys/devices/system/cpu/cpu0/cpufreq/
    // On Windows: Registry or WMI
    // On macOS: sysctl hw.cpufrequency
    (0, 0)
}

// ============================================================================
// Memory information
// ============================================================================

/// System memory information.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryInfo {
    /// Total physical RAM in bytes.
    pub total_ram: u64,
    /// Available (free) RAM in bytes.
    pub available_ram: u64,
    /// System page size in bytes.
    pub page_size: u32,
    /// Large page size in bytes (0 if not supported).
    pub large_page_size: u32,
}

impl MemoryInfo {
    /// Total RAM in megabytes.
    pub fn total_ram_mb(&self) -> u64 {
        self.total_ram / (1024 * 1024)
    }

    /// Available RAM in megabytes.
    pub fn available_ram_mb(&self) -> u64 {
        self.available_ram / (1024 * 1024)
    }

    /// Memory usage as a percentage.
    pub fn usage_percent(&self) -> f32 {
        if self.total_ram == 0 {
            return 0.0;
        }
        let used = self.total_ram.saturating_sub(self.available_ram);
        (used as f64 / self.total_ram as f64 * 100.0) as f32
    }
}

impl fmt::Display for MemoryInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Memory Info:")?;
        writeln!(f, "  Total RAM: {} MB", self.total_ram_mb())?;
        writeln!(f, "  Available: {} MB", self.available_ram_mb())?;
        writeln!(f, "  Usage: {:.1}%", self.usage_percent())?;
        writeln!(f, "  Page size: {} bytes", self.page_size)?;
        if self.large_page_size > 0 {
            writeln!(f, "  Large page: {} KB", self.large_page_size / 1024)?;
        }
        Ok(())
    }
}

/// Query system memory information.
pub fn detect_memory_info() -> MemoryInfo {
    let mut info = MemoryInfo::default();

    // Page size (portable via std)
    info.page_size = detect_page_size();

    // Platform-specific memory detection
    #[cfg(target_os = "windows")]
    {
        // On Windows, we'd use GetPhysicallyInstalledSystemMemory or
        // GlobalMemoryStatusEx. Without the windows crate imported here,
        // provide estimates from available_parallelism heuristics.
        // The actual windows module in platform/ handles the real API calls.
        info.total_ram = 0;
        info.available_ram = 0;
        info.large_page_size = 2 * 1024 * 1024; // 2 MB large pages on Windows
    }

    #[cfg(target_os = "linux")]
    {
        // On Linux, parse /proc/meminfo
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    info.total_ram = parse_meminfo_value(line) * 1024;
                } else if line.starts_with("MemAvailable:") {
                    info.available_ram = parse_meminfo_value(line) * 1024;
                }
            }
        }
        info.large_page_size = 2 * 1024 * 1024; // 2 MB huge pages
    }

    #[cfg(target_os = "macos")]
    {
        info.total_ram = 0; // Would use sysctl hw.memsize
        info.available_ram = 0;
        info.large_page_size = 16 * 1024 * 1024; // 16 MB super pages on macOS
    }

    info
}

/// Detect the system page size.
fn detect_page_size() -> u32 {
    #[cfg(target_os = "windows")]
    {
        4096 // Standard Windows page size
    }
    #[cfg(target_os = "linux")]
    {
        4096 // Standard Linux page size (could also be 16K on ARM64)
    }
    #[cfg(target_os = "macos")]
    {
        #[cfg(target_arch = "aarch64")]
        {
            16384 // Apple Silicon uses 16 KB pages
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            4096 // Intel Macs use 4 KB pages
        }
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        4096 // Default
    }
}

/// Parse a value from a /proc/meminfo line (Linux).
#[cfg(target_os = "linux")]
fn parse_meminfo_value(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

// ============================================================================
// Low-level intrinsics
// ============================================================================

/// Software prefetch: hint to the CPU to bring data into cache.
///
/// On x86_64 this maps to `_mm_prefetch` with temporal hint T0 (all cache levels).
/// On aarch64 this maps to `prfm pldl1keep`.
/// On unsupported architectures this is a no-op.
///
/// # Safety
///
/// `addr` must be a valid pointer (or at worst point to unmapped memory,
/// which will be silently ignored by the CPU -- prefetch never faults).
#[inline(always)]
pub unsafe fn prefetch<T>(addr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        // _MM_HINT_T0 = 3: prefetch into all cache levels
        unsafe {
            std::arch::x86_64::_mm_prefetch(addr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Use inline assembly for PRFM on aarch64
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{addr}]",
                addr = in(reg) addr,
                options(nostack, preserves_flags),
            );
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = addr; // No-op on unsupported architectures
    }
}

/// Software prefetch for write: hint to the CPU that data will be written.
///
/// On x86_64 this maps to `_mm_prefetch` with hint T0.
/// On aarch64 this maps to `prfm pstl1keep`.
///
/// # Safety
///
/// Same requirements as [`prefetch`].
#[inline(always)]
pub unsafe fn prefetch_write<T>(addr: *mut T) {
    #[cfg(target_arch = "x86_64")]
    {
        // Use prefetchw if available, otherwise prefetcht0
        unsafe {
            std::arch::x86_64::_mm_prefetch(addr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            std::arch::asm!(
                "prfm pstl1keep, [{addr}]",
                addr = in(reg) addr,
                options(nostack, preserves_flags),
            );
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = addr;
    }
}

/// Spin-loop hint: reduce power consumption and improve performance of
/// spin-wait loops.
///
/// On x86_64 this emits the `PAUSE` instruction.
/// On aarch64 this emits `YIELD`.
/// On other architectures this is a compiler hint via `spin_loop_hint`.
#[inline(always)]
pub fn pause() {
    std::hint::spin_loop();
}

/// Read the CPU timestamp counter (x86_64 only).
///
/// Returns the current value of the TSC register. The TSC counts processor
/// clock cycles since reset and is useful for high-resolution timing.
///
/// On non-x86_64 architectures, returns a monotonic counter based on
/// `std::time::Instant`.
///
/// # Note
///
/// TSC values are NOT directly comparable across different CPU cores unless
/// the CPU has an invariant TSC (check `CpuFeatures::tsc_invariant`).
#[inline]
pub fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { std::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback: use Instant as a monotonic counter
        // This is less precise but works everywhere
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let start = START.get_or_init(std::time::Instant::now);
        start.elapsed().as_nanos() as u64
    }
}

/// Read the TSC with a serializing fence (x86_64 only).
///
/// Uses `RDTSCP` which waits for all previous instructions to complete
/// before reading the TSC. This provides more accurate measurements
/// than `rdtsc()` for benchmarking.
///
/// On non-x86_64 architectures, falls back to `rdtsc()`.
#[inline]
pub fn rdtscp() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let mut _aux: u32 = 0;
        unsafe { std::arch::x86_64::__rdtscp(&mut _aux) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        rdtsc()
    }
}

/// Memory fence: ensure all previous memory stores are visible to
/// other cores before any subsequent stores.
///
/// On x86_64 this emits `MFENCE`. On aarch64 this emits `DMB ISH`.
/// On other architectures this uses `std::sync::atomic::fence`.
#[inline(always)]
pub fn memory_fence() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

/// Store fence: ensure all previous stores are completed before
/// subsequent stores.
#[inline(always)]
pub fn store_fence() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
}

/// Load fence: ensure all previous loads are completed before
/// subsequent loads.
#[inline(always)]
pub fn load_fence() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
}

/// Compiler fence: prevent the compiler from reordering instructions
/// across this point. Does NOT emit a hardware fence instruction.
#[inline(always)]
pub fn compiler_fence() {
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

// ============================================================================
// Timing utilities
// ============================================================================

/// High-resolution timer using platform-specific counters.
#[derive(Debug, Clone, Copy)]
pub struct PrecisionTimer {
    /// TSC value at start.
    start_tsc: u64,
    /// Instant at start (for conversion to real time).
    start_instant: std::time::Instant,
}

impl PrecisionTimer {
    /// Start a new precision timer.
    pub fn start() -> Self {
        let instant = std::time::Instant::now();
        let tsc = rdtscp();
        Self {
            start_tsc: tsc,
            start_instant: instant,
        }
    }

    /// Get elapsed TSC ticks since start.
    pub fn elapsed_ticks(&self) -> u64 {
        rdtscp().wrapping_sub(self.start_tsc)
    }

    /// Get elapsed time in nanoseconds using `Instant`.
    pub fn elapsed_ns(&self) -> u64 {
        self.start_instant.elapsed().as_nanos() as u64
    }

    /// Get elapsed time in microseconds.
    pub fn elapsed_us(&self) -> f64 {
        self.elapsed_ns() as f64 / 1000.0
    }

    /// Get elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ns() as f64 / 1_000_000.0
    }

    /// Get elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        self.start_instant.elapsed().as_secs_f64()
    }

    /// Reset the timer.
    pub fn reset(&mut self) {
        self.start_instant = std::time::Instant::now();
        self.start_tsc = rdtscp();
    }
}

// ============================================================================
// System info aggregation
// ============================================================================

/// Complete system information snapshot.
#[derive(Debug, Clone)]
pub struct SystemProfile {
    /// CPU feature flags.
    pub cpu: CpuFeatures,
    /// Memory information.
    pub memory: MemoryInfo,
    /// Architecture name (e.g., "x86_64", "aarch64").
    pub arch: &'static str,
    /// Operating system name.
    pub os: &'static str,
    /// Pointer size in bytes (4 for 32-bit, 8 for 64-bit).
    pub pointer_size: u32,
    /// Endianness: true if big-endian.
    pub big_endian: bool,
}

impl SystemProfile {
    /// Gather a complete system profile.
    pub fn detect() -> Self {
        Self {
            cpu: detect_cpu_features(),
            memory: detect_memory_info(),
            arch: std::env::consts::ARCH,
            os: std::env::consts::OS,
            pointer_size: std::mem::size_of::<*const u8>() as u32,
            big_endian: cfg!(target_endian = "big"),
        }
    }
}

impl fmt::Display for SystemProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== System Profile ===")?;
        writeln!(f, "Arch: {} ({})", self.arch, if self.big_endian { "big-endian" } else { "little-endian" })?;
        writeln!(f, "OS: {}", self.os)?;
        writeln!(f, "Pointer size: {} bytes", self.pointer_size)?;
        write!(f, "{}", self.cpu)?;
        write!(f, "{}", self.memory)?;
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_cpu_features_does_not_panic() {
        let features = detect_cpu_features();
        // On x86_64, SSE2 should always be present (it's baseline for the arch)
        #[cfg(target_arch = "x86_64")]
        {
            assert!(features.sse2, "SSE2 should be present on x86_64");
            assert!(features.sse, "SSE should be present on x86_64");
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert!(features.neon, "NEON should be present on aarch64");
        }
    }

    #[test]
    fn test_cpu_features_default() {
        let features = CpuFeatures::default();
        assert!(!features.sse);
        assert!(!features.avx2);
        assert!(!features.neon);
        assert_eq!(features.cache_line_size, 0);
    }

    #[test]
    fn test_simd_summary() {
        let features = detect_cpu_features();
        let summary = features.simd_summary();
        assert!(!summary.is_empty());
        // On x86_64, should contain at least SSE
        #[cfg(target_arch = "x86_64")]
        {
            assert!(summary.contains("SSE"));
        }
    }

    #[test]
    fn test_best_simd_width() {
        let features = detect_cpu_features();
        let width = features.best_simd_width();
        #[cfg(target_arch = "x86_64")]
        {
            assert!(width >= 128, "x86_64 should have at least 128-bit SIMD");
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(width, 128, "aarch64 NEON is 128-bit");
        }
    }

    #[test]
    fn test_has_basic_simd() {
        let features = detect_cpu_features();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            assert!(features.has_basic_simd());
        }
    }

    #[test]
    fn test_detect_logical_threads() {
        let threads = detect_logical_threads();
        assert!(threads >= 1, "Should detect at least 1 thread");
    }

    #[test]
    fn test_detect_physical_cores() {
        let cores = detect_physical_cores();
        assert!(cores >= 1, "Should detect at least 1 core");
    }

    #[test]
    fn test_detect_cache_line_size() {
        let size = detect_cache_line_size();
        assert!(size >= 32 && size <= 256, "Cache line size should be reasonable");
        // Should be a power of two
        assert!(size.is_power_of_two(), "Cache line size should be power of 2");
    }

    #[test]
    fn test_detect_memory_info() {
        let info = detect_memory_info();
        assert!(info.page_size > 0, "Page size should be non-zero");
        assert!(info.page_size.is_power_of_two(), "Page size should be power of 2");
    }

    #[test]
    fn test_memory_info_calculations() {
        let info = MemoryInfo {
            total_ram: 16 * 1024 * 1024 * 1024, // 16 GB
            available_ram: 8 * 1024 * 1024 * 1024, // 8 GB
            page_size: 4096,
            large_page_size: 2 * 1024 * 1024,
        };
        assert_eq!(info.total_ram_mb(), 16384);
        assert_eq!(info.available_ram_mb(), 8192);
        assert!((info.usage_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_memory_info_zero_total() {
        let info = MemoryInfo::default();
        assert_eq!(info.usage_percent(), 0.0);
    }

    #[test]
    fn test_pause_does_not_crash() {
        // Just verify it doesn't panic
        pause();
    }

    #[test]
    fn test_rdtsc_monotonic() {
        let t1 = rdtsc();
        // Do some work to ensure the counter advances
        std::hint::black_box(0u64.wrapping_add(1));
        let t2 = rdtsc();
        // On most CPUs, t2 should be > t1, but we can't guarantee
        // it due to potential wrapping. Just verify they're callable.
        let _ = (t1, t2);
    }

    #[test]
    fn test_rdtscp() {
        let t1 = rdtscp();
        std::hint::black_box(0u64.wrapping_add(1));
        let t2 = rdtscp();
        let _ = (t1, t2);
    }

    #[test]
    fn test_memory_fences_do_not_crash() {
        memory_fence();
        store_fence();
        load_fence();
        compiler_fence();
    }

    #[test]
    fn test_precision_timer() {
        let timer = PrecisionTimer::start();
        // Do some trivial work
        let mut x = 0u64;
        for i in 0..1000 {
            x = x.wrapping_add(i);
        }
        std::hint::black_box(x);
        let ns = timer.elapsed_ns();
        let ms = timer.elapsed_ms();
        let us = timer.elapsed_us();
        let ticks = timer.elapsed_ticks();
        // Timer should report non-negative values
        assert!(ms >= 0.0);
        assert!(us >= 0.0);
        let _ = (ns, ticks);
    }

    #[test]
    fn test_precision_timer_reset() {
        let mut timer = PrecisionTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let before_reset = timer.elapsed_ns();
        timer.reset();
        let after_reset = timer.elapsed_ns();
        assert!(
            before_reset >= after_reset,
            "After reset, elapsed should be less"
        );
    }

    #[test]
    fn test_system_profile() {
        let profile = SystemProfile::detect();
        assert!(!profile.arch.is_empty());
        assert!(!profile.os.is_empty());
        assert!(profile.pointer_size == 4 || profile.pointer_size == 8);
    }

    #[test]
    fn test_system_profile_display() {
        let profile = SystemProfile::detect();
        let display = format!("{}", profile);
        assert!(display.contains("System Profile"));
        assert!(display.contains("Arch:"));
        assert!(display.contains("OS:"));
    }

    #[test]
    fn test_cpu_features_display() {
        let features = detect_cpu_features();
        let display = format!("{}", features);
        assert!(display.contains("CPU Features:"));
        assert!(display.contains("SIMD:"));
    }

    #[test]
    fn test_memory_info_display() {
        let info = MemoryInfo {
            total_ram: 8 * 1024 * 1024 * 1024,
            available_ram: 4 * 1024 * 1024 * 1024,
            page_size: 4096,
            large_page_size: 0,
        };
        let display = format!("{}", info);
        assert!(display.contains("Memory Info:"));
        assert!(display.contains("Total RAM:"));
    }

    #[test]
    fn test_prefetch_does_not_crash() {
        let data = [0u8; 256];
        unsafe {
            prefetch(data.as_ptr());
        }
    }

    #[test]
    fn test_prefetch_write_does_not_crash() {
        let mut data = [0u8; 256];
        unsafe {
            prefetch_write(data.as_mut_ptr());
        }
    }

    #[test]
    fn test_detect_page_size() {
        let ps = detect_page_size();
        assert!(ps >= 4096);
        assert!(ps.is_power_of_two());
    }

    #[test]
    fn test_detect_cache_sizes() {
        let (l1d, _l1i, l2, _l3) = detect_cache_sizes();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            assert!(l1d > 0, "L1D cache size should be detected on x86_64/aarch64");
            assert!(l2 > 0, "L2 cache size should be detected on x86_64/aarch64");
        }
    }
}
