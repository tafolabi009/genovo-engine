// engine/core/src/version.rs
//
// Engine versioning for the Genovo engine.
//
// Provides semantic version types and build information:
//
// - **Semantic version struct** -- Major.Minor.Patch with pre-release tags.
// - **Version comparison** -- Ordering and compatibility checks.
// - **Build info** -- Commit hash, build date, platform, compiler.
// - **Feature flags** -- Runtime feature availability checks.
// - **Version compatibility** -- Check if two versions are compatible.

use std::fmt;

// ---------------------------------------------------------------------------
// Semantic version
// ---------------------------------------------------------------------------

/// A semantic version (major.minor.patch).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub pre_release: Option<String>,
    pub build_metadata: Option<String>,
}

impl SemanticVersion {
    /// Create a new version.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
            build_metadata: None,
        }
    }

    /// Create with pre-release tag.
    pub fn with_pre_release(mut self, pre: &str) -> Self {
        self.pre_release = Some(pre.to_string());
        self
    }

    /// Create with build metadata.
    pub fn with_build(mut self, build: &str) -> Self {
        self.build_metadata = Some(build.to_string());
        self
    }

    /// Parse from string "major.minor.patch[-pre][+build]".
    pub fn parse(s: &str) -> Option<Self> {
        let (version_part, build) = if let Some(idx) = s.find('+') {
            (&s[..idx], Some(s[idx + 1..].to_string()))
        } else {
            (s, None)
        };

        let (version_part, pre) = if let Some(idx) = version_part.find('-') {
            (&version_part[..idx], Some(version_part[idx + 1..].to_string()))
        } else {
            (version_part, None)
        };

        let parts: Vec<&str> = version_part.split('.').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return None;
        }

        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        let patch = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

        Some(Self {
            major,
            minor,
            patch,
            pre_release: pre,
            build_metadata: build,
        })
    }

    /// Check if this is a pre-release version.
    pub fn is_pre_release(&self) -> bool {
        self.pre_release.is_some()
    }

    /// Check if this version is compatible with another (same major, >= minor).
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        if self.major == 0 && other.major == 0 {
            self.minor == other.minor
        } else {
            self.major == other.major && self.minor >= other.minor
        }
    }

    /// Check if this version is newer than another.
    pub fn is_newer_than(&self, other: &Self) -> bool {
        self > other
    }

    /// Get as a tuple (major, minor, patch).
    pub fn as_tuple(&self) -> (u32, u32, u32) {
        (self.major, self.minor, self.patch)
    }

    /// Bump major version.
    pub fn bump_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }

    /// Bump minor version.
    pub fn bump_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Bump patch version.
    pub fn bump_patch(&self) -> Self {
        Self::new(self.major, self.minor, self.patch + 1)
    }
}

impl fmt::Display for SemanticVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(pre) = &self.pre_release {
            write!(f, "-{}", pre)?;
        }
        if let Some(build) = &self.build_metadata {
            write!(f, "+{}", build)?;
        }
        Ok(())
    }
}

impl PartialOrd for SemanticVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SemanticVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.major.cmp(&other.major) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self.minor.cmp(&other.minor) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self.patch.cmp(&other.patch) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        // Pre-release versions have lower precedence.
        match (&self.pre_release, &other.pre_release) {
            (None, None) => std::cmp::Ordering::Equal,
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (Some(a), Some(b)) => a.cmp(b),
        }
    }
}

// ---------------------------------------------------------------------------
// Build info
// ---------------------------------------------------------------------------

/// Build information.
#[derive(Debug, Clone)]
pub struct BuildInfo {
    pub version: SemanticVersion,
    pub commit_hash: String,
    pub commit_short: String,
    pub build_date: String,
    pub build_timestamp: u64,
    pub platform: String,
    pub arch: String,
    pub compiler: String,
    pub compiler_version: String,
    pub debug_build: bool,
    pub profile: String,
    pub branch: String,
}

impl BuildInfo {
    /// Create build info for the current environment.
    pub fn current() -> Self {
        Self {
            version: ENGINE_VERSION,
            commit_hash: String::new(),
            commit_short: String::new(),
            build_date: String::new(),
            build_timestamp: 0,
            platform: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            compiler: "rustc".to_string(),
            compiler_version: String::new(),
            debug_build: cfg!(debug_assertions),
            profile: if cfg!(debug_assertions) { "debug" } else { "release" }.to_string(),
            branch: String::new(),
        }
    }

    /// Get a short display string.
    pub fn short_string(&self) -> String {
        format!(
            "Genovo {} ({} {} {})",
            self.version, self.platform, self.arch, self.profile
        )
    }

    /// Get a long display string.
    pub fn long_string(&self) -> String {
        format!(
            "Genovo Engine v{}\nCommit: {}\nDate: {}\nPlatform: {} {}\nCompiler: {} {}\nProfile: {}",
            self.version,
            if self.commit_short.is_empty() { "unknown" } else { &self.commit_short },
            if self.build_date.is_empty() { "unknown" } else { &self.build_date },
            self.platform,
            self.arch,
            self.compiler,
            self.compiler_version,
            self.profile,
        )
    }
}

impl fmt::Display for BuildInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_string())
    }
}

// ---------------------------------------------------------------------------
// Feature flags
// ---------------------------------------------------------------------------

/// Runtime feature flags.
#[derive(Debug, Clone)]
pub struct FeatureFlags {
    flags: HashMap<String, bool>,
}

impl FeatureFlags {
    pub fn new() -> Self {
        Self {
            flags: HashMap::new(),
        }
    }

    pub fn set(&mut self, feature: &str, enabled: bool) {
        self.flags.insert(feature.to_string(), enabled);
    }

    pub fn is_enabled(&self, feature: &str) -> bool {
        self.flags.get(feature).copied().unwrap_or(false)
    }

    pub fn enabled_features(&self) -> Vec<&str> {
        self.flags
            .iter()
            .filter(|(_, &v)| v)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    pub fn all_features(&self) -> &HashMap<String, bool> {
        &self.flags
    }

    /// Create default engine feature flags.
    pub fn engine_defaults() -> Self {
        let mut flags = Self::new();
        flags.set("vulkan", cfg!(feature = "vulkan"));
        flags.set("dx12", cfg!(feature = "dx12"));
        flags.set("metal", cfg!(feature = "metal"));
        flags.set("wgpu", true);
        flags.set("physics", true);
        flags.set("audio", true);
        flags.set("networking", true);
        flags.set("editor", cfg!(debug_assertions));
        flags.set("profiling", true);
        flags.set("hot_reload", cfg!(debug_assertions));
        flags
    }
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self::new()
    }
}

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Version compatibility
// ---------------------------------------------------------------------------

/// Version compatibility checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityLevel {
    /// Fully compatible.
    Full,
    /// Compatible but with deprecation warnings.
    Deprecated,
    /// May have breaking changes.
    Partial,
    /// Not compatible.
    Incompatible,
}

impl fmt::Display for CompatibilityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "Fully Compatible"),
            Self::Deprecated => write!(f, "Compatible (deprecated)"),
            Self::Partial => write!(f, "Partially Compatible"),
            Self::Incompatible => write!(f, "Incompatible"),
        }
    }
}

/// Check compatibility between two versions.
pub fn check_compatibility(
    engine_version: &SemanticVersion,
    required_version: &SemanticVersion,
) -> CompatibilityLevel {
    if engine_version.major != required_version.major {
        return CompatibilityLevel::Incompatible;
    }
    if engine_version.minor < required_version.minor {
        return CompatibilityLevel::Incompatible;
    }
    if engine_version.minor > required_version.minor + 2 {
        return CompatibilityLevel::Deprecated;
    }
    if engine_version.minor > required_version.minor {
        return CompatibilityLevel::Full;
    }
    if engine_version.patch >= required_version.patch {
        CompatibilityLevel::Full
    } else {
        CompatibilityLevel::Partial
    }
}

// ---------------------------------------------------------------------------
// Current engine version
// ---------------------------------------------------------------------------

/// The current Genovo engine version.
pub const ENGINE_VERSION: SemanticVersion = SemanticVersion::new(0, 1, 0);

/// Get the current build info.
pub fn build_info() -> BuildInfo {
    BuildInfo::current()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parse() {
        let v = SemanticVersion::parse("1.2.3").unwrap();
        assert_eq!(v, SemanticVersion::new(1, 2, 3));

        let v = SemanticVersion::parse("1.2.3-beta.1+build123").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.pre_release, Some("beta.1".to_string()));
        assert_eq!(v.build_metadata, Some("build123".to_string()));
    }

    #[test]
    fn test_version_ordering() {
        assert!(SemanticVersion::new(1, 0, 0) > SemanticVersion::new(0, 99, 99));
        assert!(SemanticVersion::new(1, 1, 0) > SemanticVersion::new(1, 0, 99));
        assert!(SemanticVersion::new(1, 0, 1) > SemanticVersion::new(1, 0, 0));
    }

    #[test]
    fn test_pre_release_ordering() {
        let release = SemanticVersion::new(1, 0, 0);
        let pre = SemanticVersion::new(1, 0, 0).with_pre_release("alpha");
        assert!(release > pre);
    }

    #[test]
    fn test_compatibility() {
        let engine = SemanticVersion::new(1, 5, 0);
        let required = SemanticVersion::new(1, 3, 0);
        assert_eq!(check_compatibility(&engine, &required), CompatibilityLevel::Full);

        let old = SemanticVersion::new(0, 1, 0);
        assert_eq!(check_compatibility(&engine, &old), CompatibilityLevel::Incompatible);
    }

    #[test]
    fn test_feature_flags() {
        let mut flags = FeatureFlags::new();
        flags.set("vulkan", true);
        flags.set("dx12", false);
        assert!(flags.is_enabled("vulkan"));
        assert!(!flags.is_enabled("dx12"));
        assert!(!flags.is_enabled("nonexistent"));
    }

    #[test]
    fn test_version_display() {
        let v = SemanticVersion::new(1, 2, 3).with_pre_release("beta");
        assert_eq!(v.to_string(), "1.2.3-beta");
    }

    #[test]
    fn test_bump() {
        let v = SemanticVersion::new(1, 2, 3);
        assert_eq!(v.bump_major(), SemanticVersion::new(2, 0, 0));
        assert_eq!(v.bump_minor(), SemanticVersion::new(1, 3, 0));
        assert_eq!(v.bump_patch(), SemanticVersion::new(1, 2, 4));
    }
}
