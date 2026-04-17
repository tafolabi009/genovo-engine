//! Physics material database with named materials and combination rules.
//!
//! Provides:
//! - Named physics materials with friction, restitution, and density
//! - Built-in material presets: ice, rubber, wood, metal, glass, concrete, sand
//! - Material combination rules for contact resolution (average, min, max, multiply)
//! - Contact sound hints for audio integration
//! - Material database with lookup by name or ID
//! - Custom material creation and registration
//! - Serialization-friendly material definitions

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default static friction coefficient.
const DEFAULT_STATIC_FRICTION: f32 = 0.5;
/// Default dynamic friction coefficient.
const DEFAULT_DYNAMIC_FRICTION: f32 = 0.4;
/// Default restitution (bounciness).
const DEFAULT_RESTITUTION: f32 = 0.3;
/// Default density in kg/m^3.
const DEFAULT_DENSITY: f32 = 1000.0;
/// Small epsilon.
const EPSILON: f32 = 1e-7;

// ---------------------------------------------------------------------------
// Material ID
// ---------------------------------------------------------------------------

/// Unique identifier for a physics material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysMaterialId(pub u32);

impl PhysMaterialId {
    /// The default material ID.
    pub const DEFAULT: PhysMaterialId = PhysMaterialId(0);
}

// ---------------------------------------------------------------------------
// Combination rule
// ---------------------------------------------------------------------------

/// How to combine material properties when two materials interact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineRule {
    /// Average of the two values: (a + b) / 2.
    Average,
    /// Minimum of the two values.
    Min,
    /// Maximum of the two values.
    Max,
    /// Product of the two values: a * b.
    Multiply,
    /// Geometric mean: sqrt(a * b).
    GeometricMean,
}

impl CombineRule {
    /// Apply the combination rule to two values.
    pub fn combine(&self, a: f32, b: f32) -> f32 {
        match self {
            CombineRule::Average => (a + b) * 0.5,
            CombineRule::Min => a.min(b),
            CombineRule::Max => a.max(b),
            CombineRule::Multiply => a * b,
            CombineRule::GeometricMean => (a * b).abs().sqrt(),
        }
    }
}

impl Default for CombineRule {
    fn default() -> Self {
        CombineRule::Average
    }
}

// ---------------------------------------------------------------------------
// Contact sound hint
// ---------------------------------------------------------------------------

/// Hint for what audio to play when this material is involved in a contact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContactSoundHint {
    /// No sound.
    Silent,
    /// Soft thud (cloth, rubber, flesh).
    SoftThud,
    /// Hard thud (wood, plastic).
    HardThud,
    /// Metallic clang.
    MetalClang,
    /// Glass clink or shatter.
    GlassImpact,
    /// Stone/concrete impact.
    StoneImpact,
    /// Scraping sound (for sliding contacts).
    Scrape,
    /// Squeaking (rubber on smooth surfaces).
    Squeak,
    /// Crunching (gravel, sand, snow).
    Crunch,
    /// Splashing (water contact).
    Splash,
    /// Ice cracking.
    IceCrack,
    /// Chain or metal rattle.
    Rattle,
    /// Custom sound ID.
    Custom(u32),
}

impl Default for ContactSoundHint {
    fn default() -> Self {
        ContactSoundHint::HardThud
    }
}

// ---------------------------------------------------------------------------
// Surface type
// ---------------------------------------------------------------------------

/// Surface type classification for gameplay purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceType {
    /// Default surface.
    Default,
    /// Ice/frozen surface.
    Ice,
    /// Rubber/high-grip surface.
    Rubber,
    /// Wood surface.
    Wood,
    /// Metal surface.
    Metal,
    /// Glass surface.
    Glass,
    /// Concrete/stone surface.
    Concrete,
    /// Sand/loose surface.
    Sand,
    /// Mud/soft ground.
    Mud,
    /// Grass.
    Grass,
    /// Water surface.
    Water,
    /// Snow.
    Snow,
    /// Gravel.
    Gravel,
    /// Carpet/fabric.
    Fabric,
    /// Plastic.
    Plastic,
    /// Custom surface type.
    Custom(u32),
}

impl Default for SurfaceType {
    fn default() -> Self {
        SurfaceType::Default
    }
}

// ---------------------------------------------------------------------------
// Physics material
// ---------------------------------------------------------------------------

/// A complete physics material definition.
#[derive(Debug, Clone)]
pub struct PhysMaterial {
    /// Unique material ID.
    pub id: PhysMaterialId,
    /// Human-readable name.
    pub name: String,
    /// Static friction coefficient (used when objects are not sliding).
    pub static_friction: f32,
    /// Dynamic (kinetic) friction coefficient (used when objects are sliding).
    pub dynamic_friction: f32,
    /// Restitution / coefficient of restitution (0 = inelastic, 1 = perfectly elastic).
    pub restitution: f32,
    /// Density in kg/m^3 (used for mass computation from volume).
    pub density: f32,
    /// How to combine friction when two materials interact.
    pub friction_combine: CombineRule,
    /// How to combine restitution when two materials interact.
    pub restitution_combine: CombineRule,
    /// Surface type for gameplay classification.
    pub surface_type: SurfaceType,
    /// Sound hint for contact audio.
    pub contact_sound: ContactSoundHint,
    /// Sound hint for sliding audio.
    pub slide_sound: ContactSoundHint,
    /// Sound hint for rolling audio.
    pub roll_sound: ContactSoundHint,
    /// Minimum impact speed to trigger contact sound.
    pub sound_threshold_speed: f32,
    /// Whether this material leaves marks (tire tracks, scratches).
    pub leaves_marks: bool,
    /// Particle effect hint for impacts (e.g., sparks, dust, splashes).
    pub impact_effect: ImpactEffect,
    /// Speed modifier for characters walking on this surface (1.0 = normal).
    pub walk_speed_modifier: f32,
    /// Softness: 0.0 = perfectly rigid, 1.0 = very soft (affects contact stiffness).
    pub softness: f32,
}

/// What visual effect to spawn on impact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpactEffect {
    /// No effect.
    None,
    /// Dust puff.
    Dust,
    /// Sparks (metal on metal).
    Sparks,
    /// Water splash.
    Splash,
    /// Snow puff.
    SnowPuff,
    /// Sand spray.
    SandSpray,
    /// Glass shards.
    GlassShards,
    /// Wood splinters.
    WoodSplinters,
    /// Mud splatter.
    MudSplatter,
    /// Custom effect ID.
    Custom(u32),
}

impl Default for ImpactEffect {
    fn default() -> Self {
        ImpactEffect::None
    }
}

impl Default for PhysMaterial {
    fn default() -> Self {
        Self {
            id: PhysMaterialId::DEFAULT,
            name: "default".to_string(),
            static_friction: DEFAULT_STATIC_FRICTION,
            dynamic_friction: DEFAULT_DYNAMIC_FRICTION,
            restitution: DEFAULT_RESTITUTION,
            density: DEFAULT_DENSITY,
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Average,
            surface_type: SurfaceType::Default,
            contact_sound: ContactSoundHint::HardThud,
            slide_sound: ContactSoundHint::Scrape,
            roll_sound: ContactSoundHint::Silent,
            sound_threshold_speed: 0.5,
            leaves_marks: false,
            impact_effect: ImpactEffect::None,
            walk_speed_modifier: 1.0,
            softness: 0.0,
        }
    }
}

impl PhysMaterial {
    /// Create a new material with the given name and basic properties.
    pub fn new(name: &str, static_friction: f32, dynamic_friction: f32, restitution: f32) -> Self {
        Self {
            id: PhysMaterialId(0), // Assigned by database
            name: name.to_string(),
            static_friction,
            dynamic_friction,
            restitution,
            ..Default::default()
        }
    }

    /// Create an ice material.
    pub fn ice() -> Self {
        Self {
            name: "ice".to_string(),
            static_friction: 0.05,
            dynamic_friction: 0.03,
            restitution: 0.1,
            density: 917.0,
            friction_combine: CombineRule::Min,
            restitution_combine: CombineRule::Average,
            surface_type: SurfaceType::Ice,
            contact_sound: ContactSoundHint::IceCrack,
            slide_sound: ContactSoundHint::Scrape,
            roll_sound: ContactSoundHint::Silent,
            sound_threshold_speed: 0.3,
            leaves_marks: true,
            impact_effect: ImpactEffect::None,
            walk_speed_modifier: 0.7,
            softness: 0.0,
            ..Default::default()
        }
    }

    /// Create a rubber material.
    pub fn rubber() -> Self {
        Self {
            name: "rubber".to_string(),
            static_friction: 1.0,
            dynamic_friction: 0.8,
            restitution: 0.8,
            density: 1200.0,
            friction_combine: CombineRule::Max,
            restitution_combine: CombineRule::Max,
            surface_type: SurfaceType::Rubber,
            contact_sound: ContactSoundHint::SoftThud,
            slide_sound: ContactSoundHint::Squeak,
            roll_sound: ContactSoundHint::Silent,
            sound_threshold_speed: 0.8,
            leaves_marks: true,
            impact_effect: ImpactEffect::None,
            walk_speed_modifier: 1.0,
            softness: 0.3,
            ..Default::default()
        }
    }

    /// Create a wood material.
    pub fn wood() -> Self {
        Self {
            name: "wood".to_string(),
            static_friction: 0.5,
            dynamic_friction: 0.4,
            restitution: 0.3,
            density: 600.0,
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Average,
            surface_type: SurfaceType::Wood,
            contact_sound: ContactSoundHint::HardThud,
            slide_sound: ContactSoundHint::Scrape,
            roll_sound: ContactSoundHint::Silent,
            sound_threshold_speed: 0.5,
            leaves_marks: true,
            impact_effect: ImpactEffect::WoodSplinters,
            walk_speed_modifier: 1.0,
            softness: 0.05,
            ..Default::default()
        }
    }

    /// Create a metal material.
    pub fn metal() -> Self {
        Self {
            name: "metal".to_string(),
            static_friction: 0.6,
            dynamic_friction: 0.4,
            restitution: 0.4,
            density: 7800.0,
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Average,
            surface_type: SurfaceType::Metal,
            contact_sound: ContactSoundHint::MetalClang,
            slide_sound: ContactSoundHint::Scrape,
            roll_sound: ContactSoundHint::Rattle,
            sound_threshold_speed: 0.3,
            leaves_marks: true,
            impact_effect: ImpactEffect::Sparks,
            walk_speed_modifier: 1.0,
            softness: 0.0,
            ..Default::default()
        }
    }

    /// Create a glass material.
    pub fn glass() -> Self {
        Self {
            name: "glass".to_string(),
            static_friction: 0.4,
            dynamic_friction: 0.3,
            restitution: 0.5,
            density: 2500.0,
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Max,
            surface_type: SurfaceType::Glass,
            contact_sound: ContactSoundHint::GlassImpact,
            slide_sound: ContactSoundHint::Squeak,
            roll_sound: ContactSoundHint::Silent,
            sound_threshold_speed: 0.2,
            leaves_marks: false,
            impact_effect: ImpactEffect::GlassShards,
            walk_speed_modifier: 1.0,
            softness: 0.0,
            ..Default::default()
        }
    }

    /// Create a concrete material.
    pub fn concrete() -> Self {
        Self {
            name: "concrete".to_string(),
            static_friction: 0.7,
            dynamic_friction: 0.6,
            restitution: 0.2,
            density: 2400.0,
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Average,
            surface_type: SurfaceType::Concrete,
            contact_sound: ContactSoundHint::StoneImpact,
            slide_sound: ContactSoundHint::Scrape,
            roll_sound: ContactSoundHint::Silent,
            sound_threshold_speed: 0.5,
            leaves_marks: true,
            impact_effect: ImpactEffect::Dust,
            walk_speed_modifier: 1.0,
            softness: 0.0,
            ..Default::default()
        }
    }

    /// Create a sand material.
    pub fn sand() -> Self {
        Self {
            name: "sand".to_string(),
            static_friction: 0.6,
            dynamic_friction: 0.5,
            restitution: 0.05,
            density: 1600.0,
            friction_combine: CombineRule::Max,
            restitution_combine: CombineRule::Min,
            surface_type: SurfaceType::Sand,
            contact_sound: ContactSoundHint::Crunch,
            slide_sound: ContactSoundHint::Crunch,
            roll_sound: ContactSoundHint::Crunch,
            sound_threshold_speed: 0.3,
            leaves_marks: true,
            impact_effect: ImpactEffect::SandSpray,
            walk_speed_modifier: 0.8,
            softness: 0.4,
            ..Default::default()
        }
    }

    /// Create a mud material.
    pub fn mud() -> Self {
        Self {
            name: "mud".to_string(),
            static_friction: 0.4,
            dynamic_friction: 0.3,
            restitution: 0.02,
            density: 1800.0,
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Min,
            surface_type: SurfaceType::Mud,
            contact_sound: ContactSoundHint::Splash,
            slide_sound: ContactSoundHint::Splash,
            roll_sound: ContactSoundHint::Splash,
            sound_threshold_speed: 0.2,
            leaves_marks: true,
            impact_effect: ImpactEffect::MudSplatter,
            walk_speed_modifier: 0.6,
            softness: 0.6,
            ..Default::default()
        }
    }

    /// Create a snow material.
    pub fn snow() -> Self {
        Self {
            name: "snow".to_string(),
            static_friction: 0.3,
            dynamic_friction: 0.2,
            restitution: 0.05,
            density: 400.0,
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Min,
            surface_type: SurfaceType::Snow,
            contact_sound: ContactSoundHint::Crunch,
            slide_sound: ContactSoundHint::Crunch,
            roll_sound: ContactSoundHint::Crunch,
            sound_threshold_speed: 0.2,
            leaves_marks: true,
            impact_effect: ImpactEffect::SnowPuff,
            walk_speed_modifier: 0.75,
            softness: 0.5,
            ..Default::default()
        }
    }

    /// Create a gravel material.
    pub fn gravel() -> Self {
        Self {
            name: "gravel".to_string(),
            static_friction: 0.7,
            dynamic_friction: 0.6,
            restitution: 0.1,
            density: 1800.0,
            friction_combine: CombineRule::Max,
            restitution_combine: CombineRule::Min,
            surface_type: SurfaceType::Gravel,
            contact_sound: ContactSoundHint::Crunch,
            slide_sound: ContactSoundHint::Crunch,
            roll_sound: ContactSoundHint::Crunch,
            sound_threshold_speed: 0.3,
            leaves_marks: true,
            impact_effect: ImpactEffect::Dust,
            walk_speed_modifier: 0.85,
            softness: 0.2,
            ..Default::default()
        }
    }

    /// Create a fabric/carpet material.
    pub fn fabric() -> Self {
        Self {
            name: "fabric".to_string(),
            static_friction: 0.5,
            dynamic_friction: 0.4,
            restitution: 0.1,
            density: 300.0,
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Min,
            surface_type: SurfaceType::Fabric,
            contact_sound: ContactSoundHint::SoftThud,
            slide_sound: ContactSoundHint::Silent,
            roll_sound: ContactSoundHint::Silent,
            sound_threshold_speed: 1.0,
            leaves_marks: false,
            impact_effect: ImpactEffect::Dust,
            walk_speed_modifier: 0.95,
            softness: 0.4,
            ..Default::default()
        }
    }

    /// Set the density and return self for builder pattern.
    pub fn with_density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }

    /// Set the sound threshold and return self.
    pub fn with_sound_threshold(mut self, threshold: f32) -> Self {
        self.sound_threshold_speed = threshold;
        self
    }
}

// ---------------------------------------------------------------------------
// Combined material (result of two materials interacting)
// ---------------------------------------------------------------------------

/// The result of combining two materials for a contact pair.
#[derive(Debug, Clone)]
pub struct CombinedMaterial {
    /// Combined static friction coefficient.
    pub static_friction: f32,
    /// Combined dynamic friction coefficient.
    pub dynamic_friction: f32,
    /// Combined restitution.
    pub restitution: f32,
    /// Contact sound hint (from the harder/louder material).
    pub contact_sound: ContactSoundHint,
    /// Slide sound hint.
    pub slide_sound: ContactSoundHint,
    /// Impact effect (from the material more likely to produce effects).
    pub impact_effect: ImpactEffect,
    /// Average softness.
    pub softness: f32,
    /// IDs of the two materials that were combined.
    pub material_a: PhysMaterialId,
    /// ID of material B.
    pub material_b: PhysMaterialId,
}

impl CombinedMaterial {
    /// Combine two materials using their respective combination rules.
    pub fn combine(a: &PhysMaterial, b: &PhysMaterial) -> Self {
        // Use the higher-priority combine rule between the two materials.
        // Priority: Max > Multiply > Average > Min
        let friction_rule = Self::higher_priority_rule(a.friction_combine, b.friction_combine);
        let restitution_rule =
            Self::higher_priority_rule(a.restitution_combine, b.restitution_combine);

        let static_friction = friction_rule.combine(a.static_friction, b.static_friction);
        let dynamic_friction = friction_rule.combine(a.dynamic_friction, b.dynamic_friction);
        let restitution = restitution_rule.combine(a.restitution, b.restitution);

        // Sound: use the harder material's sound
        let contact_sound = if a.density >= b.density {
            a.contact_sound
        } else {
            b.contact_sound
        };

        let slide_sound = if a.dynamic_friction >= b.dynamic_friction {
            a.slide_sound
        } else {
            b.slide_sound
        };

        // Impact effect: prefer the more dramatic one
        let impact_effect = Self::choose_impact_effect(a.impact_effect, b.impact_effect);

        let softness = (a.softness + b.softness) * 0.5;

        CombinedMaterial {
            static_friction,
            dynamic_friction,
            restitution,
            contact_sound,
            slide_sound,
            impact_effect,
            softness,
            material_a: a.id,
            material_b: b.id,
        }
    }

    /// Determine the higher-priority combination rule.
    fn higher_priority_rule(a: CombineRule, b: CombineRule) -> CombineRule {
        let priority = |r: CombineRule| -> u8 {
            match r {
                CombineRule::Min => 0,
                CombineRule::Average => 1,
                CombineRule::GeometricMean => 2,
                CombineRule::Multiply => 3,
                CombineRule::Max => 4,
            }
        };
        if priority(a) >= priority(b) { a } else { b }
    }

    /// Choose the more "dramatic" impact effect.
    fn choose_impact_effect(a: ImpactEffect, b: ImpactEffect) -> ImpactEffect {
        if a == ImpactEffect::None {
            return b;
        }
        if b == ImpactEffect::None {
            return a;
        }
        // Prefer sparks and splashes over dust
        match (a, b) {
            (ImpactEffect::Sparks, _) | (_, ImpactEffect::Sparks) => ImpactEffect::Sparks,
            (ImpactEffect::Splash, _) | (_, ImpactEffect::Splash) => ImpactEffect::Splash,
            (ImpactEffect::GlassShards, _) | (_, ImpactEffect::GlassShards) => {
                ImpactEffect::GlassShards
            }
            _ => a,
        }
    }
}

// ---------------------------------------------------------------------------
// Material override (per-pair)
// ---------------------------------------------------------------------------

/// Override material properties for a specific pair of materials.
#[derive(Debug, Clone)]
pub struct MaterialPairOverride {
    /// First material ID.
    pub material_a: PhysMaterialId,
    /// Second material ID.
    pub material_b: PhysMaterialId,
    /// Override static friction (if Some).
    pub static_friction: Option<f32>,
    /// Override dynamic friction (if Some).
    pub dynamic_friction: Option<f32>,
    /// Override restitution (if Some).
    pub restitution: Option<f32>,
    /// Override contact sound (if Some).
    pub contact_sound: Option<ContactSoundHint>,
    /// Override impact effect (if Some).
    pub impact_effect: Option<ImpactEffect>,
}

// ---------------------------------------------------------------------------
// Material database
// ---------------------------------------------------------------------------

/// A database of physics materials with lookup by name or ID.
#[derive(Debug)]
pub struct MaterialDatabase {
    /// Materials indexed by their ID.
    materials: HashMap<PhysMaterialId, PhysMaterial>,
    /// Name to ID mapping for lookup by name.
    name_to_id: HashMap<String, PhysMaterialId>,
    /// Per-pair overrides.
    pair_overrides: Vec<MaterialPairOverride>,
    /// Next material ID to assign.
    next_id: u32,
    /// Cached combined materials for performance.
    combined_cache: HashMap<(u32, u32), CombinedMaterial>,
    /// Whether the cache is dirty (needs rebuild).
    cache_dirty: bool,
}

impl MaterialDatabase {
    /// Create a new empty material database.
    pub fn new() -> Self {
        let mut db = Self {
            materials: HashMap::new(),
            name_to_id: HashMap::new(),
            pair_overrides: Vec::new(),
            next_id: 1, // 0 is reserved for default
            combined_cache: HashMap::new(),
            cache_dirty: true,
        };

        // Register the default material
        let mut default_mat = PhysMaterial::default();
        default_mat.id = PhysMaterialId::DEFAULT;
        db.materials.insert(PhysMaterialId::DEFAULT, default_mat);
        db.name_to_id.insert("default".to_string(), PhysMaterialId::DEFAULT);

        db
    }

    /// Create a database pre-populated with all built-in materials.
    pub fn with_builtins() -> Self {
        let mut db = Self::new();
        db.register(PhysMaterial::ice());
        db.register(PhysMaterial::rubber());
        db.register(PhysMaterial::wood());
        db.register(PhysMaterial::metal());
        db.register(PhysMaterial::glass());
        db.register(PhysMaterial::concrete());
        db.register(PhysMaterial::sand());
        db.register(PhysMaterial::mud());
        db.register(PhysMaterial::snow());
        db.register(PhysMaterial::gravel());
        db.register(PhysMaterial::fabric());
        db
    }

    /// Register a new material in the database. Returns its assigned ID.
    pub fn register(&mut self, mut material: PhysMaterial) -> PhysMaterialId {
        let id = PhysMaterialId(self.next_id);
        self.next_id += 1;

        material.id = id;
        self.name_to_id.insert(material.name.clone(), id);
        self.materials.insert(id, material);
        self.cache_dirty = true;

        id
    }

    /// Look up a material by ID.
    pub fn get(&self, id: PhysMaterialId) -> Option<&PhysMaterial> {
        self.materials.get(&id)
    }

    /// Look up a material by name.
    pub fn get_by_name(&self, name: &str) -> Option<&PhysMaterial> {
        self.name_to_id
            .get(name)
            .and_then(|id| self.materials.get(id))
    }

    /// Get the ID for a material name.
    pub fn id_for_name(&self, name: &str) -> Option<PhysMaterialId> {
        self.name_to_id.get(name).copied()
    }

    /// Get or create the default material.
    pub fn default_material(&self) -> &PhysMaterial {
        self.materials.get(&PhysMaterialId::DEFAULT).unwrap()
    }

    /// Get the number of registered materials.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Get all material names.
    pub fn material_names(&self) -> Vec<&str> {
        self.materials.values().map(|m| m.name.as_str()).collect()
    }

    /// Get all material IDs.
    pub fn material_ids(&self) -> Vec<PhysMaterialId> {
        self.materials.keys().copied().collect()
    }

    /// Add a per-pair material override.
    pub fn add_pair_override(&mut self, override_: MaterialPairOverride) {
        self.pair_overrides.push(override_);
        self.cache_dirty = true;
    }

    /// Combine two materials and return the contact properties.
    pub fn combine(&self, id_a: PhysMaterialId, id_b: PhysMaterialId) -> CombinedMaterial {
        let mat_a = self.materials.get(&id_a).unwrap_or(
            self.materials.get(&PhysMaterialId::DEFAULT).unwrap(),
        );
        let mat_b = self.materials.get(&id_b).unwrap_or(
            self.materials.get(&PhysMaterialId::DEFAULT).unwrap(),
        );

        let mut combined = CombinedMaterial::combine(mat_a, mat_b);

        // Apply pair overrides
        for override_ in &self.pair_overrides {
            let matches = (override_.material_a == id_a && override_.material_b == id_b)
                || (override_.material_a == id_b && override_.material_b == id_a);

            if matches {
                if let Some(sf) = override_.static_friction {
                    combined.static_friction = sf;
                }
                if let Some(df) = override_.dynamic_friction {
                    combined.dynamic_friction = df;
                }
                if let Some(r) = override_.restitution {
                    combined.restitution = r;
                }
                if let Some(sound) = override_.contact_sound {
                    combined.contact_sound = sound;
                }
                if let Some(effect) = override_.impact_effect {
                    combined.impact_effect = effect;
                }
            }
        }

        combined
    }

    /// Combine two materials with caching for performance.
    pub fn combine_cached(&mut self, id_a: PhysMaterialId, id_b: PhysMaterialId) -> &CombinedMaterial {
        if self.cache_dirty {
            self.combined_cache.clear();
            self.cache_dirty = false;
        }

        // Normalize key order so (a,b) and (b,a) map to the same entry
        let key = if id_a.0 <= id_b.0 {
            (id_a.0, id_b.0)
        } else {
            (id_b.0, id_a.0)
        };

        if !self.combined_cache.contains_key(&key) {
            let combined = self.combine(id_a, id_b);
            self.combined_cache.insert(key, combined);
        }

        self.combined_cache.get(&key).unwrap()
    }

    /// Remove a material from the database.
    pub fn remove(&mut self, id: PhysMaterialId) {
        if id == PhysMaterialId::DEFAULT {
            return; // Cannot remove the default material
        }
        if let Some(mat) = self.materials.remove(&id) {
            self.name_to_id.remove(&mat.name);
            self.cache_dirty = true;
        }
    }

    /// Update a material's friction values.
    pub fn set_friction(
        &mut self,
        id: PhysMaterialId,
        static_friction: f32,
        dynamic_friction: f32,
    ) {
        if let Some(mat) = self.materials.get_mut(&id) {
            mat.static_friction = static_friction;
            mat.dynamic_friction = dynamic_friction;
            self.cache_dirty = true;
        }
    }

    /// Update a material's restitution.
    pub fn set_restitution(&mut self, id: PhysMaterialId, restitution: f32) {
        if let Some(mat) = self.materials.get_mut(&id) {
            mat.restitution = restitution;
            self.cache_dirty = true;
        }
    }

    /// Get the contact sound for a material pair interaction at a given impact speed.
    pub fn contact_sound_for_pair(
        &self,
        id_a: PhysMaterialId,
        id_b: PhysMaterialId,
        impact_speed: f32,
    ) -> Option<ContactSoundHint> {
        let mat_a = self.materials.get(&id_a)?;
        let mat_b = self.materials.get(&id_b)?;

        let threshold = mat_a.sound_threshold_speed.min(mat_b.sound_threshold_speed);
        if impact_speed < threshold {
            return None;
        }

        let combined = CombinedMaterial::combine(mat_a, mat_b);
        Some(combined.contact_sound)
    }

    /// Iterate over all materials.
    pub fn iter(&self) -> impl Iterator<Item = &PhysMaterial> {
        self.materials.values()
    }
}
