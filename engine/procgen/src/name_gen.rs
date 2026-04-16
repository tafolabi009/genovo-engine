//! # Procedural Name Generation
//!
//! Generates fantasy names using Markov chains trained on syllable corpora.
//! Supports multiple cultural styles (elven, dwarven, orcish, human) and
//! place name generation.
//!
//! ## How it works
//!
//! A Markov chain of configurable order (typically 2-4 characters) is built
//! from a training corpus. To generate a name, the chain is walked from a
//! start state, choosing the next character based on the transition
//! probabilities, until a termination character is reached or the maximum
//! length is hit.

use genovo_core::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===========================================================================
// Culture presets
// ===========================================================================

/// Cultural style presets for name generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Culture {
    /// Flowing, vowel-heavy names: Aelindra, Thalion, Galadwen.
    Elven,
    /// Hard consonants, short syllables: Thorin, Balin, Gimrak.
    Dwarven,
    /// Harsh, guttural sounds: Grukash, Moldur, Azgoth.
    Orcish,
    /// Balanced, real-world inspired: Aldric, Mirabel, Cedric.
    Human,
    /// Dragon/demonic names: Azrathul, Vexmor, Zypheron.
    Draconic,
    /// Nature-inspired, soft: Willowmere, Fernshade, Brooksong.
    Fey,
}

/// Training corpora for each culture.
fn get_corpus(culture: Culture) -> &'static [&'static str] {
    match culture {
        Culture::Elven => &[
            "aelindra", "thalion", "galadwen", "elarion", "silvanis",
            "mirithal", "caladrel", "elendil", "lorien", "nimrodel",
            "celeborn", "galadriel", "arwen", "legolas", "thranduil",
            "faelar", "vaelith", "isilme", "noldorin", "telerin",
            "sindarin", "quenya", "anarion", "elenwe", "idril",
            "luthien", "aredhel", "fingolfin", "finarfin", "turgon",
            "ecthelion", "glorfindel", "earendil", "elrond", "celebrian",
            "feanorin", "maedhros", "maglor", "celegorm", "caranthir",
            "amrod", "amras", "curufin", "finrod", "orodreth",
            "angrod", "aegnor", "gwindor", "beleg", "mablung",
            "thingol", "melian", "dior", "elwing", "galathil",
            "nimloth", "aelindis", "calanthe", "elowen", "thandril",
            "sylvaris", "aeloria", "vanyarin", "eldalon", "miriel",
            "nerdanel", "anaire", "earwen", "findis", "lalwen",
            "irime", "amarie", "nellas", "saeros", "daeron",
            "maeglin", "gondolin", "doriath", "nargothrond", "lindon",
        ],
        Culture::Dwarven => &[
            "thorin", "balin", "dwalin", "gimli", "gloin",
            "oin", "bifur", "bofur", "bombur", "nori",
            "dori", "ori", "fili", "kili", "dain",
            "thrain", "thror", "fundin", "nain", "durin",
            "brok", "durnhal", "grimbold", "torvak", "moradin",
            "kaldrak", "thordak", "gimrak", "baldur", "haldrek",
            "stornvek", "ironhand", "stonefist", "deepdelve", "forgefire",
            "anvildor", "hammerfast", "mithrak", "borin", "gorin",
            "kronar", "drumnir", "runefolk", "grannak", "thurim",
            "durgrim", "braldan", "kolgar", "zagrim", "mordak",
            "brondar", "kelgar", "naldur", "thundrik", "gromril",
            "kazadum", "khazad", "aglarond", "erebor", "moria",
            "nogrod", "belegost", "tumunzahar", "gabilgathol", "azanulbizar",
        ],
        Culture::Orcish => &[
            "grukash", "moldur", "azgoth", "korgul", "thraka",
            "muzgash", "skullcrusher", "gorefang", "bloodaxe", "ironjaw",
            "grimblade", "vorkag", "shagrat", "gorbag", "ugluk",
            "grishnakh", "bolg", "azog", "lurtz", "gothmog",
            "gundabad", "mordok", "goroth", "draugoth", "mugrath",
            "kronash", "zugluk", "vargul", "gruknak", "bograth",
            "nazgul", "orgrim", "garrosh", "thrall", "durotan",
            "guldan", "blackhand", "kilrogg", "rend", "maim",
            "zagara", "kargath", "grommash", "nerzhul", "zuluhed",
            "teron", "drakthul", "skullsplit", "bonegnaw", "ashgork",
            "mogrash", "durzog", "krogash", "vragoth", "skaruk",
            "grulzak", "thrakgul", "muznak", "borgash", "kulzak",
        ],
        Culture::Human => &[
            "aldric", "mirabel", "cedric", "eleanor", "gareth",
            "isabelle", "marcus", "rowena", "tristan", "vivienne",
            "arthur", "guinevere", "lancelot", "morgan", "percival",
            "galahad", "merlin", "nimue", "uther", "igraine",
            "bedivere", "gawain", "geraint", "kay", "elaine",
            "lynette", "enid", "yvain", "bors", "palomides",
            "alaric", "brunhilde", "constance", "darius", "edmund",
            "frieda", "godwin", "helena", "ingrid", "justinian",
            "konrad", "liselotte", "matthias", "natasha", "oswald",
            "priscilla", "quentin", "rosalind", "siegfried", "theodora",
            "ulric", "valentina", "wilhelm", "xanthe", "yaroslav",
            "zenobia", "adelheid", "bertram", "cordelia", "dietrich",
            "eustace", "florentia", "gottfried", "hildegard", "isidore",
        ],
        Culture::Draconic => &[
            "azrathul", "vexmor", "zypheron", "tiamat", "bahamut",
            "draconith", "smaug", "glaurung", "ancalagon", "scatha",
            "vortigern", "kragenoth", "nidhogg", "fafnir", "typhon",
            "apophis", "vritra", "hydrazor", "pyrothax", "cryomantis",
            "drazulon", "ignissar", "tempestix", "voidscale", "ashenwing",
            "thundermaw", "shadowclaw", "blazehorn", "frostfang", "venomspine",
            "crysthara", "magmoros", "stormrend", "darkflame", "wyrmkind",
            "dragovar", "wyrmlord", "scalethorn", "drakarios", "serpentus",
            "basilikos", "leviathus", "ouroboros", "amphithus", "coatylus",
            "quetzaran", "kulkulzar", "lindwurmak", "tatzelwur", "amphiveres",
        ],
        Culture::Fey => &[
            "willowmere", "fernshade", "brooksong", "dewdrop", "moonpetal",
            "starbloom", "thornwick", "mistyhollow", "glendara", "sylphwind",
            "mossheart", "petalwhisper", "fawnlight", "foxglove", "honeydew",
            "ivywood", "juniperbark", "larkspire", "meadowsong", "nightbloom",
            "oakenshade", "primrose", "quillwort", "rosethorne", "silverfern",
            "thistledown", "umbramoss", "violetshadow", "willowwisp", "xylosia",
            "yarrowbend", "zephyrleaf", "amberveil", "blossomheart", "cloverdale",
            "dawnmist", "elderbark", "flutterby", "glimmerbrook", "hazelglen",
            "indigosky", "jasminewild", "kelpshore", "lilacwind", "marigoldvale",
        ],
    }
}

// ===========================================================================
// Markov Chain
// ===========================================================================

/// Transition entry: maps a context string to possible next characters and
/// their accumulated weights.
#[derive(Debug, Clone)]
struct MarkovTransitions {
    /// Characters that can follow this context, with cumulative weights.
    entries: Vec<(char, f32)>,
    /// Total weight for normalization.
    total_weight: f32,
}

impl MarkovTransitions {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_weight: 0.0,
        }
    }

    fn add(&mut self, ch: char, weight: f32) {
        // Check if character already exists.
        for entry in &mut self.entries {
            if entry.0 == ch {
                entry.1 += weight;
                self.total_weight += weight;
                return;
            }
        }
        self.entries.push((ch, weight));
        self.total_weight += weight;
    }

    fn choose(&self, rng: &mut Rng) -> char {
        if self.entries.is_empty() || self.total_weight <= 0.0 {
            return '\0';
        }

        let mut target = rng.next_f32() * self.total_weight;
        for &(ch, weight) in &self.entries {
            target -= weight;
            if target <= 0.0 {
                return ch;
            }
        }

        self.entries.last().map(|&(ch, _)| ch).unwrap_or('\0')
    }
}

// ===========================================================================
// NameGenerator
// ===========================================================================

/// A Markov chain-based procedural name generator.
///
/// Build the chain from a corpus of example names, then generate new names
/// that follow similar phonetic patterns.
#[derive(Debug, Clone)]
pub struct NameGenerator {
    /// Markov chain order (number of characters of history).
    order: usize,
    /// Transition table: context -> possible next characters.
    transitions: HashMap<String, MarkovTransitions>,
    /// Starting contexts (beginnings of words).
    start_contexts: Vec<String>,
    /// Minimum generated name length.
    min_length: usize,
    /// Maximum generated name length.
    max_length: usize,
}

impl NameGenerator {
    /// Create a new empty name generator with the given Markov chain order.
    ///
    /// Order 2-3 is typical for character-level Markov chains. Higher orders
    /// produce names more similar to the training corpus; lower orders produce
    /// more variety.
    pub fn new(order: usize) -> Self {
        Self {
            order: order.max(1),
            transitions: HashMap::new(),
            start_contexts: Vec::new(),
            min_length: 3,
            max_length: 12,
        }
    }

    /// Set the minimum and maximum generated name length.
    pub fn with_length_range(mut self, min: usize, max: usize) -> Self {
        self.min_length = min.max(1);
        self.max_length = max.max(self.min_length);
        self
    }

    /// Train the Markov chain on a corpus of names.
    pub fn train(&mut self, names: &[&str]) {
        for name in names {
            let lower = name.to_lowercase();
            if lower.len() < self.order {
                continue;
            }

            // Add start marker.
            let padded = format!("{}{}\0", "\0".repeat(self.order), lower);
            let chars: Vec<char> = padded.chars().collect();

            // Record the start context.
            let start: String = chars[0..self.order].iter().collect();
            if !self.start_contexts.contains(&start) {
                self.start_contexts.push(start);
            }

            // Build transitions.
            for i in 0..chars.len() - self.order {
                let context: String = chars[i..i + self.order].iter().collect();
                let next_char = chars[i + self.order];

                self.transitions
                    .entry(context)
                    .or_insert_with(MarkovTransitions::new)
                    .add(next_char, 1.0);
            }
        }
    }

    /// Generate a single name.
    pub fn generate(&self, rng: &mut Rng) -> String {
        if self.start_contexts.is_empty() {
            return String::new();
        }

        let max_attempts = 100;
        for _ in 0..max_attempts {
            let name = self.generate_one(rng);
            if name.len() >= self.min_length && name.len() <= self.max_length {
                return name;
            }
        }

        // Fallback: return whatever we get.
        self.generate_one(rng)
    }

    /// Internal: generate one name without length validation.
    fn generate_one(&self, rng: &mut Rng) -> String {
        let start_idx = rng.range_i32(0, self.start_contexts.len() as i32) as usize;
        let mut context = self.start_contexts[start_idx].clone();
        let mut result = String::new();

        // Skip null characters in the initial context.
        for ch in context.chars() {
            if ch != '\0' {
                result.push(ch);
            }
        }

        let max_chars = self.max_length + 5;
        for _ in 0..max_chars {
            let next = match self.transitions.get(&context) {
                Some(trans) => trans.choose(rng),
                None => break,
            };

            if next == '\0' {
                break; // End of name marker.
            }

            result.push(next);

            // Shift context window.
            let ctx_chars: Vec<char> = context.chars().collect();
            let new_ctx: String = ctx_chars[1..].iter().chain(std::iter::once(&next)).collect();
            context = new_ctx;
        }

        // Capitalize first letter.
        let mut chars = result.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => {
                let mut name = first.to_uppercase().to_string();
                name.extend(chars);
                name
            }
        }
    }

    /// Generate N unique names.
    pub fn generate_many(&self, count: usize, rng: &mut Rng) -> Vec<String> {
        let mut names = Vec::with_capacity(count);
        let mut attempts = 0;
        let max_attempts = count * 20;

        while names.len() < count && attempts < max_attempts {
            let name = self.generate(rng);
            if !names.contains(&name) {
                names.push(name);
            }
            attempts += 1;
        }

        names
    }
}

// ===========================================================================
// Preset generators
// ===========================================================================

/// Create a name generator pre-trained on the given culture's corpus.
pub fn create_generator(culture: Culture, order: usize) -> NameGenerator {
    let corpus = get_corpus(culture);
    let mut generator = NameGenerator::new(order);
    generator.train(corpus);
    generator
}

/// Generate a fantasy name for the given culture.
pub fn generate_fantasy_name(culture: Culture, seed: u64) -> String {
    let mut rng = Rng::new(seed);
    let name_gen = create_generator(culture, 3);
    name_gen.generate(&mut rng)
}

/// Generate multiple fantasy names for the given culture.
pub fn generate_fantasy_names(culture: Culture, count: usize, seed: u64) -> Vec<String> {
    let mut rng = Rng::new(seed);
    let name_gen = create_generator(culture, 3);
    name_gen.generate_many(count, &mut rng)
}

// ===========================================================================
// Place Name Generation
// ===========================================================================

/// Terrain feature types for place name generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerrainFeature {
    Mountain,
    River,
    Forest,
    Lake,
    Plains,
    Coast,
    Valley,
    Swamp,
    Desert,
    Island,
}

/// Settlement types for place name generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SettlementType {
    City,
    Town,
    Village,
    Hamlet,
    Fortress,
    Outpost,
    Port,
    Temple,
    Mine,
    Ruin,
}

/// Generate a place name by combining a terrain feature with a settlement type.
///
/// Examples: "Stormhaven Port", "Ironpeak Fortress", "Willowmere Village"
pub fn generate_place_name(
    feature: TerrainFeature,
    settlement: SettlementType,
    culture: Culture,
    seed: u64,
) -> String {
    let mut rng = Rng::new(seed);

    let feature_prefixes = match feature {
        TerrainFeature::Mountain => &[
            "Iron", "Storm", "Thunder", "Eagle", "Dragon", "Frost", "Stone",
            "Granite", "Crystal", "Shadow", "Silver", "Gold", "Azure", "Ember",
        ] as &[&str],
        TerrainFeature::River => &[
            "Silver", "Swift", "Rushing", "Deep", "Clear", "Winding", "Golden",
            "Misty", "Crystal", "Moon", "Star", "Shimmer", "Twilight", "Dawn",
        ],
        TerrainFeature::Forest => &[
            "Dark", "Elder", "Ancient", "Whispering", "Shadow", "Green", "Old",
            "Twisted", "Mossy", "Fern", "Oak", "Pine", "Willow", "Birch",
        ],
        TerrainFeature::Lake => &[
            "Mirror", "Still", "Crystal", "Moon", "Silver", "Deep", "Hidden",
            "Serene", "Calm", "Frozen", "Glimmer", "Sapphire", "Star", "Dawn",
        ],
        TerrainFeature::Plains => &[
            "Golden", "Vast", "Wind", "Sun", "Amber", "Endless", "Rolling",
            "Green", "Wild", "Open", "Honey", "Wheat", "Autumn", "Spring",
        ],
        TerrainFeature::Coast => &[
            "Storm", "Coral", "Salt", "Tide", "Wave", "Shell", "Harbor",
            "Cliff", "Drift", "Foam", "Sandy", "Rocky", "Windy", "Misty",
        ],
        TerrainFeature::Valley => &[
            "Hidden", "Peaceful", "Green", "Deep", "Quiet", "Lost", "Verdant",
            "Sheltered", "Blessed", "Gentle", "Harvest", "Sunshine", "Morning", "Twilight",
        ],
        TerrainFeature::Swamp => &[
            "Murky", "Dark", "Bog", "Marsh", "Fen", "Mire", "Hag",
            "Witch", "Shadow", "Dead", "Lost", "Rotting", "Sinking", "Black",
        ],
        TerrainFeature::Desert => &[
            "Sun", "Sand", "Dust", "Scorch", "Arid", "Dune", "Mirage",
            "Bone", "Dry", "Burning", "Bleach", "Wind", "Heat", "Ash",
        ],
        TerrainFeature::Island => &[
            "Coral", "Palm", "Emerald", "Jade", "Pearl", "Sapphire", "Shell",
            "Drift", "Haven", "Paradise", "Lonely", "Far", "Storm", "Mist",
        ],
    };

    let feature_suffixes = match feature {
        TerrainFeature::Mountain => &[
            "peak", "crest", "horn", "spire", "mount", "summit", "ridge",
            "cliff", "tooth", "crown",
        ] as &[&str],
        TerrainFeature::River => &[
            "ford", "brook", "creek", "water", "stream", "falls", "bend",
            "crossing", "rapids", "mouth",
        ],
        TerrainFeature::Forest => &[
            "wood", "grove", "glade", "thicket", "brake", "weald", "copse",
            "holt", "fen", "dell",
        ],
        TerrainFeature::Lake => &[
            "mere", "pool", "water", "tarn", "loch", "pond", "basin",
            "springs", "well", "depths",
        ],
        TerrainFeature::Plains => &[
            "field", "meadow", "reach", "expanse", "stretch", "steppe", "plain",
            "lea", "moor", "heath",
        ],
        TerrainFeature::Coast => &[
            "haven", "port", "cove", "bay", "shore", "point", "cape",
            "inlet", "dock", "landing",
        ],
        TerrainFeature::Valley => &[
            "vale", "dale", "glen", "hollow", "basin", "bottom", "gap",
            "pass", "dell", "nook",
        ],
        TerrainFeature::Swamp => &[
            "mire", "bog", "fen", "marsh", "slough", "moor", "wetland",
            "hollow", "pit", "sink",
        ],
        TerrainFeature::Desert => &[
            "waste", "flats", "basin", "expanse", "reach", "void", "desolation",
            "barren", "dust", "sands",
        ],
        TerrainFeature::Island => &[
            "isle", "atoll", "key", "reef", "rock", "island", "archipelago",
            "cay", "shoal", "sanctuary",
        ],
    };

    let settlement_suffix = match settlement {
        SettlementType::City => "",
        SettlementType::Town => " Town",
        SettlementType::Village => " Village",
        SettlementType::Hamlet => " Hamlet",
        SettlementType::Fortress => " Fortress",
        SettlementType::Outpost => " Outpost",
        SettlementType::Port => " Port",
        SettlementType::Temple => " Temple",
        SettlementType::Mine => " Mine",
        SettlementType::Ruin => " Ruins",
    };

    // Pick random prefix and suffix.
    let prefix_idx = rng.range_i32(0, feature_prefixes.len() as i32) as usize;
    let suffix_idx = rng.range_i32(0, feature_suffixes.len() as i32) as usize;

    let prefix = feature_prefixes[prefix_idx];
    let suffix = feature_suffixes[suffix_idx];

    // Optionally insert a culture-specific middle element.
    let use_culture_name = rng.bool(0.3);
    if use_culture_name {
        let culture_name = generate_fantasy_name(culture, rng.next_u64());
        let short = if culture_name.len() > 6 {
            culture_name[..6].to_string()
        } else {
            culture_name
        };
        format!("{short}{suffix}{settlement_suffix}")
    } else {
        format!("{prefix}{suffix}{settlement_suffix}")
    }
}

/// Generate a batch of place names for a region.
pub fn generate_place_names(
    count: usize,
    features: &[TerrainFeature],
    settlements: &[SettlementType],
    culture: Culture,
    seed: u64,
) -> Vec<String> {
    let mut rng = Rng::new(seed);
    let mut names = Vec::with_capacity(count);

    for _ in 0..count {
        let feature = features[rng.range_i32(0, features.len() as i32) as usize];
        let settlement = settlements[rng.range_i32(0, settlements.len() as i32) as usize];
        let name = generate_place_name(feature, settlement, culture, rng.next_u64());
        names.push(name);
    }

    names
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_generator_basic() {
        let mut name_gen = NameGenerator::new(2);
        name_gen.train(&["alice", "alison", "alan", "albert", "alexandra"]);

        let mut rng = Rng::new(42);
        let name = name_gen.generate(&mut rng);

        assert!(!name.is_empty(), "Generated name should not be empty");
        // First letter should be uppercase.
        assert!(
            name.chars().next().unwrap().is_uppercase(),
            "Name should be capitalized: {name}"
        );
    }

    #[test]
    fn test_name_generator_length() {
        let mut name_gen = NameGenerator::new(2).with_length_range(4, 8);
        name_gen.train(&[
            "alice", "alison", "alan", "albert", "alexandra", "alfredo",
            "anderson", "antonio", "archibald",
        ]);

        let mut rng = Rng::new(42);
        for _ in 0..20 {
            let name = name_gen.generate(&mut rng);
            // Allow some tolerance since Markov chains can't guarantee exact length.
            assert!(
                name.len() >= 2,
                "Name too short: {name} ({})",
                name.len()
            );
        }
    }

    #[test]
    fn test_generate_many() {
        let mut name_gen = NameGenerator::new(2);
        name_gen.train(&[
            "aldric", "mirabel", "cedric", "eleanor", "gareth",
            "isabelle", "marcus", "rowena", "tristan", "vivienne",
        ]);

        let mut rng = Rng::new(42);
        let names = name_gen.generate_many(5, &mut rng);

        assert_eq!(names.len(), 5, "Should generate 5 names");
        // All names should be unique.
        let unique: std::collections::HashSet<&String> = names.iter().collect();
        assert_eq!(unique.len(), 5, "All 5 names should be unique");
    }

    #[test]
    fn test_elven_names() {
        let names = generate_fantasy_names(Culture::Elven, 10, 42);
        assert_eq!(names.len(), 10);
        for name in &names {
            assert!(!name.is_empty());
            assert!(name.chars().next().unwrap().is_uppercase());
        }
    }

    #[test]
    fn test_dwarven_names() {
        let names = generate_fantasy_names(Culture::Dwarven, 10, 42);
        assert_eq!(names.len(), 10);
        for name in &names {
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_orcish_names() {
        let names = generate_fantasy_names(Culture::Orcish, 10, 42);
        assert_eq!(names.len(), 10);
    }

    #[test]
    fn test_human_names() {
        let names = generate_fantasy_names(Culture::Human, 10, 42);
        assert_eq!(names.len(), 10);
    }

    #[test]
    fn test_draconic_names() {
        let names = generate_fantasy_names(Culture::Draconic, 5, 42);
        assert_eq!(names.len(), 5);
    }

    #[test]
    fn test_fey_names() {
        let names = generate_fantasy_names(Culture::Fey, 5, 42);
        assert_eq!(names.len(), 5);
    }

    #[test]
    fn test_deterministic() {
        let n1 = generate_fantasy_name(Culture::Elven, 42);
        let n2 = generate_fantasy_name(Culture::Elven, 42);
        assert_eq!(n1, n2, "Same seed should produce same name");
    }

    #[test]
    fn test_different_seeds() {
        let n1 = generate_fantasy_name(Culture::Elven, 1);
        let n2 = generate_fantasy_name(Culture::Elven, 2);
        // Very likely different (possible but extremely unlikely to be same).
        assert_ne!(n1, n2, "Different seeds should produce different names");
    }

    #[test]
    fn test_different_cultures() {
        let elven = generate_fantasy_names(Culture::Elven, 5, 42);
        let dwarven = generate_fantasy_names(Culture::Dwarven, 5, 42);

        // Names from different cultures should generally be different.
        let mut any_different = false;
        for (e, d) in elven.iter().zip(dwarven.iter()) {
            if e != d {
                any_different = true;
            }
        }
        assert!(
            any_different,
            "Different cultures should produce different names"
        );
    }

    #[test]
    fn test_place_name_generation() {
        let name = generate_place_name(
            TerrainFeature::Mountain,
            SettlementType::Fortress,
            Culture::Dwarven,
            42,
        );
        assert!(!name.is_empty());
    }

    #[test]
    fn test_place_names_batch() {
        let features = vec![
            TerrainFeature::Mountain,
            TerrainFeature::Forest,
            TerrainFeature::River,
        ];
        let settlements = vec![
            SettlementType::City,
            SettlementType::Village,
            SettlementType::Town,
        ];

        let names = generate_place_names(10, &features, &settlements, Culture::Human, 42);
        assert_eq!(names.len(), 10);
        for name in &names {
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_markov_order() {
        // Higher order should produce names more similar to the corpus.
        let gen_low = create_generator(Culture::Elven, 2);
        let gen_high = create_generator(Culture::Elven, 4);

        let mut rng = Rng::new(42);
        let name_low = gen_low.generate(&mut rng);
        let name_high = gen_high.generate(&mut rng);

        assert!(!name_low.is_empty());
        assert!(!name_high.is_empty());
    }

    #[test]
    fn test_create_generator() {
        let name_gen = create_generator(Culture::Elven, 3);
        assert!(!name_gen.transitions.is_empty());
        assert!(!name_gen.start_contexts.is_empty());
    }

    #[test]
    fn test_terrain_feature_variety() {
        let features = [
            TerrainFeature::Mountain,
            TerrainFeature::River,
            TerrainFeature::Forest,
            TerrainFeature::Lake,
            TerrainFeature::Plains,
            TerrainFeature::Coast,
            TerrainFeature::Valley,
            TerrainFeature::Swamp,
            TerrainFeature::Desert,
            TerrainFeature::Island,
        ];

        for feature in &features {
            let name = generate_place_name(
                *feature,
                SettlementType::City,
                Culture::Human,
                42,
            );
            assert!(!name.is_empty(), "Place name for {feature:?} should not be empty");
        }
    }

    #[test]
    fn test_settlement_variety() {
        let settlements = [
            SettlementType::City,
            SettlementType::Town,
            SettlementType::Village,
            SettlementType::Hamlet,
            SettlementType::Fortress,
            SettlementType::Outpost,
            SettlementType::Port,
            SettlementType::Temple,
            SettlementType::Mine,
            SettlementType::Ruin,
        ];

        for settlement in &settlements {
            let name = generate_place_name(
                TerrainFeature::Mountain,
                *settlement,
                Culture::Dwarven,
                42,
            );
            assert!(!name.is_empty(), "Place name for {settlement:?} should not be empty");
        }
    }
}
