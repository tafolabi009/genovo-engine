// engine/gameplay/src/achievement_v2.rs
//
// Extended achievement system for the Genovo gameplay framework.
//
// Provides compound conditions (AND/OR/NOT), progress tracking, secret
// achievements, categories, rewards (items/titles/cosmetics), notification
// queue, and statistics integration.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

pub type AchievementId = u32;
pub type CategoryId = u32;
pub type StatId = u32;
pub type RewardId = u32;

pub const MAX_NOTIFICATION_QUEUE: usize = 16;
pub const NOTIFICATION_DISPLAY_TIME: f32 = 5.0;

// ---------------------------------------------------------------------------
// Achievement condition
// ---------------------------------------------------------------------------

/// Compound condition tree for achievements.
#[derive(Debug, Clone)]
pub enum AchievementCondition {
    StatReached { stat_id: StatId, threshold: f64 },
    StatAccumulated { stat_id: StatId, total: f64 },
    EventOccurred { event_name: String, count: u32 },
    ItemCollected { item_id: String, count: u32 },
    LevelCompleted { level_id: String },
    TimePlayedHours(f64),
    And(Vec<AchievementCondition>),
    Or(Vec<AchievementCondition>),
    Not(Box<AchievementCondition>),
    Custom(String),
}

impl AchievementCondition {
    pub fn stat(stat_id: StatId, threshold: f64) -> Self { Self::StatReached { stat_id, threshold } }
    pub fn event(name: &str, count: u32) -> Self { Self::EventOccurred { event_name: name.to_string(), count } }
    pub fn and(conds: Vec<Self>) -> Self { Self::And(conds) }
    pub fn or(conds: Vec<Self>) -> Self { Self::Or(conds) }
    pub fn not(cond: Self) -> Self { Self::Not(Box::new(cond)) }
}

// ---------------------------------------------------------------------------
// Reward
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum RewardType {
    Item { item_id: String, count: u32 },
    Title(String),
    Cosmetic { cosmetic_id: String, cosmetic_type: String },
    Currency { currency: String, amount: u64 },
    Experience(u64),
    Unlock(String),
    Badge { badge_id: String, tier: u32 },
}

#[derive(Debug, Clone)]
pub struct AchievementReward {
    pub id: RewardId,
    pub reward_type: RewardType,
    pub description: String,
    pub claimed: bool,
}

impl AchievementReward {
    pub fn item(id: RewardId, item_id: &str, count: u32) -> Self {
        Self { id, reward_type: RewardType::Item { item_id: item_id.to_string(), count }, description: format!("{} x{}", item_id, count), claimed: false }
    }
    pub fn title(id: RewardId, title: &str) -> Self {
        Self { id, reward_type: RewardType::Title(title.to_string()), description: format!("Title: {}", title), claimed: false }
    }
    pub fn cosmetic(id: RewardId, cosmetic_id: &str, ctype: &str) -> Self {
        Self { id, reward_type: RewardType::Cosmetic { cosmetic_id: cosmetic_id.to_string(), cosmetic_type: ctype.to_string() }, description: format!("{}: {}", ctype, cosmetic_id), claimed: false }
    }
}

// ---------------------------------------------------------------------------
// Category and rarity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AchievementCategory {
    pub id: CategoryId,
    pub name: String,
    pub description: String,
    pub icon: Option<String>,
    pub sort_order: u32,
    pub achievements: Vec<AchievementId>,
}

impl AchievementCategory {
    pub fn new(id: CategoryId, name: &str, desc: &str) -> Self {
        Self { id, name: name.to_string(), description: desc.to_string(), icon: None, sort_order: 0, achievements: Vec::new() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AchievementRarity { Common, Uncommon, Rare, Epic, Legendary }

impl AchievementRarity {
    pub fn points(&self) -> u32 {
        match self { Self::Common => 5, Self::Uncommon => 10, Self::Rare => 25, Self::Epic => 50, Self::Legendary => 100 }
    }
}

impl fmt::Display for AchievementRarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Common => write!(f, "Common"), Self::Uncommon => write!(f, "Uncommon"), Self::Rare => write!(f, "Rare"), Self::Epic => write!(f, "Epic"), Self::Legendary => write!(f, "Legendary") }
    }
}

// ---------------------------------------------------------------------------
// Achievement definition
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AchievementDefinition {
    pub id: AchievementId,
    pub name: String,
    pub description: String,
    pub secret_description: Option<String>,
    pub icon: Option<String>,
    pub category: CategoryId,
    pub rarity: AchievementRarity,
    pub condition: AchievementCondition,
    pub rewards: Vec<AchievementReward>,
    pub secret: bool,
    pub hidden_until_progress: f32,
    pub points: u32,
    pub prerequisite: Option<AchievementId>,
    pub sort_order: u32,
}

impl AchievementDefinition {
    pub fn new(id: AchievementId, name: &str, desc: &str, condition: AchievementCondition) -> Self {
        Self {
            id, name: name.to_string(), description: desc.to_string(),
            secret_description: None, icon: None, category: 0,
            rarity: AchievementRarity::Common, condition, rewards: Vec::new(),
            secret: false, hidden_until_progress: 0.0, points: 5,
            prerequisite: None, sort_order: 0,
        }
    }
    pub fn with_rarity(mut self, r: AchievementRarity) -> Self { self.rarity = r; self.points = r.points(); self }
    pub fn secret(mut self) -> Self { self.secret = true; self }
    pub fn with_category(mut self, c: CategoryId) -> Self { self.category = c; self }
    pub fn with_reward(mut self, r: AchievementReward) -> Self { self.rewards.push(r); self }
    pub fn with_prerequisite(mut self, p: AchievementId) -> Self { self.prerequisite = Some(p); self }
}

// ---------------------------------------------------------------------------
// Progress and notification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AchievementProgress {
    pub achievement_id: AchievementId,
    pub unlocked: bool,
    pub unlock_time: f64,
    pub progress: f64,
    pub max_progress: f64,
    pub rewards_claimed: HashSet<RewardId>,
    pub notified: bool,
}

impl AchievementProgress {
    pub fn new(id: AchievementId, max: f64) -> Self {
        Self { achievement_id: id, unlocked: false, unlock_time: 0.0, progress: 0.0, max_progress: max, rewards_claimed: HashSet::new(), notified: false }
    }
    pub fn completion_ratio(&self) -> f64 {
        if self.max_progress <= 0.0 { return if self.unlocked { 1.0 } else { 0.0 }; }
        (self.progress / self.max_progress).min(1.0)
    }
}

#[derive(Debug, Clone)]
pub struct AchievementNotification {
    pub achievement_id: AchievementId,
    pub name: String,
    pub description: String,
    pub rarity: AchievementRarity,
    pub icon: Option<String>,
    pub display_time_remaining: f32,
    pub rewards: Vec<String>,
}

// ---------------------------------------------------------------------------
// Statistics store
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct StatisticsStore {
    pub stats: HashMap<StatId, f64>,
    pub event_counts: HashMap<String, u32>,
    pub items_collected: HashMap<String, u32>,
    pub levels_completed: HashSet<String>,
    pub total_play_time_seconds: f64,
}

impl StatisticsStore {
    pub fn new() -> Self { Self::default() }
    pub fn set_stat(&mut self, id: StatId, v: f64) { self.stats.insert(id, v); }
    pub fn get_stat(&self, id: StatId) -> f64 { *self.stats.get(&id).unwrap_or(&0.0) }
    pub fn increment_stat(&mut self, id: StatId, a: f64) { *self.stats.entry(id).or_insert(0.0) += a; }
    pub fn record_event(&mut self, e: &str) { *self.event_counts.entry(e.to_string()).or_insert(0) += 1; }
    pub fn event_count(&self, e: &str) -> u32 { *self.event_counts.get(e).unwrap_or(&0) }
    pub fn collect_item(&mut self, i: &str, c: u32) { *self.items_collected.entry(i.to_string()).or_insert(0) += c; }
    pub fn item_count(&self, i: &str) -> u32 { *self.items_collected.get(i).unwrap_or(&0) }
    pub fn complete_level(&mut self, l: &str) { self.levels_completed.insert(l.to_string()); }
    pub fn is_level_completed(&self, l: &str) -> bool { self.levels_completed.contains(l) }
}

// ---------------------------------------------------------------------------
// Achievement system
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct AchievementSystemStats {
    pub total_achievements: u32,
    pub unlocked_achievements: u32,
    pub total_points: u32,
    pub earned_points: u32,
    pub secret_achievements: u32,
    pub secret_unlocked: u32,
    pub rewards_pending: u32,
}

pub struct AchievementSystemV2 {
    definitions: HashMap<AchievementId, AchievementDefinition>,
    progress: HashMap<AchievementId, AchievementProgress>,
    categories: HashMap<CategoryId, AchievementCategory>,
    notifications: VecDeque<AchievementNotification>,
    statistics: StatisticsStore,
    stats: AchievementSystemStats,
    game_time: f64,
    custom_conditions: HashMap<String, bool>,
}

impl AchievementSystemV2 {
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(), progress: HashMap::new(),
            categories: HashMap::new(), notifications: VecDeque::new(),
            statistics: StatisticsStore::new(), stats: AchievementSystemStats::default(),
            game_time: 0.0, custom_conditions: HashMap::new(),
        }
    }

    pub fn register_category(&mut self, cat: AchievementCategory) { self.categories.insert(cat.id, cat); }

    pub fn register_achievement(&mut self, def: AchievementDefinition) {
        let id = def.id;
        let max = self.estimate_max(&def.condition);
        if let Some(cat) = self.categories.get_mut(&def.category) { cat.achievements.push(id); }
        self.definitions.insert(id, def);
        self.progress.entry(id).or_insert_with(|| AchievementProgress::new(id, max));
    }

    fn estimate_max(&self, c: &AchievementCondition) -> f64 {
        match c {
            AchievementCondition::StatReached { threshold, .. } => *threshold,
            AchievementCondition::StatAccumulated { total, .. } => *total,
            AchievementCondition::EventOccurred { count, .. } => *count as f64,
            AchievementCondition::ItemCollected { count, .. } => *count as f64,
            AchievementCondition::TimePlayedHours(h) => *h,
            _ => 1.0,
        }
    }

    pub fn update(&mut self, dt: f64) {
        self.game_time += dt;
        self.statistics.total_play_time_seconds += dt;
        let ids: Vec<AchievementId> = self.definitions.keys().cloned().collect();
        for id in ids {
            if self.progress.get(&id).map_or(false, |p| p.unlocked) { continue; }
            let prereq_met = self.definitions.get(&id).and_then(|d| d.prerequisite).map_or(true, |pre| self.progress.get(&pre).map_or(false, |p| p.unlocked));
            if !prereq_met { continue; }
            let cond = self.definitions.get(&id).map(|d| d.condition.clone());
            if let Some(cond) = cond {
                let (met, prog) = self.evaluate(&cond);
                if let Some(p) = self.progress.get_mut(&id) {
                    p.progress = prog;
                    if met && !p.unlocked { p.unlocked = true; p.unlock_time = self.game_time; self.create_notification(id); }
                }
            }
        }
        for n in &mut self.notifications { n.display_time_remaining -= dt as f32; }
        self.notifications.retain(|n| n.display_time_remaining > 0.0);
        self.update_stats();
    }

    fn evaluate(&self, c: &AchievementCondition) -> (bool, f64) {
        match c {
            AchievementCondition::StatReached { stat_id, threshold } => { let v = self.statistics.get_stat(*stat_id); (v >= *threshold, v) }
            AchievementCondition::StatAccumulated { stat_id, total } => { let v = self.statistics.get_stat(*stat_id); (v >= *total, v) }
            AchievementCondition::EventOccurred { event_name, count } => { let c = self.statistics.event_count(event_name) as f64; (c >= *count as f64, c) }
            AchievementCondition::ItemCollected { item_id, count } => { let c = self.statistics.item_count(item_id) as f64; (c >= *count as f64, c) }
            AchievementCondition::LevelCompleted { level_id } => { let d = self.statistics.is_level_completed(level_id); (d, if d { 1.0 } else { 0.0 }) }
            AchievementCondition::TimePlayedHours(h) => { let hrs = self.statistics.total_play_time_seconds / 3600.0; (hrs >= *h, hrs) }
            AchievementCondition::And(cs) => { let rs: Vec<_> = cs.iter().map(|c| self.evaluate(c)).collect(); (rs.iter().all(|(m,_)| *m), if rs.is_empty() { 0.0 } else { rs.iter().map(|(_,p)| p).sum::<f64>() / rs.len() as f64 }) }
            AchievementCondition::Or(cs) => { let rs: Vec<_> = cs.iter().map(|c| self.evaluate(c)).collect(); (rs.iter().any(|(m,_)| *m), rs.iter().map(|(_,p)| *p).fold(0.0f64, f64::max)) }
            AchievementCondition::Not(c) => { let (m, _) = self.evaluate(c); (!m, if m { 0.0 } else { 1.0 }) }
            AchievementCondition::Custom(k) => { let m = *self.custom_conditions.get(k).unwrap_or(&false); (m, if m { 1.0 } else { 0.0 }) }
        }
    }

    fn create_notification(&mut self, id: AchievementId) {
        if let Some(def) = self.definitions.get(&id) {
            let rewards: Vec<String> = def.rewards.iter().map(|r| r.description.clone()).collect();
            if self.notifications.len() >= MAX_NOTIFICATION_QUEUE { self.notifications.pop_front(); }
            self.notifications.push_back(AchievementNotification {
                achievement_id: id, name: def.name.clone(), description: def.description.clone(),
                rarity: def.rarity, icon: def.icon.clone(), display_time_remaining: NOTIFICATION_DISPLAY_TIME, rewards,
            });
        }
    }

    fn update_stats(&mut self) {
        self.stats = AchievementSystemStats::default();
        self.stats.total_achievements = self.definitions.len() as u32;
        for (id, def) in &self.definitions {
            self.stats.total_points += def.points;
            if def.secret { self.stats.secret_achievements += 1; }
            if let Some(p) = self.progress.get(id) {
                if p.unlocked { self.stats.unlocked_achievements += 1; self.stats.earned_points += def.points; if def.secret { self.stats.secret_unlocked += 1; } }
            }
        }
    }

    pub fn statistics_mut(&mut self) -> &mut StatisticsStore { &mut self.statistics }
    pub fn statistics(&self) -> &StatisticsStore { &self.statistics }
    pub fn set_custom_condition(&mut self, k: &str, v: bool) { self.custom_conditions.insert(k.to_string(), v); }
    pub fn is_unlocked(&self, id: AchievementId) -> bool { self.progress.get(&id).map_or(false, |p| p.unlocked) }
    pub fn get_progress(&self, id: AchievementId) -> Option<&AchievementProgress> { self.progress.get(&id) }
    pub fn notifications(&self) -> &VecDeque<AchievementNotification> { &self.notifications }
    pub fn stats(&self) -> &AchievementSystemStats { &self.stats }
    pub fn completion_percentage(&self) -> f32 { if self.stats.total_achievements == 0 { 0.0 } else { self.stats.unlocked_achievements as f32 / self.stats.total_achievements as f32 * 100.0 } }
    pub fn claim_reward(&mut self, aid: AchievementId, rid: RewardId) -> bool {
        if let Some(p) = self.progress.get_mut(&aid) { if p.unlocked && !p.rewards_claimed.contains(&rid) { p.rewards_claimed.insert(rid); return true; } } false
    }
    pub fn get_category_achievements(&self, cat: CategoryId) -> Vec<&AchievementDefinition> {
        self.categories.get(&cat).map(|c| c.achievements.iter().filter_map(|id| self.definitions.get(id)).collect()).unwrap_or_default()
    }
    pub fn reset_all(&mut self) {
        for p in self.progress.values_mut() { p.unlocked = false; p.progress = 0.0; p.rewards_claimed.clear(); }
        self.statistics = StatisticsStore::new(); self.notifications.clear(); self.custom_conditions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unlock() {
        let mut s = AchievementSystemV2::new();
        s.register_achievement(AchievementDefinition::new(1, "First Kill", "Kill an enemy", AchievementCondition::event("kill", 1)));
        s.statistics_mut().record_event("kill");
        s.update(0.016);
        assert!(s.is_unlocked(1));
    }

    #[test]
    fn test_compound() {
        let mut s = AchievementSystemV2::new();
        s.register_achievement(AchievementDefinition::new(1, "Combo", "Both", AchievementCondition::and(vec![
            AchievementCondition::event("kill", 1), AchievementCondition::event("collect", 1),
        ])));
        s.statistics_mut().record_event("kill");
        s.update(0.016);
        assert!(!s.is_unlocked(1));
        s.statistics_mut().record_event("collect");
        s.update(0.016);
        assert!(s.is_unlocked(1));
    }
}
