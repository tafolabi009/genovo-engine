// engine/gameplay/src/inventory_v2.rs
//
// Enhanced grid-based inventory system for the Genovo engine.
//
// Features:
// - Grid-based inventory (items occupy WxH cells, like Diablo/Escape from Tarkov)
// - Auto-sort and compact (optimize space usage)
// - Item categories and filtering
// - Item comparison tooltips (stat diffs between equipped and candidate)
// - Equipment set bonuses (wearing matching gear grants bonuses)
// - Item durability and repair system

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const MAX_GRID_WIDTH: u32 = 20;
pub const MAX_GRID_HEIGHT: u32 = 20;
pub const MAX_ITEM_WIDTH: u32 = 4;
pub const MAX_ITEM_HEIGHT: u32 = 4;
pub const MAX_EQUIPMENT_SETS: usize = 16;
pub const MAX_SET_PIECES: usize = 8;
pub const DURABILITY_BROKEN_THRESHOLD: f32 = 0.0;

// ---------------------------------------------------------------------------
// Item definition
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ItemTypeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ItemInstanceId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemCategory {
    Weapon, Armor, Consumable, Material, QuestItem, Ammo,
    Accessory, Junk, Key, Crafting, Currency, Special,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ItemRarityV2 {
    Common, Uncommon, Rare, Epic, Legendary, Mythic,
}

impl ItemRarityV2 {
    pub fn color(&self) -> (f32, f32, f32) {
        match self {
            Self::Common => (0.7, 0.7, 0.7),
            Self::Uncommon => (0.2, 0.8, 0.2),
            Self::Rare => (0.2, 0.4, 1.0),
            Self::Epic => (0.6, 0.2, 0.8),
            Self::Legendary => (1.0, 0.6, 0.1),
            Self::Mythic => (1.0, 0.2, 0.2),
        }
    }

    pub fn sell_multiplier(&self) -> f32 {
        match self {
            Self::Common => 1.0, Self::Uncommon => 2.0, Self::Rare => 5.0,
            Self::Epic => 15.0, Self::Legendary => 50.0, Self::Mythic => 200.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EquipSlotV2 {
    Head, Chest, Legs, Feet, Hands, MainHand, OffHand,
    Ring1, Ring2, Necklace, Belt, Back, Shoulder,
}

#[derive(Debug, Clone)]
pub struct ItemStatV2 {
    pub name: String,
    pub value: f32,
    pub is_percentage: bool,
}

impl ItemStatV2 {
    pub fn flat(name: &str, value: f32) -> Self {
        Self { name: name.to_string(), value, is_percentage: false }
    }
    pub fn percent(name: &str, value: f32) -> Self {
        Self { name: name.to_string(), value, is_percentage: true }
    }
}

#[derive(Debug, Clone)]
pub struct ItemTypeDefV2 {
    pub id: ItemTypeId,
    pub name: String,
    pub description: String,
    pub category: ItemCategory,
    pub rarity: ItemRarityV2,
    pub grid_width: u32,
    pub grid_height: u32,
    pub max_stack: u32,
    pub weight: f32,
    pub base_value: u32,
    pub equip_slot: Option<EquipSlotV2>,
    pub stats: Vec<ItemStatV2>,
    pub max_durability: f32,
    pub set_id: Option<EquipmentSetId>,
    pub icon_index: u32,
    pub can_be_sold: bool,
    pub can_be_dropped: bool,
    pub can_be_destroyed: bool,
    pub level_requirement: u32,
}

impl ItemTypeDefV2 {
    pub fn new(id: ItemTypeId, name: &str, category: ItemCategory) -> Self {
        Self {
            id, name: name.to_string(), description: String::new(),
            category, rarity: ItemRarityV2::Common,
            grid_width: 1, grid_height: 1, max_stack: 1,
            weight: 1.0, base_value: 10,
            equip_slot: None, stats: Vec::new(),
            max_durability: 100.0, set_id: None,
            icon_index: 0, can_be_sold: true,
            can_be_dropped: true, can_be_destroyed: true,
            level_requirement: 1,
        }
    }

    pub fn with_size(mut self, w: u32, h: u32) -> Self {
        self.grid_width = w;
        self.grid_height = h;
        self
    }

    pub fn with_stack(mut self, max: u32) -> Self {
        self.max_stack = max;
        self
    }

    pub fn with_rarity(mut self, r: ItemRarityV2) -> Self {
        self.rarity = r;
        self
    }

    pub fn with_slot(mut self, slot: EquipSlotV2) -> Self {
        self.equip_slot = Some(slot);
        self
    }

    pub fn with_stat(mut self, stat: ItemStatV2) -> Self {
        self.stats.push(stat);
        self
    }

    pub fn with_set(mut self, set: EquipmentSetId) -> Self {
        self.set_id = Some(set);
        self
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }
}

// ---------------------------------------------------------------------------
// Item instance
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ItemInstanceV2 {
    pub instance_id: ItemInstanceId,
    pub type_id: ItemTypeId,
    pub stack_count: u32,
    pub current_durability: f32,
    pub custom_name: Option<String>,
    pub enchantments: Vec<ItemStatV2>,
    pub is_favourite: bool,
    pub is_locked: bool,
    pub is_new: bool,
    pub creation_time: f64,
}

impl ItemInstanceV2 {
    pub fn new(instance_id: ItemInstanceId, type_id: ItemTypeId) -> Self {
        Self {
            instance_id, type_id, stack_count: 1,
            current_durability: 100.0, custom_name: None,
            enchantments: Vec::new(), is_favourite: false,
            is_locked: false, is_new: true, creation_time: 0.0,
        }
    }

    pub fn with_stack(mut self, count: u32) -> Self {
        self.stack_count = count;
        self
    }

    pub fn with_durability(mut self, d: f32) -> Self {
        self.current_durability = d;
        self
    }

    pub fn durability_fraction(&self, max: f32) -> f32 {
        if max <= 0.0 { 1.0 } else { (self.current_durability / max).clamp(0.0, 1.0) }
    }

    pub fn is_broken(&self) -> bool {
        self.current_durability <= DURABILITY_BROKEN_THRESHOLD
    }

    pub fn degrade(&mut self, amount: f32) {
        self.current_durability = (self.current_durability - amount).max(0.0);
    }

    pub fn repair(&mut self, amount: f32, max: f32) {
        self.current_durability = (self.current_durability + amount).min(max);
    }

    pub fn repair_full(&mut self, max: f32) {
        self.current_durability = max;
    }

    pub fn mark_seen(&mut self) {
        self.is_new = false;
    }

    pub fn total_stats(&self, base_stats: &[ItemStatV2]) -> Vec<ItemStatV2> {
        let mut result = base_stats.to_vec();
        for ench in &self.enchantments {
            if let Some(existing) = result.iter_mut().find(|s| s.name == ench.name) {
                existing.value += ench.value;
            } else {
                result.push(ench.clone());
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Grid inventory
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridCell {
    Empty,
    Occupied(ItemInstanceId),
}

#[derive(Debug, Clone)]
pub struct GridItemPlacement {
    pub instance: ItemInstanceV2,
    pub grid_x: u32,
    pub grid_y: u32,
    pub item_width: u32,
    pub item_height: u32,
    pub rotated: bool,
}

#[derive(Debug)]
pub struct GridInventory {
    pub width: u32,
    pub height: u32,
    pub cells: Vec<GridCell>,
    pub items: HashMap<ItemInstanceId, GridItemPlacement>,
    pub weight_limit: f32,
    pub current_weight: f32,
    next_instance_id: u64,
}

impl GridInventory {
    pub fn new(width: u32, height: u32) -> Self {
        let w = width.min(MAX_GRID_WIDTH);
        let h = height.min(MAX_GRID_HEIGHT);
        Self {
            width: w, height: h,
            cells: vec![GridCell::Empty; (w * h) as usize],
            items: HashMap::new(),
            weight_limit: 100.0,
            current_weight: 0.0,
            next_instance_id: 1,
        }
    }

    fn cell_index(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    fn is_area_free(&self, x: u32, y: u32, w: u32, h: u32, exclude: Option<ItemInstanceId>) -> bool {
        if x + w > self.width || y + h > self.height {
            return false;
        }
        for dy in 0..h {
            for dx in 0..w {
                let idx = self.cell_index(x + dx, y + dy);
                match self.cells[idx] {
                    GridCell::Empty => {}
                    GridCell::Occupied(id) => {
                        if let Some(excl) = exclude {
                            if id == excl {
                                continue;
                            }
                        }
                        return false;
                    }
                }
            }
        }
        true
    }

    fn mark_area(&mut self, x: u32, y: u32, w: u32, h: u32, id: ItemInstanceId) {
        for dy in 0..h {
            for dx in 0..w {
                let idx = self.cell_index(x + dx, y + dy);
                self.cells[idx] = GridCell::Occupied(id);
            }
        }
    }

    fn clear_area(&mut self, x: u32, y: u32, w: u32, h: u32) {
        for dy in 0..h {
            for dx in 0..w {
                let idx = self.cell_index(x + dx, y + dy);
                self.cells[idx] = GridCell::Empty;
            }
        }
    }

    pub fn place_at(
        &mut self,
        x: u32,
        y: u32,
        instance: ItemInstanceV2,
        item_def: &ItemTypeDefV2,
    ) -> Result<ItemInstanceId, InventoryErrorV2> {
        let w = item_def.grid_width;
        let h = item_def.grid_height;
        if !self.is_area_free(x, y, w, h, None) {
            return Err(InventoryErrorV2::NoSpace);
        }
        let id = instance.instance_id;
        self.mark_area(x, y, w, h, id);
        self.current_weight += item_def.weight * instance.stack_count as f32;
        self.items.insert(id, GridItemPlacement {
            instance, grid_x: x, grid_y: y,
            item_width: w, item_height: h, rotated: false,
        });
        Ok(id)
    }

    pub fn auto_place(
        &mut self,
        instance: ItemInstanceV2,
        item_def: &ItemTypeDefV2,
    ) -> Result<ItemInstanceId, InventoryErrorV2> {
        let w = item_def.grid_width;
        let h = item_def.grid_height;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.is_area_free(x, y, w, h, None) {
                    return self.place_at(x, y, instance, item_def);
                }
            }
        }
        if w != h {
            for y in 0..self.height {
                for x in 0..self.width {
                    if self.is_area_free(x, y, h, w, None) {
                        let id = instance.instance_id;
                        self.mark_area(x, y, h, w, id);
                        self.current_weight += item_def.weight * instance.stack_count as f32;
                        self.items.insert(id, GridItemPlacement {
                            instance, grid_x: x, grid_y: y,
                            item_width: h, item_height: w, rotated: true,
                        });
                        return Ok(id);
                    }
                }
            }
        }
        Err(InventoryErrorV2::NoSpace)
    }

    pub fn remove(&mut self, id: ItemInstanceId) -> Option<ItemInstanceV2> {
        if let Some(placement) = self.items.remove(&id) {
            self.clear_area(placement.grid_x, placement.grid_y, placement.item_width, placement.item_height);
            Some(placement.instance)
        } else {
            None
        }
    }

    pub fn move_item(&mut self, id: ItemInstanceId, new_x: u32, new_y: u32) -> Result<(), InventoryErrorV2> {
        let placement = self.items.get(&id).ok_or(InventoryErrorV2::ItemNotFound)?;
        let w = placement.item_width;
        let h = placement.item_height;
        let old_x = placement.grid_x;
        let old_y = placement.grid_y;
        if !self.is_area_free(new_x, new_y, w, h, Some(id)) {
            return Err(InventoryErrorV2::NoSpace);
        }
        self.clear_area(old_x, old_y, w, h);
        self.mark_area(new_x, new_y, w, h, id);
        if let Some(p) = self.items.get_mut(&id) {
            p.grid_x = new_x;
            p.grid_y = new_y;
        }
        Ok(())
    }

    pub fn get(&self, id: ItemInstanceId) -> Option<&GridItemPlacement> {
        self.items.get(&id)
    }

    pub fn auto_sort(&mut self) {
        let mut placements: Vec<_> = self.items.values().cloned().collect();
        placements.sort_by(|a, b| {
            let area_a = a.item_width * a.item_height;
            let area_b = b.item_width * b.item_height;
            area_b.cmp(&area_a)
        });
        self.cells.fill(GridCell::Empty);
        self.items.clear();
        for mut p in placements {
            let w = p.item_width;
            let h = p.item_height;
            let id = p.instance.instance_id;
            let mut placed = false;
            for y in 0..self.height {
                if placed { break; }
                for x in 0..self.width {
                    if self.is_area_free(x, y, w, h, None) {
                        self.mark_area(x, y, w, h, id);
                        p.grid_x = x;
                        p.grid_y = y;
                        self.items.insert(id, p.clone());
                        placed = true;
                        break;
                    }
                }
            }
        }
    }

    pub fn empty_cells(&self) -> u32 {
        self.cells.iter().filter(|c| **c == GridCell::Empty).count() as u32
    }

    pub fn total_cells(&self) -> u32 {
        self.width * self.height
    }

    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    pub fn next_id(&mut self) -> ItemInstanceId {
        let id = ItemInstanceId(self.next_instance_id);
        self.next_instance_id += 1;
        id
    }
}

// ---------------------------------------------------------------------------
// Equipment set bonuses
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EquipmentSetId(pub u32);

#[derive(Debug, Clone)]
pub struct SetBonusV2 {
    pub pieces_required: u32,
    pub stats: Vec<ItemStatV2>,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct EquipmentSetV2 {
    pub id: EquipmentSetId,
    pub name: String,
    pub piece_item_ids: Vec<ItemTypeId>,
    pub bonuses: Vec<SetBonusV2>,
}

impl EquipmentSetV2 {
    pub fn new(id: EquipmentSetId, name: &str) -> Self {
        Self { id, name: name.to_string(), piece_item_ids: Vec::new(), bonuses: Vec::new() }
    }

    pub fn add_piece(&mut self, item_id: ItemTypeId) {
        self.piece_item_ids.push(item_id);
    }

    pub fn add_bonus(&mut self, pieces: u32, stats: Vec<ItemStatV2>, desc: &str) {
        self.bonuses.push(SetBonusV2 {
            pieces_required: pieces,
            stats,
            description: desc.to_string(),
        });
    }

    pub fn active_bonuses(&self, equipped_count: u32) -> Vec<&SetBonusV2> {
        self.bonuses.iter().filter(|b| equipped_count >= b.pieces_required).collect()
    }

    pub fn total_pieces(&self) -> usize {
        self.piece_item_ids.len()
    }
}

// ---------------------------------------------------------------------------
// Item comparison
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct StatDiffV2 {
    pub stat_name: String,
    pub current_value: f32,
    pub new_value: f32,
    pub difference: f32,
    pub is_percentage: bool,
    pub is_improvement: bool,
}

pub fn compare_items_v2(current: &ItemTypeDefV2, candidate: &ItemTypeDefV2) -> Vec<StatDiffV2> {
    let mut diffs = Vec::new();
    let mut current_stats: HashMap<&str, (f32, bool)> = HashMap::new();
    for s in &current.stats {
        current_stats.insert(&s.name, (s.value, s.is_percentage));
    }
    for s in &candidate.stats {
        let (current_val, _) = current_stats.remove(s.name.as_str()).unwrap_or((0.0, false));
        let diff = s.value - current_val;
        diffs.push(StatDiffV2 {
            stat_name: s.name.clone(),
            current_value: current_val,
            new_value: s.value,
            difference: diff,
            is_percentage: s.is_percentage,
            is_improvement: diff > 0.0,
        });
    }
    for (name, (val, is_pct)) in current_stats {
        diffs.push(StatDiffV2 {
            stat_name: name.to_string(),
            current_value: val,
            new_value: 0.0,
            difference: -val,
            is_percentage: is_pct,
            is_improvement: false,
        });
    }
    diffs
}

// ---------------------------------------------------------------------------
// Durability repair
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RepairCostV2 {
    pub currency_cost: u32,
    pub material_costs: Vec<(ItemTypeId, u32)>,
    pub durability_restored: f32,
}

pub fn calculate_repair_cost_v2(item: &ItemInstanceV2, item_def: &ItemTypeDefV2) -> RepairCostV2 {
    let missing = item_def.max_durability - item.current_durability;
    let fraction = missing / item_def.max_durability;
    let base_cost = (item_def.base_value as f32 * fraction * 0.3) as u32;
    let rarity_mult = item_def.rarity.sell_multiplier();
    RepairCostV2 {
        currency_cost: (base_cost as f32 * rarity_mult) as u32,
        material_costs: Vec::new(),
        durability_restored: missing,
    }
}

// ---------------------------------------------------------------------------
// Item registry
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ItemRegistryV2 {
    pub items: HashMap<ItemTypeId, ItemTypeDefV2>,
    pub sets: HashMap<EquipmentSetId, EquipmentSetV2>,
}

impl ItemRegistryV2 {
    pub fn new() -> Self {
        Self { items: HashMap::new(), sets: HashMap::new() }
    }

    pub fn register(&mut self, def: ItemTypeDefV2) {
        self.items.insert(def.id, def);
    }

    pub fn register_set(&mut self, set: EquipmentSetV2) {
        self.sets.insert(set.id, set);
    }

    pub fn get(&self, id: ItemTypeId) -> Option<&ItemTypeDefV2> {
        self.items.get(&id)
    }

    pub fn get_set(&self, id: EquipmentSetId) -> Option<&EquipmentSetV2> {
        self.sets.get(&id)
    }

    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    pub fn set_count(&self) -> usize {
        self.sets.len()
    }
}

impl Default for ItemRegistryV2 {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InventoryErrorV2 {
    NoSpace,
    ItemNotFound,
    StackFull,
    WeightLimit,
    InvalidPosition,
    ItemLocked,
    CannotDrop,
}

impl std::fmt::Display for InventoryErrorV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSpace => write!(f, "No space in inventory"),
            Self::ItemNotFound => write!(f, "Item not found"),
            Self::StackFull => write!(f, "Stack is full"),
            Self::WeightLimit => write!(f, "Weight limit exceeded"),
            Self::InvalidPosition => write!(f, "Invalid grid position"),
            Self::ItemLocked => write!(f, "Item is locked"),
            Self::CannotDrop => write!(f, "Item cannot be dropped"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_item_def() -> ItemTypeDefV2 {
        ItemTypeDefV2::new(ItemTypeId(1), "Sword", ItemCategory::Weapon)
            .with_size(1, 3)
            .with_slot(EquipSlotV2::MainHand)
            .with_stat(ItemStatV2::flat("Attack", 25.0))
    }

    #[test]
    fn test_grid_place() {
        let mut inv = GridInventory::new(10, 10);
        let def = test_item_def();
        let inst = ItemInstanceV2::new(inv.next_id(), ItemTypeId(1));
        let result = inv.place_at(0, 0, inst, &def);
        assert!(result.is_ok());
        assert_eq!(inv.item_count(), 1);
    }

    #[test]
    fn test_grid_auto_place() {
        let mut inv = GridInventory::new(5, 5);
        let def = test_item_def();
        for _ in 0..5 {
            let inst = ItemInstanceV2::new(inv.next_id(), ItemTypeId(1));
            let _ = inv.auto_place(inst, &def);
        }
        assert!(inv.item_count() >= 1);
    }

    #[test]
    fn test_grid_remove() {
        let mut inv = GridInventory::new(10, 10);
        let def = test_item_def();
        let inst = ItemInstanceV2::new(inv.next_id(), ItemTypeId(1));
        let id = inv.place_at(0, 0, inst, &def).unwrap();
        assert!(inv.remove(id).is_some());
        assert_eq!(inv.item_count(), 0);
    }

    #[test]
    fn test_durability() {
        let mut inst = ItemInstanceV2::new(ItemInstanceId(1), ItemTypeId(1))
            .with_durability(100.0);
        inst.degrade(30.0);
        assert!((inst.current_durability - 70.0).abs() < 0.01);
        inst.repair(50.0, 100.0);
        assert!((inst.current_durability - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_item_comparison() {
        let a = ItemTypeDefV2::new(ItemTypeId(1), "Old Sword", ItemCategory::Weapon)
            .with_stat(ItemStatV2::flat("Attack", 10.0));
        let b = ItemTypeDefV2::new(ItemTypeId(2), "New Sword", ItemCategory::Weapon)
            .with_stat(ItemStatV2::flat("Attack", 20.0));
        let diffs = compare_items_v2(&a, &b);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].is_improvement);
    }

    #[test]
    fn test_set_bonuses() {
        let mut set = EquipmentSetV2::new(EquipmentSetId(1), "Dragon Set");
        set.add_bonus(2, vec![ItemStatV2::flat("HP", 50.0)], "2-piece: +50 HP");
        set.add_bonus(4, vec![ItemStatV2::flat("Attack", 20.0)], "4-piece: +20 ATK");
        assert_eq!(set.active_bonuses(1).len(), 0);
        assert_eq!(set.active_bonuses(2).len(), 1);
        assert_eq!(set.active_bonuses(4).len(), 2);
    }
}
