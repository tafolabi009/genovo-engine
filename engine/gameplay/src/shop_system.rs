// engine/gameplay/src/shop_system.rs
//
// Shop interface system for the Genovo engine.
//
// Provides a complete shop/store system for buying and selling items:
//
// - **Shop inventory** -- Shops have their own inventory with item stocks.
// - **Buy/sell transactions** -- Complete transaction logic with validation.
// - **Dynamic pricing** -- Prices can fluctuate based on supply, demand, and modifiers.
// - **Limited stock** -- Items can have finite quantities that deplete.
// - **Restock timers** -- Shops restock their inventory on configurable intervals.
// - **Shop categories** -- Items organized into browseable categories.
// - **Currency display** -- Support for multiple currencies.
// - **Purchase confirmation** -- Transaction confirmation with preview.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Unique shop identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShopId(pub u32);

/// Item definition identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ItemId(pub u32);

/// Currency type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CurrencyId(pub u32);

impl fmt::Display for ShopId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shop({})", self.0)
    }
}

impl fmt::Display for ItemId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Item({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Currency
// ---------------------------------------------------------------------------

/// A currency definition.
#[derive(Debug, Clone)]
pub struct CurrencyDef {
    /// Currency ID.
    pub id: CurrencyId,
    /// Display name (e.g., "Gold", "Gems").
    pub name: String,
    /// Symbol/icon identifier.
    pub symbol: String,
    /// Maximum amount a player can hold.
    pub max_amount: u64,
    /// Whether this currency can be earned in-game (vs premium).
    pub earnable: bool,
}

/// A player's wallet holding multiple currencies.
#[derive(Debug, Clone, Default)]
pub struct ShopWallet {
    /// Currency balances.
    pub balances: HashMap<CurrencyId, u64>,
}

impl ShopWallet {
    /// Get the balance of a currency.
    pub fn balance(&self, currency: CurrencyId) -> u64 {
        self.balances.get(&currency).copied().unwrap_or(0)
    }

    /// Check if the wallet can afford a price.
    pub fn can_afford(&self, price: &Price) -> bool {
        for (currency, amount) in &price.costs {
            if self.balance(*currency) < *amount {
                return false;
            }
        }
        true
    }

    /// Deduct a price from the wallet. Returns false if insufficient funds.
    pub fn deduct(&mut self, price: &Price) -> bool {
        if !self.can_afford(price) {
            return false;
        }
        for (currency, amount) in &price.costs {
            let balance = self.balances.entry(*currency).or_insert(0);
            *balance -= amount;
        }
        true
    }

    /// Add currency.
    pub fn add(&mut self, currency: CurrencyId, amount: u64) {
        let balance = self.balances.entry(currency).or_insert(0);
        *balance = balance.saturating_add(amount);
    }
}

// ---------------------------------------------------------------------------
// Price
// ---------------------------------------------------------------------------

/// A price consisting of one or more currency costs.
#[derive(Debug, Clone, Default)]
pub struct Price {
    /// Map of currency -> amount.
    pub costs: HashMap<CurrencyId, u64>,
}

impl Price {
    /// Create a single-currency price.
    pub fn single(currency: CurrencyId, amount: u64) -> Self {
        let mut costs = HashMap::new();
        costs.insert(currency, amount);
        Self { costs }
    }

    /// Create a free price (no cost).
    pub fn free() -> Self {
        Self {
            costs: HashMap::new(),
        }
    }

    /// Check if this price is free.
    pub fn is_free(&self) -> bool {
        self.costs.values().all(|&v| v == 0)
    }

    /// Apply a price modifier (multiplier).
    pub fn with_modifier(&self, modifier: f32) -> Price {
        let mut costs = HashMap::new();
        for (&currency, &amount) in &self.costs {
            costs.insert(currency, (amount as f64 * modifier as f64) as u64);
        }
        Price { costs }
    }

    /// Get the total cost in a single currency (for display).
    pub fn primary_cost(&self) -> u64 {
        self.costs.values().copied().next().unwrap_or(0)
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .costs
            .iter()
            .map(|(id, amount)| format!("{}x Currency({})", amount, id.0))
            .collect();
        write!(f, "{}", parts.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Shop category
// ---------------------------------------------------------------------------

/// Category for organizing shop items.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShopCategory {
    /// Category name.
    pub name: String,
    /// Display order.
    pub order: u32,
    /// Icon identifier.
    pub icon: String,
}

impl ShopCategory {
    pub fn new(name: &str, order: u32) -> Self {
        Self {
            name: name.to_string(),
            order,
            icon: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Shop item
// ---------------------------------------------------------------------------

/// An item listing in a shop.
#[derive(Debug, Clone)]
pub struct ShopItem {
    /// Item definition ID.
    pub item_id: ItemId,
    /// Display name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Base buy price.
    pub base_buy_price: Price,
    /// Base sell price (what the shop pays for this item).
    pub base_sell_price: Price,
    /// Category.
    pub category: String,
    /// Current stock (-1 = unlimited).
    pub stock: i32,
    /// Maximum stock.
    pub max_stock: i32,
    /// Restock amount per cycle.
    pub restock_amount: i32,
    /// Whether this item can be sold back.
    pub sellable: bool,
    /// Whether this item is currently available.
    pub available: bool,
    /// Level requirement to purchase.
    pub level_requirement: u32,
    /// Icon/thumbnail path.
    pub icon: String,
    /// Item rarity (0=common, 1=uncommon, 2=rare, 3=epic, 4=legendary).
    pub rarity: u8,
    /// Discount percentage (0-100).
    pub discount: u32,
    /// Whether this is a featured/highlighted item.
    pub featured: bool,
}

impl ShopItem {
    /// Create a new shop item.
    pub fn new(item_id: ItemId, name: &str, buy_price: Price) -> Self {
        let sell_price = buy_price.with_modifier(0.5); // Default: sell for half.
        Self {
            item_id,
            name: name.to_string(),
            description: String::new(),
            base_buy_price: buy_price,
            base_sell_price: sell_price,
            category: "General".to_string(),
            stock: -1,
            max_stock: -1,
            restock_amount: 0,
            sellable: true,
            available: true,
            level_requirement: 0,
            icon: String::new(),
            rarity: 0,
            discount: 0,
            featured: false,
        }
    }

    /// Get the effective buy price (with discount).
    pub fn effective_buy_price(&self) -> Price {
        if self.discount > 0 {
            let modifier = 1.0 - (self.discount as f32 / 100.0);
            self.base_buy_price.with_modifier(modifier)
        } else {
            self.base_buy_price.clone()
        }
    }

    /// Whether the item is in stock.
    pub fn in_stock(&self) -> bool {
        self.stock < 0 || self.stock > 0
    }

    /// Decrease stock by one.
    pub fn decrease_stock(&mut self) {
        if self.stock > 0 {
            self.stock -= 1;
        }
    }

    /// Restock the item.
    pub fn restock(&mut self) {
        if self.max_stock >= 0 {
            self.stock = (self.stock + self.restock_amount).min(self.max_stock);
        }
    }
}

// ---------------------------------------------------------------------------
// Transaction
// ---------------------------------------------------------------------------

/// A pending or completed transaction.
#[derive(Debug, Clone)]
pub struct Transaction {
    /// Shop ID.
    pub shop_id: ShopId,
    /// Item ID.
    pub item_id: ItemId,
    /// Quantity.
    pub quantity: u32,
    /// Total price.
    pub total_price: Price,
    /// Whether this is a buy (true) or sell (false).
    pub is_buy: bool,
    /// Transaction result.
    pub result: TransactionResult,
}

/// Result of a transaction attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionResult {
    /// Transaction succeeded.
    Success,
    /// Insufficient funds.
    InsufficientFunds,
    /// Item out of stock.
    OutOfStock,
    /// Item not available.
    NotAvailable,
    /// Level requirement not met.
    LevelRequirementNotMet,
    /// Inventory full (buyer can't hold more).
    InventoryFull,
    /// Transaction cancelled.
    Cancelled,
    /// Item not sellable.
    NotSellable,
}

impl fmt::Display for TransactionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "Success"),
            Self::InsufficientFunds => write!(f, "Insufficient funds"),
            Self::OutOfStock => write!(f, "Out of stock"),
            Self::NotAvailable => write!(f, "Not available"),
            Self::LevelRequirementNotMet => write!(f, "Level requirement not met"),
            Self::InventoryFull => write!(f, "Inventory full"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::NotSellable => write!(f, "Item cannot be sold"),
        }
    }
}

// ---------------------------------------------------------------------------
// Purchase confirmation
// ---------------------------------------------------------------------------

/// Data for a purchase confirmation dialog.
#[derive(Debug, Clone)]
pub struct PurchaseConfirmation {
    /// Item being purchased.
    pub item_name: String,
    /// Item icon.
    pub item_icon: String,
    /// Quantity.
    pub quantity: u32,
    /// Total price.
    pub total_price: Price,
    /// Current wallet balance.
    pub wallet_balance: HashMap<CurrencyId, u64>,
    /// Whether the purchase is affordable.
    pub affordable: bool,
    /// Remaining balance after purchase.
    pub remaining_balance: HashMap<CurrencyId, u64>,
}

// ---------------------------------------------------------------------------
// Shop
// ---------------------------------------------------------------------------

/// A shop instance with inventory.
#[derive(Debug, Clone)]
pub struct Shop {
    /// Shop ID.
    pub id: ShopId,
    /// Shop name.
    pub name: String,
    /// Shop description.
    pub description: String,
    /// Items in the shop.
    pub items: Vec<ShopItem>,
    /// Categories.
    pub categories: Vec<ShopCategory>,
    /// Restock interval (seconds).
    pub restock_interval: f32,
    /// Time since last restock.
    pub restock_timer: f32,
    /// Global price modifier (affects all items).
    pub price_modifier: f32,
    /// Sell price modifier (affects sell prices).
    pub sell_modifier: f32,
    /// Whether the shop is open.
    pub open: bool,
    /// Transaction history.
    pub history: Vec<Transaction>,
    /// Maximum history entries.
    pub max_history: usize,
    /// Player level (for requirement checking).
    player_level: u32,
}

impl Shop {
    /// Create a new shop.
    pub fn new(id: ShopId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            description: String::new(),
            items: Vec::new(),
            categories: vec![ShopCategory::new("All", 0)],
            restock_interval: 300.0,
            restock_timer: 0.0,
            price_modifier: 1.0,
            sell_modifier: 1.0,
            open: true,
            history: Vec::new(),
            max_history: 100,
            player_level: 1,
        }
    }

    /// Add an item to the shop.
    pub fn add_item(&mut self, item: ShopItem) {
        if !self.categories.iter().any(|c| c.name == item.category) {
            self.categories.push(ShopCategory::new(&item.category, self.categories.len() as u32));
        }
        self.items.push(item);
    }

    /// Remove an item from the shop.
    pub fn remove_item(&mut self, item_id: ItemId) {
        self.items.retain(|i| i.item_id != item_id);
    }

    /// Get an item by ID.
    pub fn get_item(&self, item_id: ItemId) -> Option<&ShopItem> {
        self.items.iter().find(|i| i.item_id == item_id)
    }

    /// Get a mutable item by ID.
    pub fn get_item_mut(&mut self, item_id: ItemId) -> Option<&mut ShopItem> {
        self.items.iter_mut().find(|i| i.item_id == item_id)
    }

    /// Get items in a category.
    pub fn items_in_category(&self, category: &str) -> Vec<&ShopItem> {
        if category == "All" {
            self.items.iter().collect()
        } else {
            self.items.iter().filter(|i| i.category == category).collect()
        }
    }

    /// Set the player level for requirement checking.
    pub fn set_player_level(&mut self, level: u32) {
        self.player_level = level;
    }

    /// Attempt to buy an item.
    pub fn buy(
        &mut self,
        item_id: ItemId,
        quantity: u32,
        wallet: &mut ShopWallet,
    ) -> TransactionResult {
        if !self.open {
            return TransactionResult::NotAvailable;
        }

        let (price, stock, available, level_req, sellable) = {
            let item = match self.get_item(item_id) {
                Some(i) => i,
                None => return TransactionResult::NotAvailable,
            };
            if !item.available {
                return TransactionResult::NotAvailable;
            }
            if !item.in_stock() || (item.stock >= 0 && (item.stock as u32) < quantity) {
                return TransactionResult::OutOfStock;
            }
            if item.level_requirement > self.player_level {
                return TransactionResult::LevelRequirementNotMet;
            }
            let unit_price = item.effective_buy_price().with_modifier(self.price_modifier);
            let total_price = Price {
                costs: unit_price
                    .costs
                    .iter()
                    .map(|(&k, &v)| (k, v * quantity as u64))
                    .collect(),
            };
            (total_price, item.stock, item.available, item.level_requirement, item.sellable)
        };

        if !wallet.can_afford(&price) {
            return TransactionResult::InsufficientFunds;
        }

        wallet.deduct(&price);

        // Decrease stock.
        if let Some(item) = self.get_item_mut(item_id) {
            for _ in 0..quantity {
                item.decrease_stock();
            }
        }

        // Record transaction.
        let transaction = Transaction {
            shop_id: self.id,
            item_id,
            quantity,
            total_price: price,
            is_buy: true,
            result: TransactionResult::Success,
        };
        self.add_history(transaction);

        TransactionResult::Success
    }

    /// Attempt to sell an item to the shop.
    pub fn sell(
        &mut self,
        item_id: ItemId,
        quantity: u32,
        wallet: &mut ShopWallet,
    ) -> TransactionResult {
        if !self.open {
            return TransactionResult::NotAvailable;
        }

        let sell_price = {
            let item = match self.get_item(item_id) {
                Some(i) => i,
                None => return TransactionResult::NotAvailable,
            };
            if !item.sellable {
                return TransactionResult::NotSellable;
            }
            let unit_price = item.base_sell_price.with_modifier(self.sell_modifier);
            Price {
                costs: unit_price
                    .costs
                    .iter()
                    .map(|(&k, &v)| (k, v * quantity as u64))
                    .collect(),
            }
        };

        // Add currency to wallet.
        for (&currency, &amount) in &sell_price.costs {
            wallet.add(currency, amount);
        }

        // Increase shop stock if applicable.
        if let Some(item) = self.get_item_mut(item_id) {
            if item.stock >= 0 {
                item.stock += quantity as i32;
            }
        }

        // Record transaction.
        let transaction = Transaction {
            shop_id: self.id,
            item_id,
            quantity,
            total_price: sell_price,
            is_buy: false,
            result: TransactionResult::Success,
        };
        self.add_history(transaction);

        TransactionResult::Success
    }

    /// Generate a purchase confirmation.
    pub fn preview_purchase(
        &self,
        item_id: ItemId,
        quantity: u32,
        wallet: &ShopWallet,
    ) -> Option<PurchaseConfirmation> {
        let item = self.get_item(item_id)?;
        let unit_price = item.effective_buy_price().with_modifier(self.price_modifier);
        let total_price = Price {
            costs: unit_price
                .costs
                .iter()
                .map(|(&k, &v)| (k, v * quantity as u64))
                .collect(),
        };

        let affordable = wallet.can_afford(&total_price);
        let mut remaining = wallet.balances.clone();
        if affordable {
            for (&currency, &amount) in &total_price.costs {
                let bal = remaining.entry(currency).or_insert(0);
                *bal = bal.saturating_sub(amount);
            }
        }

        Some(PurchaseConfirmation {
            item_name: item.name.clone(),
            item_icon: item.icon.clone(),
            quantity,
            total_price,
            wallet_balance: wallet.balances.clone(),
            affordable,
            remaining_balance: remaining,
        })
    }

    /// Update the shop (restock timer).
    pub fn update(&mut self, dt: f32) {
        if self.restock_interval <= 0.0 {
            return;
        }
        self.restock_timer += dt;
        if self.restock_timer >= self.restock_interval {
            self.restock_timer -= self.restock_interval;
            self.restock_all();
        }
    }

    /// Restock all items.
    pub fn restock_all(&mut self) {
        for item in &mut self.items {
            item.restock();
        }
    }

    /// Add a transaction to history.
    fn add_history(&mut self, transaction: Transaction) {
        self.history.push(transaction);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get the number of items in the shop.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Get the category names.
    pub fn category_names(&self) -> Vec<&str> {
        self.categories.iter().map(|c| c.name.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Shop manager
// ---------------------------------------------------------------------------

/// Manages multiple shops.
pub struct ShopManager {
    shops: HashMap<ShopId, Shop>,
    next_id: u32,
    currencies: Vec<CurrencyDef>,
}

impl ShopManager {
    pub fn new() -> Self {
        Self {
            shops: HashMap::new(),
            next_id: 0,
            currencies: Vec::new(),
        }
    }

    /// Register a currency type.
    pub fn register_currency(&mut self, name: &str, symbol: &str) -> CurrencyId {
        let id = CurrencyId(self.currencies.len() as u32);
        self.currencies.push(CurrencyDef {
            id,
            name: name.to_string(),
            symbol: symbol.to_string(),
            max_amount: u64::MAX,
            earnable: true,
        });
        id
    }

    /// Create a new shop.
    pub fn create_shop(&mut self, name: &str) -> ShopId {
        let id = ShopId(self.next_id);
        self.next_id += 1;
        self.shops.insert(id, Shop::new(id, name));
        id
    }

    /// Get a shop.
    pub fn shop(&self, id: ShopId) -> Option<&Shop> {
        self.shops.get(&id)
    }

    /// Get a mutable shop.
    pub fn shop_mut(&mut self, id: ShopId) -> Option<&mut Shop> {
        self.shops.get_mut(&id)
    }

    /// Update all shops.
    pub fn update(&mut self, dt: f32) {
        for shop in self.shops.values_mut() {
            shop.update(dt);
        }
    }

    /// Get all shop IDs.
    pub fn shop_ids(&self) -> Vec<ShopId> {
        self.shops.keys().copied().collect()
    }
}

impl Default for ShopManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gold() -> CurrencyId {
        CurrencyId(0)
    }

    #[test]
    fn test_buy_item() {
        let mut shop = Shop::new(ShopId(0), "Test Shop");
        shop.add_item(ShopItem::new(
            ItemId(1),
            "Sword",
            Price::single(gold(), 100),
        ));

        let mut wallet = ShopWallet::default();
        wallet.add(gold(), 500);

        let result = shop.buy(ItemId(1), 1, &mut wallet);
        assert_eq!(result, TransactionResult::Success);
        assert_eq!(wallet.balance(gold()), 400);
    }

    #[test]
    fn test_insufficient_funds() {
        let mut shop = Shop::new(ShopId(0), "Test Shop");
        shop.add_item(ShopItem::new(
            ItemId(1),
            "Sword",
            Price::single(gold(), 100),
        ));

        let mut wallet = ShopWallet::default();
        wallet.add(gold(), 50);

        let result = shop.buy(ItemId(1), 1, &mut wallet);
        assert_eq!(result, TransactionResult::InsufficientFunds);
    }

    #[test]
    fn test_limited_stock() {
        let mut shop = Shop::new(ShopId(0), "Test Shop");
        let mut item = ShopItem::new(ItemId(1), "Potion", Price::single(gold(), 10));
        item.stock = 2;
        item.max_stock = 5;
        shop.add_item(item);

        let mut wallet = ShopWallet::default();
        wallet.add(gold(), 1000);

        assert_eq!(shop.buy(ItemId(1), 1, &mut wallet), TransactionResult::Success);
        assert_eq!(shop.buy(ItemId(1), 1, &mut wallet), TransactionResult::Success);
        assert_eq!(shop.buy(ItemId(1), 1, &mut wallet), TransactionResult::OutOfStock);
    }

    #[test]
    fn test_sell_item() {
        let mut shop = Shop::new(ShopId(0), "Test Shop");
        let item = ShopItem::new(ItemId(1), "Sword", Price::single(gold(), 100));
        shop.add_item(item);

        let mut wallet = ShopWallet::default();
        let result = shop.sell(ItemId(1), 1, &mut wallet);
        assert_eq!(result, TransactionResult::Success);
        assert_eq!(wallet.balance(gold()), 50); // Sell for half.
    }

    #[test]
    fn test_discount() {
        let mut item = ShopItem::new(ItemId(1), "Sword", Price::single(gold(), 100));
        item.discount = 25;
        let price = item.effective_buy_price();
        assert_eq!(price.primary_cost(), 75);
    }
}
