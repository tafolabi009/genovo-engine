//! Game economy system.
//!
//! Provides a comprehensive economy framework including currencies, shops,
//! supply/demand pricing, NPC-to-NPC trade, auctions, loot economy balancing,
//! crafting cost calculation, and gold sink/faucet tracking.
//!
//! # Key concepts
//!
//! - **Currency**: Named currency types with conversion rates.
//! - **Wallet**: Per-entity currency storage.
//! - **Shop**: A merchant with inventory, buy/sell prices, and modifiers.
//! - **PriceModifier**: Dynamic pricing adjustments (supply/demand, reputation,
//!   events, time-of-day).
//! - **AuctionHouse**: Player-driven marketplace with bidding and buyouts.
//! - **EconomyTracker**: Monitors gold sinks (removal) and faucets (injection)
//!   to maintain economic balance.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of currency types in the game.
pub const MAX_CURRENCY_TYPES: usize = 16;

/// Maximum number of items in a single shop.
pub const MAX_SHOP_ITEMS: usize = 256;

/// Maximum number of active auctions.
pub const MAX_ACTIVE_AUCTIONS: usize = 1024;

/// Default tax rate on transactions (0..1).
pub const DEFAULT_TAX_RATE: f32 = 0.05;

/// Maximum price multiplier from modifiers.
pub const MAX_PRICE_MULTIPLIER: f32 = 5.0;

/// Minimum price multiplier from modifiers.
pub const MIN_PRICE_MULTIPLIER: f32 = 0.1;

/// Default supply/demand elasticity.
pub const DEFAULT_ELASTICITY: f32 = 0.5;

/// Maximum number of recent transactions to track per shop.
pub const MAX_TRANSACTION_HISTORY: usize = 256;

/// Default auction duration in game-time seconds.
pub const DEFAULT_AUCTION_DURATION: f64 = 3600.0;

/// Minimum bid increment as a fraction of current price.
pub const MIN_BID_INCREMENT: f32 = 0.05;

/// Maximum economy tracking window entries.
pub const MAX_TRACKING_ENTRIES: usize = 1024;

/// Default buy-back ratio (what percentage of item value a shop pays).
pub const DEFAULT_BUY_BACK_RATIO: f32 = 0.4;

/// Price smoothing factor for supply/demand (0..1, higher = more responsive).
pub const PRICE_SMOOTHING: f32 = 0.1;

// ---------------------------------------------------------------------------
// CurrencyType
// ---------------------------------------------------------------------------

/// A type of in-game currency.
#[derive(Debug, Clone)]
pub struct CurrencyType {
    /// Unique currency identifier.
    pub id: u32,
    /// Display name (e.g., "Gold", "Gems").
    pub name: String,
    /// Short code (e.g., "G", "GEM").
    pub short_code: String,
    /// Icon identifier for UI.
    pub icon: String,
    /// Conversion rate to the base currency (1.0 = base).
    pub conversion_rate: f32,
    /// Maximum amount an entity can hold (0 = unlimited).
    pub max_amount: u64,
    /// Whether this currency can be traded between players.
    pub tradable: bool,
    /// Whether this currency is a premium/real-money currency.
    pub premium: bool,
}

impl CurrencyType {
    /// Create a new currency type.
    pub fn new(id: u32, name: impl Into<String>, conversion_rate: f32) -> Self {
        let name = name.into();
        let short = name.chars().take(3).collect::<String>().to_uppercase();
        Self {
            id,
            name,
            short_code: short,
            icon: String::new(),
            conversion_rate,
            max_amount: 0,
            tradable: true,
            premium: false,
        }
    }

    /// Set as a premium currency.
    pub fn as_premium(mut self) -> Self {
        self.premium = true;
        self.tradable = false;
        self
    }

    /// Set a maximum amount.
    pub fn with_max(mut self, max: u64) -> Self {
        self.max_amount = max;
        self
    }

    /// Convert an amount of this currency to the base currency.
    pub fn to_base(&self, amount: u64) -> f64 {
        amount as f64 * self.conversion_rate as f64
    }

    /// Convert from base currency to this currency.
    pub fn from_base(&self, base_amount: f64) -> u64 {
        if self.conversion_rate <= 0.0 {
            return 0;
        }
        (base_amount / self.conversion_rate as f64) as u64
    }
}

// ---------------------------------------------------------------------------
// Wallet
// ---------------------------------------------------------------------------

/// Per-entity storage of multiple currency types.
#[derive(Debug, Clone)]
pub struct Wallet {
    /// Owner entity ID.
    pub owner: u64,
    /// Currency balances by currency ID.
    balances: HashMap<u32, u64>,
    /// Total income tracked.
    pub total_income: u64,
    /// Total spending tracked.
    pub total_spending: u64,
}

impl Wallet {
    /// Create a new empty wallet.
    pub fn new(owner: u64) -> Self {
        Self {
            owner,
            balances: HashMap::new(),
            total_income: 0,
            total_spending: 0,
        }
    }

    /// Get the balance of a specific currency.
    pub fn balance(&self, currency_id: u32) -> u64 {
        self.balances.get(&currency_id).copied().unwrap_or(0)
    }

    /// Add currency to the wallet. Returns the new balance.
    pub fn add(&mut self, currency_id: u32, amount: u64, max: u64) -> u64 {
        let current = self.balances.entry(currency_id).or_insert(0);
        if max > 0 {
            *current = (*current + amount).min(max);
        } else {
            *current += amount;
        }
        self.total_income += amount;
        *current
    }

    /// Remove currency from the wallet. Returns true if successful.
    pub fn remove(&mut self, currency_id: u32, amount: u64) -> bool {
        let current = self.balances.entry(currency_id).or_insert(0);
        if *current >= amount {
            *current -= amount;
            self.total_spending += amount;
            true
        } else {
            false
        }
    }

    /// Check if the wallet has at least the given amount.
    pub fn has(&self, currency_id: u32, amount: u64) -> bool {
        self.balance(currency_id) >= amount
    }

    /// Transfer currency to another wallet. Returns true if successful.
    pub fn transfer_to(
        &mut self,
        other: &mut Wallet,
        currency_id: u32,
        amount: u64,
        max: u64,
    ) -> bool {
        if self.remove(currency_id, amount) {
            other.add(currency_id, amount, max);
            true
        } else {
            false
        }
    }

    /// Get all non-zero currency balances.
    pub fn all_balances(&self) -> &HashMap<u32, u64> {
        &self.balances
    }

    /// Clear all balances.
    pub fn clear(&mut self) {
        self.balances.clear();
    }
}

// ---------------------------------------------------------------------------
// PriceModifier
// ---------------------------------------------------------------------------

/// A modifier that adjusts item prices dynamically.
#[derive(Debug, Clone)]
pub struct PriceModifier {
    /// Modifier source.
    pub source: PriceModifierSource,
    /// Multiplier (1.0 = no change, > 1.0 = more expensive).
    pub multiplier: f32,
    /// Priority (higher = applied later).
    pub priority: i32,
    /// Expiration game time (0 = never expires).
    pub expires_at: f64,
    /// Whether this modifier is currently active.
    pub active: bool,
}

/// Source/reason for a price modifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PriceModifierSource {
    /// Supply/demand dynamics.
    SupplyDemand,
    /// Faction reputation with the buyer.
    Reputation,
    /// Special event (holiday, war, famine).
    Event(String),
    /// Time-of-day modifier.
    TimeOfDay,
    /// Character skill/perk (e.g., barter skill).
    Skill(String),
    /// Location-based (e.g., remote village = higher prices).
    Location,
    /// Merchant personality (greedy vs generous).
    Personality,
    /// Bulk discount.
    BulkDiscount,
}

impl PriceModifier {
    /// Create a new price modifier.
    pub fn new(source: PriceModifierSource, multiplier: f32) -> Self {
        Self {
            source,
            multiplier: multiplier.clamp(MIN_PRICE_MULTIPLIER, MAX_PRICE_MULTIPLIER),
            priority: 0,
            expires_at: 0.0,
            active: true,
        }
    }

    /// Set expiration time.
    pub fn expires_at(mut self, time: f64) -> Self {
        self.expires_at = time;
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Check if this modifier has expired.
    pub fn is_expired(&self, game_time: f64) -> bool {
        self.expires_at > 0.0 && game_time >= self.expires_at
    }
}

// ---------------------------------------------------------------------------
// ShopItem
// ---------------------------------------------------------------------------

/// An item listing in a shop.
#[derive(Debug, Clone)]
pub struct ShopItem {
    /// Item definition ID.
    pub item_id: String,
    /// Base price in the shop's currency.
    pub base_price: u64,
    /// Current adjusted price (after modifiers).
    pub current_price: u64,
    /// Quantity available (0 = unlimited).
    pub quantity: u32,
    /// Maximum quantity the shop can stock.
    pub max_quantity: u32,
    /// Restock rate (units per game-hour).
    pub restock_rate: f32,
    /// Time until next restock.
    pub restock_timer: f32,
    /// Currency ID used for this item.
    pub currency_id: u32,
    /// Whether the shop will buy this item from players.
    pub will_buy: bool,
    /// Buy price (what the shop pays the player).
    pub buy_price: u64,
    /// Supply level (affects supply/demand pricing).
    supply: f32,
    /// Demand level.
    demand: f32,
}

impl ShopItem {
    /// Create a new shop item.
    pub fn new(
        item_id: impl Into<String>,
        base_price: u64,
        quantity: u32,
        currency_id: u32,
    ) -> Self {
        Self {
            item_id: item_id.into(),
            base_price,
            current_price: base_price,
            quantity,
            max_quantity: quantity * 2,
            restock_rate: 1.0,
            restock_timer: 0.0,
            currency_id,
            will_buy: true,
            buy_price: (base_price as f32 * DEFAULT_BUY_BACK_RATIO) as u64,
            supply: 1.0,
            demand: 1.0,
        }
    }

    /// Update supply/demand values after a transaction.
    pub fn record_sale(&mut self) {
        self.demand += 0.1;
        if self.quantity > 0 {
            self.quantity -= 1;
        }
        self.supply = self.quantity as f32 / self.max_quantity.max(1) as f32;
    }

    /// Record a buy-back (player sells to shop).
    pub fn record_buyback(&mut self) {
        self.supply += 0.1;
        self.quantity = self.quantity.saturating_add(1).min(self.max_quantity);
    }

    /// Update the current price based on supply/demand.
    pub fn update_price(&mut self, modifiers: &[PriceModifier], game_time: f64) {
        let mut multiplier = 1.0f32;

        // Supply/demand
        if self.supply > 0.01 {
            let sd_ratio = self.demand / self.supply;
            let sd_modifier = 1.0 + (sd_ratio - 1.0) * DEFAULT_ELASTICITY;
            multiplier *= sd_modifier.clamp(MIN_PRICE_MULTIPLIER, MAX_PRICE_MULTIPLIER);
        }

        // Apply other modifiers
        let mut mods: Vec<&PriceModifier> = modifiers
            .iter()
            .filter(|m| m.active && !m.is_expired(game_time))
            .collect();
        mods.sort_by_key(|m| m.priority);

        for m in mods {
            multiplier *= m.multiplier;
        }

        multiplier = multiplier.clamp(MIN_PRICE_MULTIPLIER, MAX_PRICE_MULTIPLIER);

        let new_price = (self.base_price as f32 * multiplier) as u64;
        // Smooth price changes
        self.current_price = ((self.current_price as f32 * (1.0 - PRICE_SMOOTHING))
            + (new_price as f32 * PRICE_SMOOTHING)) as u64;
        self.current_price = self.current_price.max(1);

        // Update buy price
        self.buy_price =
            (self.current_price as f32 * DEFAULT_BUY_BACK_RATIO) as u64;
    }

    /// Restock the item over time.
    pub fn restock(&mut self, dt: f32) {
        if self.quantity >= self.max_quantity {
            return;
        }
        self.restock_timer += dt;
        let restock_interval = 3600.0 / self.restock_rate.max(0.001);
        while self.restock_timer >= restock_interval && self.quantity < self.max_quantity {
            self.restock_timer -= restock_interval;
            self.quantity += 1;
        }
    }

    /// Gradually decay demand toward 1.0.
    pub fn decay_demand(&mut self, dt: f32) {
        self.demand += (1.0 - self.demand) * 0.01 * dt;
    }
}

// ---------------------------------------------------------------------------
// TransactionRecord
// ---------------------------------------------------------------------------

/// A record of a completed transaction.
#[derive(Debug, Clone)]
pub struct TransactionRecord {
    /// Buyer entity ID.
    pub buyer: u64,
    /// Seller entity ID (0 = shop/system).
    pub seller: u64,
    /// Item ID.
    pub item_id: String,
    /// Quantity.
    pub quantity: u32,
    /// Price per unit.
    pub unit_price: u64,
    /// Total price.
    pub total_price: u64,
    /// Currency ID.
    pub currency_id: u32,
    /// Transaction type.
    pub transaction_type: TransactionType,
    /// Timestamp.
    pub timestamp: f64,
}

/// Type of economic transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionType {
    /// Player buys from shop.
    ShopBuy,
    /// Player sells to shop.
    ShopSell,
    /// Player-to-player trade.
    PlayerTrade,
    /// NPC-to-NPC trade.
    NpcTrade,
    /// Auction purchase.
    AuctionBuy,
    /// Crafting cost.
    CraftingCost,
    /// Quest reward.
    QuestReward,
    /// Loot pickup.
    LootPickup,
    /// Tax payment.
    Tax,
    /// Repair cost.
    RepairCost,
    /// Service fee (fast travel, etc.).
    ServiceFee,
}

impl TransactionType {
    /// Whether this type is a gold sink (removes currency from the economy).
    pub fn is_sink(&self) -> bool {
        matches!(
            self,
            Self::ShopBuy
                | Self::CraftingCost
                | Self::Tax
                | Self::RepairCost
                | Self::ServiceFee
        )
    }

    /// Whether this type is a gold faucet (injects currency into the economy).
    pub fn is_faucet(&self) -> bool {
        matches!(
            self,
            Self::QuestReward | Self::LootPickup | Self::ShopSell
        )
    }
}

// ---------------------------------------------------------------------------
// Shop
// ---------------------------------------------------------------------------

/// A merchant shop with inventory and dynamic pricing.
pub struct Shop {
    /// Shop unique identifier.
    pub id: u32,
    /// Display name.
    pub name: String,
    /// Owner entity ID (the merchant NPC).
    pub owner: u64,
    /// Items for sale.
    items: Vec<ShopItem>,
    /// Active price modifiers.
    modifiers: Vec<PriceModifier>,
    /// Transaction history.
    history: VecDeque<TransactionRecord>,
    /// Currency used by this shop.
    pub primary_currency: u32,
    /// Tax rate on transactions.
    pub tax_rate: f32,
    /// Whether the shop is open.
    pub open: bool,
    /// Shop wallet (the shop's own money for buying items).
    pub wallet: Wallet,
    /// Total revenue.
    pub total_revenue: u64,
    /// Total cost of bought items.
    pub total_purchases: u64,
}

impl Shop {
    /// Create a new shop.
    pub fn new(id: u32, name: impl Into<String>, owner: u64, currency: u32) -> Self {
        Self {
            id,
            name: name.into(),
            owner,
            items: Vec::new(),
            modifiers: Vec::new(),
            history: VecDeque::new(),
            primary_currency: currency,
            tax_rate: DEFAULT_TAX_RATE,
            open: true,
            wallet: Wallet::new(owner),
            total_revenue: 0,
            total_purchases: 0,
        }
    }

    /// Add an item to the shop.
    pub fn add_item(&mut self, item: ShopItem) {
        if self.items.len() < MAX_SHOP_ITEMS {
            self.items.push(item);
        }
    }

    /// Remove an item by ID.
    pub fn remove_item(&mut self, item_id: &str) {
        self.items.retain(|i| i.item_id != item_id);
    }

    /// Get an item's current price.
    pub fn get_price(&self, item_id: &str) -> Option<u64> {
        self.items
            .iter()
            .find(|i| i.item_id == item_id)
            .map(|i| i.current_price)
    }

    /// Get what the shop will pay for an item.
    pub fn get_buy_price(&self, item_id: &str) -> Option<u64> {
        self.items
            .iter()
            .find(|i| i.item_id == item_id && i.will_buy)
            .map(|i| i.buy_price)
    }

    /// Attempt to sell an item to a buyer.
    pub fn sell_to(
        &mut self,
        buyer: &mut Wallet,
        item_id: &str,
        quantity: u32,
        game_time: f64,
    ) -> Result<TransactionRecord, ShopError> {
        if !self.open {
            return Err(ShopError::ShopClosed);
        }

        let item = self
            .items
            .iter_mut()
            .find(|i| i.item_id == item_id)
            .ok_or(ShopError::ItemNotFound)?;

        if item.quantity > 0 && item.quantity < quantity {
            return Err(ShopError::InsufficientStock);
        }

        let unit_price = item.current_price;
        let subtotal = unit_price * quantity as u64;
        let tax = (subtotal as f32 * self.tax_rate) as u64;
        let total = subtotal + tax;

        if !buyer.has(item.currency_id, total) {
            return Err(ShopError::InsufficientFunds);
        }

        // Execute transaction
        buyer.remove(item.currency_id, total);
        self.wallet.add(item.currency_id, subtotal, 0);
        self.total_revenue += total;

        for _ in 0..quantity {
            item.record_sale();
        }

        let record = TransactionRecord {
            buyer: buyer.owner,
            seller: self.owner,
            item_id: item_id.to_string(),
            quantity,
            unit_price,
            total_price: total,
            currency_id: item.currency_id,
            transaction_type: TransactionType::ShopBuy,
            timestamp: game_time,
        };

        self.history.push_back(record.clone());
        if self.history.len() > MAX_TRANSACTION_HISTORY {
            self.history.pop_front();
        }

        Ok(record)
    }

    /// Buy an item from a player (player sells to the shop).
    pub fn buy_from(
        &mut self,
        seller: &mut Wallet,
        item_id: &str,
        quantity: u32,
        game_time: f64,
    ) -> Result<TransactionRecord, ShopError> {
        if !self.open {
            return Err(ShopError::ShopClosed);
        }

        let item = self
            .items
            .iter_mut()
            .find(|i| i.item_id == item_id && i.will_buy)
            .ok_or(ShopError::WontBuy)?;

        let unit_price = item.buy_price;
        let total = unit_price * quantity as u64;

        if !self.wallet.has(item.currency_id, total) {
            return Err(ShopError::ShopCantAfford);
        }

        // Execute transaction
        self.wallet.remove(item.currency_id, total);
        seller.add(item.currency_id, total, 0);
        self.total_purchases += total;

        for _ in 0..quantity {
            item.record_buyback();
        }

        let record = TransactionRecord {
            buyer: self.owner,
            seller: seller.owner,
            item_id: item_id.to_string(),
            quantity,
            unit_price,
            total_price: total,
            currency_id: item.currency_id,
            transaction_type: TransactionType::ShopSell,
            timestamp: game_time,
        };

        self.history.push_back(record.clone());
        if self.history.len() > MAX_TRANSACTION_HISTORY {
            self.history.pop_front();
        }

        Ok(record)
    }

    /// Add a price modifier.
    pub fn add_modifier(&mut self, modifier: PriceModifier) {
        self.modifiers.push(modifier);
    }

    /// Remove expired modifiers.
    pub fn clean_modifiers(&mut self, game_time: f64) {
        self.modifiers.retain(|m| !m.is_expired(game_time));
    }

    /// Update all prices and restock.
    pub fn update(&mut self, dt: f32, game_time: f64) {
        self.clean_modifiers(game_time);

        for item in &mut self.items {
            item.update_price(&self.modifiers, game_time);
            item.restock(dt);
            item.decay_demand(dt);
        }
    }

    /// Get the transaction history.
    pub fn transaction_history(&self) -> &VecDeque<TransactionRecord> {
        &self.history
    }

    /// Get all items.
    pub fn items(&self) -> &[ShopItem] {
        &self.items
    }

    /// Get item count.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Calculate the shop's total inventory value.
    pub fn inventory_value(&self) -> u64 {
        self.items
            .iter()
            .map(|i| i.current_price * i.quantity as u64)
            .sum()
    }

    /// Get the most popular item (most sales).
    pub fn most_popular_item(&self) -> Option<&str> {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for record in &self.history {
            if record.transaction_type == TransactionType::ShopBuy {
                *counts.entry(&record.item_id).or_insert(0) += record.quantity as usize;
            }
        }
        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(id, _)| id)
    }
}

/// Shop operation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShopError {
    /// The shop is closed.
    ShopClosed,
    /// Item not found in the shop.
    ItemNotFound,
    /// Not enough stock.
    InsufficientStock,
    /// Buyer doesn't have enough currency.
    InsufficientFunds,
    /// Shop won't buy this item.
    WontBuy,
    /// Shop can't afford to buy.
    ShopCantAfford,
}

// ---------------------------------------------------------------------------
// Auction
// ---------------------------------------------------------------------------

/// An item listed for auction.
#[derive(Debug, Clone)]
pub struct AuctionListing {
    /// Unique auction ID.
    pub id: u64,
    /// Seller entity ID.
    pub seller: u64,
    /// Item ID.
    pub item_id: String,
    /// Quantity.
    pub quantity: u32,
    /// Starting bid.
    pub starting_bid: u64,
    /// Current highest bid.
    pub current_bid: u64,
    /// Current highest bidder.
    pub highest_bidder: Option<u64>,
    /// Buyout price (0 = no buyout).
    pub buyout_price: u64,
    /// Currency ID.
    pub currency_id: u32,
    /// Listing timestamp.
    pub listed_at: f64,
    /// Duration in game-time seconds.
    pub duration: f64,
    /// Number of bids placed.
    pub bid_count: u32,
    /// Whether the auction has ended.
    pub ended: bool,
}

impl AuctionListing {
    /// Create a new auction listing.
    pub fn new(
        id: u64,
        seller: u64,
        item_id: impl Into<String>,
        quantity: u32,
        starting_bid: u64,
        currency_id: u32,
        listed_at: f64,
    ) -> Self {
        Self {
            id,
            seller,
            item_id: item_id.into(),
            quantity,
            starting_bid,
            current_bid: starting_bid,
            highest_bidder: None,
            buyout_price: 0,
            currency_id,
            listed_at,
            duration: DEFAULT_AUCTION_DURATION,
            bid_count: 0,
            ended: false,
        }
    }

    /// Set the buyout price.
    pub fn with_buyout(mut self, price: u64) -> Self {
        self.buyout_price = price;
        self
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration: f64) -> Self {
        self.duration = duration;
        self
    }

    /// Check if the auction has expired.
    pub fn is_expired(&self, game_time: f64) -> bool {
        game_time >= self.listed_at + self.duration
    }

    /// Place a bid. Returns true if successful.
    pub fn place_bid(&mut self, bidder: u64, amount: u64) -> bool {
        if self.ended {
            return false;
        }
        if bidder == self.seller {
            return false;
        }

        let min_bid = self.current_bid
            + (self.current_bid as f32 * MIN_BID_INCREMENT) as u64;
        let min_bid = min_bid.max(self.starting_bid);

        if amount < min_bid {
            return false;
        }

        self.current_bid = amount;
        self.highest_bidder = Some(bidder);
        self.bid_count += 1;
        true
    }

    /// Execute a buyout. Returns true if successful.
    pub fn buyout(&mut self, buyer: u64) -> bool {
        if self.ended || self.buyout_price == 0 {
            return false;
        }
        if buyer == self.seller {
            return false;
        }

        self.current_bid = self.buyout_price;
        self.highest_bidder = Some(buyer);
        self.ended = true;
        true
    }

    /// End the auction (called when time expires).
    pub fn end(&mut self) {
        self.ended = true;
    }

    /// Get the winner (highest bidder when ended).
    pub fn winner(&self) -> Option<u64> {
        if self.ended {
            self.highest_bidder
        } else {
            None
        }
    }

    /// Remaining time in seconds.
    pub fn remaining_time(&self, game_time: f64) -> f64 {
        (self.listed_at + self.duration - game_time).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// AuctionHouse
// ---------------------------------------------------------------------------

/// Player-driven marketplace with bidding.
pub struct AuctionHouse {
    /// Active listings.
    listings: Vec<AuctionListing>,
    /// Completed/expired listings.
    completed: VecDeque<AuctionListing>,
    /// Next listing ID.
    next_id: u64,
    /// Listing fee as a fraction of starting bid.
    pub listing_fee_rate: f32,
    /// Transaction fee as a fraction of final price.
    pub transaction_fee_rate: f32,
    /// Maximum listings per seller.
    pub max_per_seller: usize,
}

impl AuctionHouse {
    /// Create a new auction house.
    pub fn new() -> Self {
        Self {
            listings: Vec::new(),
            completed: VecDeque::new(),
            next_id: 1,
            listing_fee_rate: 0.02,
            transaction_fee_rate: 0.05,
            max_per_seller: 50,
        }
    }

    /// Create a new listing.
    pub fn create_listing(
        &mut self,
        seller: u64,
        item_id: impl Into<String>,
        quantity: u32,
        starting_bid: u64,
        currency_id: u32,
        game_time: f64,
    ) -> Result<u64, AuctionError> {
        if self.listings.len() >= MAX_ACTIVE_AUCTIONS {
            return Err(AuctionError::HouseFull);
        }

        let seller_count = self
            .listings
            .iter()
            .filter(|l| l.seller == seller)
            .count();
        if seller_count >= self.max_per_seller {
            return Err(AuctionError::SellerLimitReached);
        }

        let id = self.next_id;
        self.next_id += 1;

        let listing = AuctionListing::new(
            id,
            seller,
            item_id,
            quantity,
            starting_bid,
            currency_id,
            game_time,
        );
        self.listings.push(listing);

        Ok(id)
    }

    /// Place a bid on a listing.
    pub fn bid(
        &mut self,
        listing_id: u64,
        bidder: u64,
        amount: u64,
    ) -> Result<(), AuctionError> {
        let listing = self
            .listings
            .iter_mut()
            .find(|l| l.id == listing_id && !l.ended)
            .ok_or(AuctionError::ListingNotFound)?;

        if !listing.place_bid(bidder, amount) {
            return Err(AuctionError::BidTooLow);
        }

        Ok(())
    }

    /// Execute a buyout on a listing.
    pub fn buyout(
        &mut self,
        listing_id: u64,
        buyer: u64,
    ) -> Result<u64, AuctionError> {
        let listing = self
            .listings
            .iter_mut()
            .find(|l| l.id == listing_id && !l.ended)
            .ok_or(AuctionError::ListingNotFound)?;

        if listing.buyout_price == 0 {
            return Err(AuctionError::NoBuyout);
        }

        let price = listing.buyout_price;
        if !listing.buyout(buyer) {
            return Err(AuctionError::BuyoutFailed);
        }

        Ok(price)
    }

    /// Update the auction house (expire listings, etc.).
    pub fn update(&mut self, game_time: f64) {
        for listing in &mut self.listings {
            if !listing.ended && listing.is_expired(game_time) {
                listing.end();
            }
        }

        // Move completed listings
        let (ended, active): (Vec<_>, Vec<_>) =
            self.listings.drain(..).partition(|l| l.ended);
        self.listings = active;

        for listing in ended {
            self.completed.push_back(listing);
        }

        // Cap completed history
        while self.completed.len() > MAX_ACTIVE_AUCTIONS {
            self.completed.pop_front();
        }
    }

    /// Search listings by item ID.
    pub fn search(&self, item_id: &str) -> Vec<&AuctionListing> {
        self.listings
            .iter()
            .filter(|l| l.item_id == item_id && !l.ended)
            .collect()
    }

    /// Get all active listings.
    pub fn active_listings(&self) -> &[AuctionListing] {
        &self.listings
    }

    /// Get completed listings.
    pub fn completed_listings(&self) -> &VecDeque<AuctionListing> {
        &self.completed
    }

    /// Get the listing fee for a starting bid.
    pub fn listing_fee(&self, starting_bid: u64) -> u64 {
        (starting_bid as f32 * self.listing_fee_rate) as u64
    }

    /// Calculate the average selling price for an item.
    pub fn average_price(&self, item_id: &str) -> Option<u64> {
        let prices: Vec<u64> = self
            .completed
            .iter()
            .filter(|l| l.item_id == item_id && l.highest_bidder.is_some())
            .map(|l| l.current_bid)
            .collect();

        if prices.is_empty() {
            None
        } else {
            Some(prices.iter().sum::<u64>() / prices.len() as u64)
        }
    }

    /// Active listing count.
    pub fn active_count(&self) -> usize {
        self.listings.len()
    }
}

impl Default for AuctionHouse {
    fn default() -> Self {
        Self::new()
    }
}

/// Auction errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuctionError {
    HouseFull,
    SellerLimitReached,
    ListingNotFound,
    BidTooLow,
    NoBuyout,
    BuyoutFailed,
    InsufficientFunds,
}

// ---------------------------------------------------------------------------
// EconomyTracker — gold sink/faucet monitoring
// ---------------------------------------------------------------------------

/// Tracks economic health by monitoring currency flow.
pub struct EconomyTracker {
    /// Sink/faucet entries per currency.
    entries: HashMap<u32, VecDeque<FlowEntry>>,
    /// Running totals.
    totals: HashMap<u32, FlowTotals>,
    /// Maximum entries per currency.
    max_entries: usize,
}

/// A single entry in the economy flow log.
#[derive(Debug, Clone)]
pub struct FlowEntry {
    /// Amount of currency.
    pub amount: u64,
    /// Whether this was a sink (true) or faucet (false).
    pub is_sink: bool,
    /// Source description.
    pub source: String,
    /// Timestamp.
    pub timestamp: f64,
}

/// Running totals for economy tracking.
#[derive(Debug, Clone, Default)]
pub struct FlowTotals {
    /// Total currency removed from the economy.
    pub total_sinks: u64,
    /// Total currency injected into the economy.
    pub total_faucets: u64,
    /// Number of sink events.
    pub sink_count: u64,
    /// Number of faucet events.
    pub faucet_count: u64,
}

impl FlowTotals {
    /// Net flow (positive = inflation, negative = deflation).
    pub fn net_flow(&self) -> i64 {
        self.total_faucets as i64 - self.total_sinks as i64
    }

    /// Sink-to-faucet ratio (1.0 = balanced, < 1.0 = inflation, > 1.0 = deflation).
    pub fn ratio(&self) -> f32 {
        if self.total_faucets == 0 {
            return if self.total_sinks > 0 {
                f32::MAX
            } else {
                1.0
            };
        }
        self.total_sinks as f32 / self.total_faucets as f32
    }

    /// Whether the economy is inflationary.
    pub fn is_inflationary(&self) -> bool {
        self.ratio() < 0.9
    }

    /// Whether the economy is deflationary.
    pub fn is_deflationary(&self) -> bool {
        self.ratio() > 1.1
    }
}

impl EconomyTracker {
    /// Create a new economy tracker.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            totals: HashMap::new(),
            max_entries: MAX_TRACKING_ENTRIES,
        }
    }

    /// Record a gold sink event.
    pub fn record_sink(
        &mut self,
        currency_id: u32,
        amount: u64,
        source: impl Into<String>,
        timestamp: f64,
    ) {
        let entry = FlowEntry {
            amount,
            is_sink: true,
            source: source.into(),
            timestamp,
        };

        let entries = self.entries.entry(currency_id).or_insert_with(VecDeque::new);
        entries.push_back(entry);
        if entries.len() > self.max_entries {
            entries.pop_front();
        }

        let totals = self.totals.entry(currency_id).or_insert_with(Default::default);
        totals.total_sinks += amount;
        totals.sink_count += 1;
    }

    /// Record a gold faucet event.
    pub fn record_faucet(
        &mut self,
        currency_id: u32,
        amount: u64,
        source: impl Into<String>,
        timestamp: f64,
    ) {
        let entry = FlowEntry {
            amount,
            is_sink: false,
            source: source.into(),
            timestamp,
        };

        let entries = self.entries.entry(currency_id).or_insert_with(VecDeque::new);
        entries.push_back(entry);
        if entries.len() > self.max_entries {
            entries.pop_front();
        }

        let totals = self.totals.entry(currency_id).or_insert_with(Default::default);
        totals.total_faucets += amount;
        totals.faucet_count += 1;
    }

    /// Record a transaction (automatically categorizes as sink or faucet).
    pub fn record_transaction(&mut self, record: &TransactionRecord) {
        if record.transaction_type.is_sink() {
            self.record_sink(
                record.currency_id,
                record.total_price,
                format!("{:?}", record.transaction_type),
                record.timestamp,
            );
        } else if record.transaction_type.is_faucet() {
            self.record_faucet(
                record.currency_id,
                record.total_price,
                format!("{:?}", record.transaction_type),
                record.timestamp,
            );
        }
    }

    /// Get the flow totals for a currency.
    pub fn totals(&self, currency_id: u32) -> Option<&FlowTotals> {
        self.totals.get(&currency_id)
    }

    /// Get recent flow entries for a currency.
    pub fn recent_entries(&self, currency_id: u32, count: usize) -> Vec<&FlowEntry> {
        self.entries
            .get(&currency_id)
            .map_or(Vec::new(), |entries| {
                entries.iter().rev().take(count).collect()
            })
    }

    /// Calculate flow rate over a time window (amount per second).
    pub fn flow_rate(
        &self,
        currency_id: u32,
        window_start: f64,
        window_end: f64,
    ) -> (f64, f64) {
        let duration = (window_end - window_start).max(1.0);

        let entries = match self.entries.get(&currency_id) {
            Some(e) => e,
            None => return (0.0, 0.0),
        };

        let mut sinks = 0u64;
        let mut faucets = 0u64;

        for entry in entries {
            if entry.timestamp >= window_start && entry.timestamp <= window_end {
                if entry.is_sink {
                    sinks += entry.amount;
                } else {
                    faucets += entry.amount;
                }
            }
        }

        (sinks as f64 / duration, faucets as f64 / duration)
    }

    /// Get a summary report string.
    pub fn summary_report(&self, currency_id: u32) -> String {
        match self.totals.get(&currency_id) {
            Some(totals) => {
                format!(
                    "Currency {}: Sinks={} ({} events), Faucets={} ({} events), Net={}, Ratio={:.2}, Status={}",
                    currency_id,
                    totals.total_sinks,
                    totals.sink_count,
                    totals.total_faucets,
                    totals.faucet_count,
                    totals.net_flow(),
                    totals.ratio(),
                    if totals.is_inflationary() {
                        "INFLATIONARY"
                    } else if totals.is_deflationary() {
                        "DEFLATIONARY"
                    } else {
                        "BALANCED"
                    }
                )
            }
            None => format!("Currency {}: No data", currency_id),
        }
    }
}

impl Default for EconomyTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CraftingCostCalculator
// ---------------------------------------------------------------------------

/// Calculates the cost of crafting items based on material prices.
pub struct CraftingCostCalculator {
    /// Base material prices (item_id -> base price).
    material_prices: HashMap<String, u64>,
    /// Crafting station fees.
    station_fees: HashMap<String, u64>,
    /// Skill-based cost reduction (0..1).
    pub skill_discount: f32,
}

impl CraftingCostCalculator {
    /// Create a new calculator.
    pub fn new() -> Self {
        Self {
            material_prices: HashMap::new(),
            station_fees: HashMap::new(),
            skill_discount: 0.0,
        }
    }

    /// Set a material's base price.
    pub fn set_material_price(&mut self, item_id: impl Into<String>, price: u64) {
        self.material_prices.insert(item_id.into(), price);
    }

    /// Set a crafting station fee.
    pub fn set_station_fee(&mut self, station_type: impl Into<String>, fee: u64) {
        self.station_fees.insert(station_type.into(), fee);
    }

    /// Calculate the total cost to craft a recipe.
    pub fn calculate_cost(
        &self,
        materials: &[(String, u32)],
        station_type: Option<&str>,
    ) -> CraftingCost {
        let mut material_cost = 0u64;
        let mut missing_prices = Vec::new();

        for (item_id, quantity) in materials {
            match self.material_prices.get(item_id) {
                Some(&price) => {
                    material_cost += price * *quantity as u64;
                }
                None => {
                    missing_prices.push(item_id.clone());
                }
            }
        }

        let station_fee = station_type
            .and_then(|st| self.station_fees.get(st))
            .copied()
            .unwrap_or(0);

        let subtotal = material_cost + station_fee;
        let discount = (subtotal as f32 * self.skill_discount) as u64;
        let total = subtotal.saturating_sub(discount);

        CraftingCost {
            material_cost,
            station_fee,
            skill_discount: discount,
            total,
            missing_prices,
        }
    }
}

impl Default for CraftingCostCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a crafting cost calculation.
#[derive(Debug, Clone)]
pub struct CraftingCost {
    /// Cost of materials.
    pub material_cost: u64,
    /// Crafting station fee.
    pub station_fee: u64,
    /// Discount from skills.
    pub skill_discount: u64,
    /// Final total cost.
    pub total: u64,
    /// Materials with unknown prices.
    pub missing_prices: Vec<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wallet_operations() {
        let mut wallet = Wallet::new(1);
        wallet.add(0, 100, 0);
        assert_eq!(wallet.balance(0), 100);

        assert!(wallet.remove(0, 30));
        assert_eq!(wallet.balance(0), 70);

        assert!(!wallet.remove(0, 100));
        assert_eq!(wallet.balance(0), 70);
    }

    #[test]
    fn test_shop_transaction() {
        let mut shop = Shop::new(1, "Test Shop", 100, 0);
        shop.wallet.add(0, 10000, 0);
        shop.add_item(ShopItem::new("sword", 50, 10, 0));

        let mut buyer_wallet = Wallet::new(2);
        buyer_wallet.add(0, 200, 0);

        let result = shop.sell_to(&mut buyer_wallet, "sword", 1, 0.0);
        assert!(result.is_ok());

        let record = result.unwrap();
        assert_eq!(record.unit_price, 50);
        assert!(buyer_wallet.balance(0) < 200);
    }

    #[test]
    fn test_auction_bidding() {
        let mut ah = AuctionHouse::new();
        let id = ah.create_listing(1, "rare_sword", 1, 100, 0, 0.0).unwrap();

        assert!(ah.bid(id, 2, 110).is_ok());
        assert!(ah.bid(id, 3, 120).is_ok());
        assert!(ah.bid(id, 2, 115).is_err()); // too low

        let listing = ah.listings.iter().find(|l| l.id == id).unwrap();
        assert_eq!(listing.current_bid, 120);
        assert_eq!(listing.highest_bidder, Some(3));
    }

    #[test]
    fn test_economy_tracker() {
        let mut tracker = EconomyTracker::new();
        tracker.record_sink(0, 100, "shop_purchase", 1.0);
        tracker.record_faucet(0, 150, "quest_reward", 2.0);

        let totals = tracker.totals(0).unwrap();
        assert_eq!(totals.total_sinks, 100);
        assert_eq!(totals.total_faucets, 150);
        assert!(totals.is_inflationary());
    }

    #[test]
    fn test_crafting_cost() {
        let mut calc = CraftingCostCalculator::new();
        calc.set_material_price("iron_ore", 10);
        calc.set_material_price("wood", 5);
        calc.set_station_fee("forge", 20);

        let cost = calc.calculate_cost(
            &[("iron_ore".to_string(), 3), ("wood".to_string(), 2)],
            Some("forge"),
        );
        assert_eq!(cost.material_cost, 40); // 30 + 10
        assert_eq!(cost.station_fee, 20);
        assert_eq!(cost.total, 60);
    }
}
