// engine/networking/src/server_browser.rs
//
// Server browser: server list with ping/player count/map/mode, server query
// protocol, favorites, history, password servers, server filtering/sorting,
// refresh mechanism.

use std::collections::HashMap;
use std::time::{Duration, Instant};

pub type ServerId = u64;

// --- Server info ---
#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub id: ServerId,
    pub name: String,
    pub address: String,
    pub port: u16,
    pub map_name: String,
    pub game_mode: String,
    pub current_players: u32,
    pub max_players: u32,
    pub ping_ms: u32,
    pub has_password: bool,
    pub is_official: bool,
    pub is_dedicated: bool,
    pub version: String,
    pub tags: Vec<String>,
    pub custom_data: HashMap<String, String>,
    pub last_queried: Option<Instant>,
    pub anti_cheat: bool,
    pub region: String,
    pub country_code: String,
}

impl ServerInfo {
    pub fn new(id: ServerId, name: &str, address: &str, port: u16) -> Self {
        Self {
            id, name: name.to_string(), address: address.to_string(), port,
            map_name: String::new(), game_mode: String::new(),
            current_players: 0, max_players: 0, ping_ms: 999,
            has_password: false, is_official: false, is_dedicated: true,
            version: String::new(), tags: Vec::new(), custom_data: HashMap::new(),
            last_queried: None, anti_cheat: false, region: String::new(), country_code: String::new(),
        }
    }
    pub fn is_full(&self) -> bool { self.current_players >= self.max_players }
    pub fn is_empty(&self) -> bool { self.current_players == 0 }
    pub fn player_ratio(&self) -> f32 { if self.max_players > 0 { self.current_players as f32 / self.max_players as f32 } else { 0.0 } }
    pub fn connection_string(&self) -> String { format!("{}:{}", self.address, self.port) }
}

// --- Filter ---
#[derive(Debug, Clone)]
pub struct ServerFilter {
    pub name_contains: Option<String>,
    pub map_name: Option<String>,
    pub game_mode: Option<String>,
    pub max_ping: Option<u32>,
    pub not_full: bool,
    pub not_empty: bool,
    pub has_password: Option<bool>,
    pub is_official: Option<bool>,
    pub min_players: Option<u32>,
    pub max_players: Option<u32>,
    pub region: Option<String>,
    pub tags_include: Vec<String>,
    pub tags_exclude: Vec<String>,
    pub anti_cheat_required: bool,
    pub version_match: Option<String>,
}

impl Default for ServerFilter {
    fn default() -> Self {
        Self {
            name_contains: None, map_name: None, game_mode: None, max_ping: None,
            not_full: false, not_empty: false, has_password: None, is_official: None,
            min_players: None, max_players: None, region: None,
            tags_include: Vec::new(), tags_exclude: Vec::new(),
            anti_cheat_required: false, version_match: None,
        }
    }
}

impl ServerFilter {
    pub fn matches(&self, server: &ServerInfo) -> bool {
        if let Some(ref name) = self.name_contains {
            if !server.name.to_lowercase().contains(&name.to_lowercase()) { return false; }
        }
        if let Some(ref map) = self.map_name { if server.map_name != *map { return false; } }
        if let Some(ref mode) = self.game_mode { if server.game_mode != *mode { return false; } }
        if let Some(max_ping) = self.max_ping { if server.ping_ms > max_ping { return false; } }
        if self.not_full && server.is_full() { return false; }
        if self.not_empty && server.is_empty() { return false; }
        if let Some(pw) = self.has_password { if server.has_password != pw { return false; } }
        if let Some(official) = self.is_official { if server.is_official != official { return false; } }
        if let Some(min) = self.min_players { if server.current_players < min { return false; } }
        if let Some(max) = self.max_players { if server.max_players > max { return false; } }
        if let Some(ref region) = self.region { if server.region != *region { return false; } }
        for tag in &self.tags_include { if !server.tags.contains(tag) { return false; } }
        for tag in &self.tags_exclude { if server.tags.contains(tag) { return false; } }
        if self.anti_cheat_required && !server.anti_cheat { return false; }
        if let Some(ref ver) = self.version_match { if server.version != *ver { return false; } }
        true
    }
}

// --- Sort ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortCriteria { Name, Ping, Players, Map, Mode }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection { Ascending, Descending }

#[derive(Debug, Clone, Copy)]
pub struct SortConfig { pub criteria: SortCriteria, pub direction: SortDirection }

impl Default for SortConfig {
    fn default() -> Self { Self { criteria: SortCriteria::Ping, direction: SortDirection::Ascending } }
}

// --- Query protocol ---
#[derive(Debug, Clone, Copy)]
pub struct ServerQuery { pub id: u64, pub sent_at: Instant, pub address: u64, pub port: u16, pub retries: u32 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryState { Idle, Querying, Done, Failed }

// --- Server browser ---
pub struct ServerBrowser {
    servers: HashMap<ServerId, ServerInfo>,
    filtered: Vec<ServerId>,
    favorites: Vec<ServerId>,
    history: Vec<(ServerId, Instant)>,
    filter: ServerFilter,
    sort: SortConfig,
    query_state: QueryState,
    last_refresh: Option<Instant>,
    refresh_interval: Duration,
    max_history: usize,
    pending_queries: Vec<ServerQuery>,
    next_id: ServerId,
    pub stats: BrowserStats,
}

#[derive(Debug, Clone, Default)]
pub struct BrowserStats {
    pub total_servers: u32,
    pub filtered_servers: u32,
    pub favorites_count: u32,
    pub avg_ping: u32,
    pub total_players: u32,
    pub last_refresh_time_ms: u64,
}

impl ServerBrowser {
    pub fn new() -> Self {
        Self {
            servers: HashMap::new(), filtered: Vec::new(),
            favorites: Vec::new(), history: Vec::new(),
            filter: ServerFilter::default(), sort: SortConfig::default(),
            query_state: QueryState::Idle, last_refresh: None,
            refresh_interval: Duration::from_secs(30), max_history: 50,
            pending_queries: Vec::new(), next_id: 1, stats: BrowserStats::default(),
        }
    }

    pub fn add_server(&mut self, mut info: ServerInfo) -> ServerId {
        let id = self.next_id; self.next_id += 1;
        info.id = id;
        info.last_queried = Some(Instant::now());
        self.servers.insert(id, info);
        self.apply_filter_and_sort();
        id
    }

    pub fn update_server(&mut self, id: ServerId, info: ServerInfo) {
        self.servers.insert(id, info);
        self.apply_filter_and_sort();
    }

    pub fn remove_server(&mut self, id: ServerId) {
        self.servers.remove(&id);
        self.filtered.retain(|&i| i != id);
    }

    pub fn get_server(&self, id: ServerId) -> Option<&ServerInfo> { self.servers.get(&id) }

    pub fn set_filter(&mut self, filter: ServerFilter) { self.filter = filter; self.apply_filter_and_sort(); }
    pub fn set_sort(&mut self, sort: SortConfig) { self.sort = sort; self.apply_filter_and_sort(); }

    pub fn filtered_servers(&self) -> Vec<&ServerInfo> {
        self.filtered.iter().filter_map(|id| self.servers.get(id)).collect()
    }

    pub fn all_servers(&self) -> Vec<&ServerInfo> { self.servers.values().collect() }

    // --- Favorites ---
    pub fn add_favorite(&mut self, id: ServerId) { if !self.favorites.contains(&id) { self.favorites.push(id); } }
    pub fn remove_favorite(&mut self, id: ServerId) { self.favorites.retain(|&f| f != id); }
    pub fn is_favorite(&self, id: ServerId) -> bool { self.favorites.contains(&id) }
    pub fn favorite_servers(&self) -> Vec<&ServerInfo> {
        self.favorites.iter().filter_map(|id| self.servers.get(id)).collect()
    }

    // --- History ---
    pub fn add_to_history(&mut self, id: ServerId) {
        self.history.retain(|(h, _)| *h != id);
        self.history.push((id, Instant::now()));
        if self.history.len() > self.max_history { self.history.remove(0); }
    }
    pub fn history_servers(&self) -> Vec<&ServerInfo> {
        self.history.iter().rev().filter_map(|(id, _)| self.servers.get(id)).collect()
    }

    // --- Refresh ---
    pub fn refresh(&mut self) {
        self.query_state = QueryState::Querying;
        self.last_refresh = Some(Instant::now());
    }

    pub fn is_refreshing(&self) -> bool { self.query_state == QueryState::Querying }

    pub fn complete_refresh(&mut self) {
        self.query_state = QueryState::Done;
        self.apply_filter_and_sort();
        self.update_stats();
    }

    pub fn needs_refresh(&self) -> bool {
        self.last_refresh.map(|t| t.elapsed() > self.refresh_interval).unwrap_or(true)
    }

    // --- Internal ---
    fn apply_filter_and_sort(&mut self) {
        self.filtered = self.servers.values()
            .filter(|s| self.filter.matches(s))
            .map(|s| s.id)
            .collect();

        self.filtered.sort_by(|&a, &b| {
            let sa = &self.servers[&a];
            let sb = &self.servers[&b];
            let ord = match self.sort.criteria {
                SortCriteria::Name => sa.name.cmp(&sb.name),
                SortCriteria::Ping => sa.ping_ms.cmp(&sb.ping_ms),
                SortCriteria::Players => sa.current_players.cmp(&sb.current_players),
                SortCriteria::Map => sa.map_name.cmp(&sb.map_name),
                SortCriteria::Mode => sa.game_mode.cmp(&sb.game_mode),
            };
            if self.sort.direction == SortDirection::Descending { ord.reverse() } else { ord }
        });
    }

    fn update_stats(&mut self) {
        let total = self.servers.len() as u32;
        let filtered = self.filtered.len() as u32;
        let total_ping: u64 = self.servers.values().map(|s| s.ping_ms as u64).sum();
        let total_players: u32 = self.servers.values().map(|s| s.current_players).sum();
        self.stats = BrowserStats {
            total_servers: total, filtered_servers: filtered,
            favorites_count: self.favorites.len() as u32,
            avg_ping: if total > 0 { (total_ping / total as u64) as u32 } else { 0 },
            total_players,
            last_refresh_time_ms: self.last_refresh.map(|t| t.elapsed().as_millis() as u64).unwrap_or(0),
        };
    }

    // --- Predefined filters ---
    pub fn filter_low_ping(&mut self, max_ping: u32) {
        self.filter.max_ping = Some(max_ping);
        self.apply_filter_and_sort();
    }

    pub fn filter_has_players(&mut self) {
        self.filter.not_empty = true;
        self.apply_filter_and_sort();
    }

    pub fn clear_filter(&mut self) { self.filter = ServerFilter::default(); self.apply_filter_and_sort(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_server(name: &str, players: u32, max: u32, ping: u32) -> ServerInfo {
        let mut s = ServerInfo::new(0, name, "127.0.0.1", 7777);
        s.current_players = players; s.max_players = max; s.ping_ms = ping;
        s.map_name = "TestMap".into(); s.game_mode = "TDM".into();
        s
    }

    #[test]
    fn test_add_and_filter() {
        let mut browser = ServerBrowser::new();
        browser.add_server(make_server("Server A", 10, 16, 30));
        browser.add_server(make_server("Server B", 0, 16, 100));
        browser.add_server(make_server("Server C", 16, 16, 50));

        browser.set_filter(ServerFilter { not_full: true, not_empty: true, ..Default::default() });
        assert_eq!(browser.filtered_servers().len(), 1);
        assert_eq!(browser.filtered_servers()[0].name, "Server A");
    }

    #[test]
    fn test_sort_by_ping() {
        let mut browser = ServerBrowser::new();
        browser.add_server(make_server("High Ping", 5, 16, 200));
        browser.add_server(make_server("Low Ping", 5, 16, 10));
        browser.add_server(make_server("Med Ping", 5, 16, 80));

        let servers = browser.filtered_servers();
        assert_eq!(servers[0].name, "Low Ping");
        assert_eq!(servers[2].name, "High Ping");
    }

    #[test]
    fn test_favorites() {
        let mut browser = ServerBrowser::new();
        let id = browser.add_server(make_server("Fav", 5, 16, 30));
        browser.add_favorite(id);
        assert!(browser.is_favorite(id));
        assert_eq!(browser.favorite_servers().len(), 1);
        browser.remove_favorite(id);
        assert!(!browser.is_favorite(id));
    }

    #[test]
    fn test_history() {
        let mut browser = ServerBrowser::new();
        let id = browser.add_server(make_server("Recent", 5, 16, 30));
        browser.add_to_history(id);
        assert_eq!(browser.history_servers().len(), 1);
    }
}
