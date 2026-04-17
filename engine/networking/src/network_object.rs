// engine/networking/src/network_object.rs
// Network objects: identity, ownership, authority transfer, RPC routing, spawn/despawn sync.
use std::collections::{HashMap, VecDeque};

pub type NetworkId = u32;
pub type ClientId = u32;
pub type RpcId = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkAuthority { Server, Client(ClientId), Shared }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkOwnership { None, Server, Client(ClientId) }

#[derive(Debug, Clone)]
pub struct NetworkIdentity { pub network_id: NetworkId, pub prefab_id: u32, pub authority: NetworkAuthority, pub ownership: NetworkOwnership, pub is_local: bool, pub spawn_frame: u64, pub scene_object: bool }

impl NetworkIdentity {
    pub fn new(network_id: NetworkId, prefab_id: u32) -> Self {
        Self { network_id, prefab_id, authority: NetworkAuthority::Server, ownership: NetworkOwnership::Server, is_local: false, spawn_frame: 0, scene_object: false }
    }
    pub fn has_authority(&self, client: ClientId) -> bool {
        match self.authority { NetworkAuthority::Server => false, NetworkAuthority::Client(c) => c == client, NetworkAuthority::Shared => true }
    }
    pub fn is_owner(&self, client: ClientId) -> bool {
        match self.ownership { NetworkOwnership::Client(c) => c == client, _ => false }
    }
}

#[derive(Debug, Clone)]
pub struct NetworkVariable { pub name: String, pub value: Vec<u8>, pub dirty: bool, pub last_sync_frame: u64, pub interpolated: bool }
impl NetworkVariable {
    pub fn new(name: &str, initial: Vec<u8>) -> Self { Self { name: name.to_string(), value: initial, dirty: true, last_sync_frame: 0, interpolated: false } }
    pub fn set(&mut self, value: Vec<u8>) { if self.value != value { self.value = value; self.dirty = true; } }
}

#[derive(Debug, Clone)]
pub struct RpcDefinition { pub id: RpcId, pub name: String, pub target: RpcTarget, pub reliable: bool }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RpcTarget { Server, AllClients, OwnerOnly, AllExceptOwner }

#[derive(Debug, Clone)]
pub struct RpcCall { pub rpc_id: RpcId, pub sender: ClientId, pub network_id: NetworkId, pub args: Vec<u8>, pub reliable: bool }

#[derive(Debug, Clone)]
pub struct SpawnMessage { pub network_id: NetworkId, pub prefab_id: u32, pub position: [f32; 3], pub rotation: [f32; 4], pub owner: ClientId, pub initial_data: Vec<u8> }

#[derive(Debug, Clone)]
pub struct DespawnMessage { pub network_id: NetworkId, pub reason: DespawnReason }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DespawnReason { Destroyed, OutOfScope, Disconnect, SceneUnload }

#[derive(Debug, Clone)]
pub enum AuthorityRequest { Request { client: ClientId, network_id: NetworkId }, Grant { client: ClientId, network_id: NetworkId }, Deny { client: ClientId, network_id: NetworkId, reason: String }, Release { client: ClientId, network_id: NetworkId } }

pub struct NetworkObjectManager {
    objects: HashMap<NetworkId, NetworkObject>,
    next_id: NetworkId,
    spawn_queue: VecDeque<SpawnMessage>,
    despawn_queue: VecDeque<DespawnMessage>,
    rpc_queue: VecDeque<RpcCall>,
    authority_requests: VecDeque<AuthorityRequest>,
    pub is_server: bool,
    pub local_client: ClientId,
    pub frame: u64,
}

pub struct NetworkObject {
    pub identity: NetworkIdentity,
    pub variables: HashMap<String, NetworkVariable>,
    pub rpcs: HashMap<RpcId, RpcDefinition>,
    pub is_spawned: bool,
}

impl NetworkObjectManager {
    pub fn new(is_server: bool, local_client: ClientId) -> Self {
        Self { objects: HashMap::new(), next_id: 1, spawn_queue: VecDeque::new(), despawn_queue: VecDeque::new(), rpc_queue: VecDeque::new(), authority_requests: VecDeque::new(), is_server, local_client, frame: 0 }
    }

    pub fn spawn(&mut self, prefab_id: u32, position: [f32; 3], rotation: [f32; 4], owner: ClientId) -> NetworkId {
        let id = self.next_id; self.next_id += 1;
        let mut identity = NetworkIdentity::new(id, prefab_id);
        identity.ownership = NetworkOwnership::Client(owner);
        identity.spawn_frame = self.frame;
        identity.is_local = owner == self.local_client;
        let obj = NetworkObject { identity, variables: HashMap::new(), rpcs: HashMap::new(), is_spawned: true };
        self.objects.insert(id, obj);
        self.spawn_queue.push_back(SpawnMessage { network_id: id, prefab_id, position, rotation, owner, initial_data: Vec::new() });
        id
    }

    pub fn despawn(&mut self, network_id: NetworkId, reason: DespawnReason) {
        if let Some(obj) = self.objects.get_mut(&network_id) { obj.is_spawned = false; }
        self.despawn_queue.push_back(DespawnMessage { network_id, reason });
    }

    pub fn get(&self, id: NetworkId) -> Option<&NetworkObject> { self.objects.get(&id) }
    pub fn get_mut(&mut self, id: NetworkId) -> Option<&mut NetworkObject> { self.objects.get_mut(&id) }

    pub fn send_rpc(&mut self, network_id: NetworkId, rpc_id: RpcId, args: Vec<u8>, reliable: bool) {
        self.rpc_queue.push_back(RpcCall { rpc_id, sender: self.local_client, network_id, args, reliable });
    }

    pub fn request_authority(&mut self, network_id: NetworkId) {
        self.authority_requests.push_back(AuthorityRequest::Request { client: self.local_client, network_id });
    }

    pub fn release_authority(&mut self, network_id: NetworkId) {
        if let Some(obj) = self.objects.get_mut(&network_id) {
            obj.identity.authority = NetworkAuthority::Server;
        }
        self.authority_requests.push_back(AuthorityRequest::Release { client: self.local_client, network_id });
    }

    pub fn grant_authority(&mut self, network_id: NetworkId, client: ClientId) {
        if let Some(obj) = self.objects.get_mut(&network_id) {
            obj.identity.authority = NetworkAuthority::Client(client);
        }
    }

    pub fn drain_spawns(&mut self) -> Vec<SpawnMessage> { self.spawn_queue.drain(..).collect() }
    pub fn drain_despawns(&mut self) -> Vec<DespawnMessage> { self.despawn_queue.drain(..).collect() }
    pub fn drain_rpcs(&mut self) -> Vec<RpcCall> { self.rpc_queue.drain(..).collect() }

    pub fn update(&mut self) { self.frame += 1; self.objects.retain(|_, obj| obj.is_spawned); }
    pub fn object_count(&self) -> usize { self.objects.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_spawn_despawn() {
        let mut mgr = NetworkObjectManager::new(true, 0);
        let id = mgr.spawn(1, [0.0; 3], [0.0, 0.0, 0.0, 1.0], 0);
        assert!(mgr.get(id).is_some());
        let spawns = mgr.drain_spawns();
        assert_eq!(spawns.len(), 1);
        mgr.despawn(id, DespawnReason::Destroyed);
        mgr.update();
        assert!(mgr.get(id).is_none());
    }
    #[test]
    fn test_authority() {
        let mut mgr = NetworkObjectManager::new(true, 0);
        let id = mgr.spawn(1, [0.0; 3], [0.0, 0.0, 0.0, 1.0], 0);
        mgr.grant_authority(id, 5);
        assert!(mgr.get(id).unwrap().identity.has_authority(5));
    }
    #[test]
    fn test_rpc() {
        let mut mgr = NetworkObjectManager::new(false, 1);
        let id = mgr.spawn(1, [0.0; 3], [0.0, 0.0, 0.0, 1.0], 1);
        mgr.send_rpc(id, 0, vec![1, 2, 3], true);
        let rpcs = mgr.drain_rpcs();
        assert_eq!(rpcs.len(), 1);
    }
}
