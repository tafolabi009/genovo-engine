//! Genovo Engine - Networking Module
//!
//! Provides multiplayer networking infrastructure including state replication,
//! client-side prediction with rollback, pluggable network transports
//! (UDP, WebSocket), remote procedure calls, lobby management, matchmaking,
//! protocol framing, state synchronization, and game session management.

pub mod lobby;
pub mod matchmaking;
pub mod prediction;
pub mod protocol;
pub mod replication;
pub mod rpc;
pub mod session;
pub mod sync;
pub mod transport;

pub use prediction::{
    EntityHitbox, Hit, InputBuffer, InterpolationBuffer, LagCompensation,
    PredictionManager, ReconciliationResult, StateSnapshot, TickState, TimestampedInput,
};
pub use replication::{
    Authority, ComponentSnapshot, EntityUpdate, NetworkId, Replicated,
    ReplicationAction, ReplicationChannel, ReplicationFrame, ReplicationManager,
    ReplicationPolicy, ReplicationPriority,
};
pub use transport::{
    Connection, ConnectionState, DeliveryMode, Packet, PacketHeader, PacketType,
    ReliabilityLayer, Transport, UdpTransport, WebSocketTransport,
};
pub use rpc::{
    FnRpcHandler, RpcBatch, RpcCall, RpcDispatchResult, RpcHandler, RpcManager,
    RpcReliability, RpcStats, RpcTarget, RateLimiter,
};
pub use lobby::{
    ChatMessage, Lobby, LobbyEvent, LobbyId, LobbyInfo, LobbyManager,
    LobbyPlayer, LobbySettings, LobbyState, PlayerId,
};
pub use matchmaking::{
    EloMatchmaker, EloRating, Match, MatchPlayerId, MatchPreferences,
    MatchmakingAlgorithm, MatchmakingQueue, PlayerProfile, QueueMatchmaker,
};
pub use protocol::{
    Channel, ChannelConfig, ChannelOrdering, ChannelReliability, Compressor,
    Encryption, Fragment, Fragmenter, MessageRegistry, NetMessage, PacketStats,
    Protocol, ReassemblyBuffer,
};
pub use sync::{
    AOIEntity, AOIGrid, DeltaCompression, InterestManagement, InterestUpdate,
    InterpolationSample, SnapshotInterpolation, StatePriority, SyncGroup,
    SyncGroupManager,
};
pub use session::{
    GameSession, LeaveReason, PlayerSession, PlayerState, SessionConfig,
    SessionEvent, SessionState,
};
