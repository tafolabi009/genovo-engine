//! Genovo Engine - Networking Module
//!
//! Provides multiplayer networking infrastructure including state replication,
//! client-side prediction with rollback, pluggable network transports
//! (UDP, WebSocket), remote procedure calls, lobby management, matchmaking,
//! protocol framing, state synchronization, game session management,
//! voice chat with mu-law codec, jitter buffering and VAD,
//! network snapshot capture with delta compression, and high-level netcode
//! integration for client/server architecture.

pub mod encryption;
pub mod lobby;
pub mod matchmaking;
pub mod netcode;
pub mod prediction;
pub mod protocol;
pub mod replication;
pub mod rpc;
pub mod session;
pub mod snapshot;
pub mod sync;
pub mod transport;
pub mod voice;

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
pub use netcode::{
    ClientConnection, ClientEvent, ClientId, ClientInput, DisconnectReason,
    InputBufferStats, NetInputBuffer, NetcodeClient, NetcodeServer, ServerEvent,
    TickSystem,
};
pub use protocol::{
    Channel, ChannelConfig, ChannelOrdering, ChannelReliability, Compressor,
    Encryption, Fragment, Fragmenter, MessageRegistry, NetMessage, PacketStats,
    Protocol, ReassemblyBuffer,
};
pub use snapshot::{
    ClientSnapshot, EntityState, ServerSnapshot, SnapshotBuffer, SnapshotCompressor,
    SnapshotConfig, SnapshotSystem, WorldSnapshot,
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
pub use voice::{
    ComfortNoiseGenerator, JitterBuffer, JitterBufferStats,
    PacketLossConcealer, SimpleCodec, VoiceActivityDetector, VoiceCapture,
    VoiceChannel, VoiceChannelType, VoiceChatError, VoiceChatManager, VoiceCodec,
    VoicePacket, VoicePlayerId, VoiceQuality, VoiceReceiver, VoiceTransmitter,
    mu_law_decode, mu_law_encode, mu_law_decode_block, mu_law_encode_block,
};
pub use encryption::{
    EncryptionError, EncryptionResult, HandshakeState, KeyExchange, PacketEncryptor,
    PacketIntegrity, PrivateKey, PublicKey, ReplayWindow, SecureChannel, SharedSecret,
    U256, modpow_u256,
};
