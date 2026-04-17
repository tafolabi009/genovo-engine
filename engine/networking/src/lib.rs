//! Genovo Engine - Networking Module
//!
//! Provides multiplayer networking infrastructure including state replication,
//! client-side prediction with rollback, pluggable network transports
//! (UDP, WebSocket), remote procedure calls, lobby management, matchmaking,
//! protocol framing, state synchronization, game session management,
//! voice chat with mu-law codec, jitter buffering and VAD,
//! network snapshot capture with delta compression, high-level netcode
//! integration for client/server architecture, HTTP client, WebSocket
//! client, and comprehensive network statistics.

pub mod encryption;
pub mod http_client;
pub mod lobby;
pub mod matchmaking;
pub mod netcode;
pub mod network_stats;
pub mod prediction;
pub mod protocol;
pub mod replication;
pub mod rpc;
pub mod session;
pub mod snapshot;
pub mod sync;
pub mod transport;
pub mod voice;
pub mod web_socket;
pub mod connection_quality;
pub mod nat_traversal;
pub mod relay_server;

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
pub use http_client::{
    HeaderMap, HttpClient, HttpClientConfig, HttpError, HttpMethod, HttpRequest,
    HttpResponse, JsonValue, ParsedUrl, PoolStats, QueryBuilder,
    base64_encode, parse_json, percent_decode, percent_encode,
};
pub use web_socket::{
    ReconnectPolicy, WebSocket, WsCloseCode, WsConfig, WsError, WsFrame,
    WsMessage, WsOpcode, WsState, WsStats, WsUrl,
};
pub use network_stats::{
    ChannelStats, GraphData, NetworkQuality, NetworkStatsCollector,
    NetworkStatsSummary, RollingSamples, RttHistogram, StatsSnapshot,
};
pub use connection_quality::{
    ConnectionQualityAssessor, ConnectionQualityConfig, PingGraph, PingGraphPoint,
    QualityIndicator, QualitySnapshot, Recommendation,
};
pub use nat_traversal::{
    CandidateType, HolePunchCoordinator, HolePunchSession, HolePunchState,
    IceCandidate, NatDetectionResult, NatType, StunAttribute, StunAttributeType,
    StunClass, StunMessage, StunMessageType, StunMethod, STUN_MAGIC_COOKIE,
};
pub use relay_server::{
    Allocation, AllocationId, AllocateResult, AuthResult, BandwidthThrottle,
    ConnectionPairing, PairingId, RelayClientId, RelayCredentials, RelayError,
    RelayMessage, RelayServer, RelayServerConfig, RelayStats, SimpleAuthenticator,
};

// Bandwidth management: per-channel bandwidth budgets, priority-based allocation,
// congestion detection, adaptive send rate, bandwidth smoothing, burst allowance.
pub mod bandwidth_manager;

// Network condition simulation: artificial latency, packet loss, jitter, bandwidth
// limit, out-of-order delivery, duplicate packets -- for testing multiplayer
// without real network issues.
pub mod network_simulation;

// Efficient packet serialization: bit-level packing, variable-length integers,
// quantized floats, string table (send index instead of full string), schema versioning.
pub mod packet_serializer;

// Server browser: server list with ping/player count/map/mode, server query
// protocol, favorites, history, password servers, server filtering/sorting,
// refresh mechanism.
pub mod server_browser;

// In-game chat: text channels (all/team/whisper), message history, chat
// commands (/team, /all, /whisper), message filtering (profanity filter stub),
// chat UI data, chat events.
pub mod chat_system;
