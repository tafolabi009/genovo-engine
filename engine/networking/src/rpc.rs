//! Remote Procedure Calls (RPC) system.
//!
//! Provides a framework for registering and invoking functions across the
//! network. Supports multiple targeting modes, reliability levels, batching,
//! rate limiting, and request/response patterns.
//!
//! ## Wire format
//!
//! ```text
//! RPC Packet:
//!   [u16 batch_count]
//!   for each RPC in batch:
//!     [u16 rpc_id]
//!     [u8  flags]          // bit 0: has_response_id, bit 1: is_response
//!     [u32 response_id]    // only if has_response_id
//!     [u16 payload_len]
//!     [... payload bytes]
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use genovo_core::{EngineError, EngineResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum RPC payload size in bytes.
pub const MAX_RPC_PAYLOAD_SIZE: usize = 4096;

/// Maximum RPCs per batch frame.
pub const MAX_RPCS_PER_BATCH: usize = 64;

/// Default rate limit: maximum RPCs per second per client.
pub const DEFAULT_RATE_LIMIT: u32 = 120;

/// Default rate limit window duration.
pub const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(1);

/// Maximum pending response futures.
pub const MAX_PENDING_RESPONSES: usize = 256;

/// Timeout for RPC responses.
pub const RPC_RESPONSE_TIMEOUT: Duration = Duration::from_secs(10);

/// Flag: this RPC frame carries a response ID for request/response.
const FLAG_HAS_RESPONSE_ID: u8 = 1 << 0;

/// Flag: this RPC frame IS a response (not a request).
const FLAG_IS_RESPONSE: u8 = 1 << 1;

// ---------------------------------------------------------------------------
// RpcTarget
// ---------------------------------------------------------------------------

/// Specifies which peers should receive an RPC invocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RpcTarget {
    /// Send to the server (client -> server).
    Server,
    /// Send to a specific client (server -> client).
    Client(u32),
    /// Send to all connected clients (server -> all clients).
    AllClients,
    /// Send to all clients except the specified one.
    AllClientsExcept(u32),
    /// Broadcast to all peers (including server, if applicable).
    Broadcast,
}

impl RpcTarget {
    /// Encode the target to bytes.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            RpcTarget::Server => vec![0],
            RpcTarget::Client(id) => {
                let mut buf = vec![1];
                buf.extend_from_slice(&id.to_be_bytes());
                buf
            }
            RpcTarget::AllClients => vec![2],
            RpcTarget::AllClientsExcept(id) => {
                let mut buf = vec![3];
                buf.extend_from_slice(&id.to_be_bytes());
                buf
            }
            RpcTarget::Broadcast => vec![4],
        }
    }

    /// Decode a target from bytes. Returns the target and bytes consumed.
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        if data.is_empty() {
            return None;
        }
        match data[0] {
            0 => Some((RpcTarget::Server, 1)),
            1 => {
                if data.len() < 5 {
                    return None;
                }
                let id = u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
                Some((RpcTarget::Client(id), 5))
            }
            2 => Some((RpcTarget::AllClients, 1)),
            3 => {
                if data.len() < 5 {
                    return None;
                }
                let id = u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
                Some((RpcTarget::AllClientsExcept(id), 5))
            }
            4 => Some((RpcTarget::Broadcast, 1)),
            _ => None,
        }
    }

    /// Returns true if the given client_id is a valid recipient for this target.
    pub fn includes_client(&self, client_id: u32) -> bool {
        match self {
            RpcTarget::Server => false,
            RpcTarget::Client(id) => *id == client_id,
            RpcTarget::AllClients => true,
            RpcTarget::AllClientsExcept(id) => *id != client_id,
            RpcTarget::Broadcast => true,
        }
    }

    /// Returns true if this target includes the server.
    pub fn includes_server(&self) -> bool {
        matches!(self, RpcTarget::Server | RpcTarget::Broadcast)
    }
}

// ---------------------------------------------------------------------------
// RpcReliability
// ---------------------------------------------------------------------------

/// Delivery guarantee for an RPC invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RpcReliability {
    /// Fire-and-forget. May be lost.
    Unreliable,
    /// Guaranteed delivery, but may arrive out of order.
    Reliable,
    /// Guaranteed delivery and in-order processing.
    ReliableOrdered,
}

impl RpcReliability {
    /// Encode to a single byte.
    pub fn as_u8(self) -> u8 {
        match self {
            RpcReliability::Unreliable => 0,
            RpcReliability::Reliable => 1,
            RpcReliability::ReliableOrdered => 2,
        }
    }

    /// Decode from a single byte.
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(RpcReliability::Unreliable),
            1 => Some(RpcReliability::Reliable),
            2 => Some(RpcReliability::ReliableOrdered),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// RpcHandler trait
// ---------------------------------------------------------------------------

/// Trait for RPC handler functions.
///
/// Implementations receive the raw argument bytes and the caller's context,
/// execute the procedure, and optionally return a response payload.
pub trait RpcHandler: Send + Sync {
    /// Handle an incoming RPC call.
    ///
    /// `caller_id` is the peer who invoked the RPC (0 for server).
    /// `args` contains the serialized arguments.
    ///
    /// Returns an optional response payload.
    fn handle(&self, caller_id: u32, args: &[u8]) -> EngineResult<Option<Vec<u8>>>;

    /// Returns a human-readable name for this handler (for debugging).
    fn name(&self) -> &str;
}

/// A simple function-pointer based RPC handler.
pub struct FnRpcHandler {
    name: String,
    func: Box<dyn Fn(u32, &[u8]) -> EngineResult<Option<Vec<u8>>> + Send + Sync>,
}

impl FnRpcHandler {
    /// Create a new function-pointer RPC handler.
    pub fn new<F>(name: impl Into<String>, func: F) -> Self
    where
        F: Fn(u32, &[u8]) -> EngineResult<Option<Vec<u8>>> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            func: Box::new(func),
        }
    }
}

impl RpcHandler for FnRpcHandler {
    fn handle(&self, caller_id: u32, args: &[u8]) -> EngineResult<Option<Vec<u8>>> {
        (self.func)(caller_id, args)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// RpcCall
// ---------------------------------------------------------------------------

/// A single RPC invocation waiting to be sent.
#[derive(Debug, Clone)]
pub struct RpcCall {
    /// The function ID (mapped from name).
    pub rpc_id: u16,
    /// The target peer(s).
    pub target: RpcTarget,
    /// Serialized argument bytes.
    pub args: Vec<u8>,
    /// Delivery reliability.
    pub reliability: RpcReliability,
    /// Optional response ID (for request/response pattern).
    pub response_id: Option<u32>,
    /// Whether this is a response to a previous call.
    pub is_response: bool,
    /// When this call was created.
    pub created_at: Instant,
}

impl RpcCall {
    /// Create a new RPC call.
    pub fn new(rpc_id: u16, target: RpcTarget, args: Vec<u8>, reliability: RpcReliability) -> Self {
        Self {
            rpc_id,
            target,
            args,
            reliability,
            response_id: None,
            is_response: false,
            created_at: Instant::now(),
        }
    }

    /// Create an RPC call that expects a response.
    pub fn with_response(
        rpc_id: u16,
        target: RpcTarget,
        args: Vec<u8>,
        reliability: RpcReliability,
        response_id: u32,
    ) -> Self {
        Self {
            rpc_id,
            target,
            args,
            reliability,
            response_id: Some(response_id),
            is_response: false,
            created_at: Instant::now(),
        }
    }

    /// Create a response RPC.
    pub fn response(rpc_id: u16, target: RpcTarget, args: Vec<u8>, response_id: u32) -> Self {
        Self {
            rpc_id,
            target,
            args,
            reliability: RpcReliability::Reliable,
            response_id: Some(response_id),
            is_response: true,
            created_at: Instant::now(),
        }
    }

    /// Encode a single RPC call to bytes for wire transmission.
    pub fn encode_payload(&self) -> Vec<u8> {
        let mut flags: u8 = 0;
        if self.response_id.is_some() {
            flags |= FLAG_HAS_RESPONSE_ID;
        }
        if self.is_response {
            flags |= FLAG_IS_RESPONSE;
        }

        let mut buf = Vec::with_capacity(5 + 4 + self.args.len());
        buf.extend_from_slice(&self.rpc_id.to_be_bytes());
        buf.push(flags);

        if let Some(rid) = self.response_id {
            buf.extend_from_slice(&rid.to_be_bytes());
        }

        buf.extend_from_slice(&(self.args.len() as u16).to_be_bytes());
        buf.extend_from_slice(&self.args);
        buf
    }

    /// Decode a single RPC call from wire bytes.
    /// Returns the call fields and bytes consumed.
    pub fn decode_payload(data: &[u8]) -> Option<(u16, u8, Option<u32>, Vec<u8>, usize)> {
        if data.len() < 3 {
            return None;
        }

        let rpc_id = u16::from_be_bytes([data[0], data[1]]);
        let flags = data[2];
        let mut offset = 3;

        let response_id = if flags & FLAG_HAS_RESPONSE_ID != 0 {
            if data.len() < offset + 4 {
                return None;
            }
            let rid = u32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;
            Some(rid)
        } else {
            None
        };

        if data.len() < offset + 2 {
            return None;
        }
        let payload_len =
            u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        if data.len() < offset + payload_len {
            return None;
        }
        let args = data[offset..offset + payload_len].to_vec();
        offset += payload_len;

        Some((rpc_id, flags, response_id, args, offset))
    }
}

// ---------------------------------------------------------------------------
// RpcBatch
// ---------------------------------------------------------------------------

/// A batch of RPC calls packed into a single network packet.
#[derive(Debug, Clone)]
pub struct RpcBatch {
    /// The RPC calls in this batch.
    pub calls: Vec<RpcCall>,
}

impl RpcBatch {
    /// Create a new empty batch.
    pub fn new() -> Self {
        Self { calls: Vec::new() }
    }

    /// Add an RPC call to the batch.
    pub fn add(&mut self, call: RpcCall) -> bool {
        if self.calls.len() >= MAX_RPCS_PER_BATCH {
            return false;
        }
        self.calls.push(call);
        true
    }

    /// Returns the number of calls in the batch.
    pub fn len(&self) -> usize {
        self.calls.len()
    }

    /// Returns true if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Encode the entire batch to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.calls.len() as u16).to_be_bytes());
        for call in &self.calls {
            let payload = call.encode_payload();
            buf.extend_from_slice(&payload);
        }
        buf
    }

    /// Decode a batch from bytes.
    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        let count = u16::from_be_bytes([data[0], data[1]]) as usize;
        let mut offset = 2;
        let mut calls = Vec::with_capacity(count);

        for _ in 0..count {
            let (rpc_id, flags, response_id, args, consumed) =
                RpcCall::decode_payload(&data[offset..])?;
            offset += consumed;

            let is_response = flags & FLAG_IS_RESPONSE != 0;

            calls.push(RpcCall {
                rpc_id,
                target: RpcTarget::Server, // target is a local concern, not on wire
                args,
                reliability: RpcReliability::Reliable,
                response_id,
                is_response,
                created_at: Instant::now(),
            });
        }

        Some(Self { calls })
    }
}

impl Default for RpcBatch {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RpcRegistration
// ---------------------------------------------------------------------------

/// Metadata for a registered RPC function.
struct RpcRegistration {
    /// Unique numeric ID for wire encoding.
    id: u16,
    /// Human-readable name.
    name: String,
    /// The handler implementation.
    handler: Box<dyn RpcHandler>,
    /// Default reliability for this RPC.
    default_reliability: RpcReliability,
}

// ---------------------------------------------------------------------------
// RateLimiter
// ---------------------------------------------------------------------------

/// Per-client rate limiter tracking RPC call frequency.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    /// Maximum calls allowed per window.
    max_calls: u32,
    /// Window duration.
    window: Duration,
    /// Timestamps of recent calls within the current window.
    call_times: VecDeque<Instant>,
    /// Total calls rejected due to rate limiting.
    rejected_count: u64,
}

impl RateLimiter {
    /// Create a new rate limiter.
    pub fn new(max_calls: u32, window: Duration) -> Self {
        Self {
            max_calls,
            window,
            call_times: VecDeque::with_capacity(max_calls as usize),
            rejected_count: 0,
        }
    }

    /// Check if a call is allowed and record it if so.
    /// Returns `true` if the call is allowed, `false` if rate-limited.
    pub fn check_and_record(&mut self) -> bool {
        let now = Instant::now();
        self.prune(now);

        if self.call_times.len() as u32 >= self.max_calls {
            self.rejected_count += 1;
            return false;
        }

        self.call_times.push_back(now);
        true
    }

    /// Check if a call would be allowed without recording it.
    pub fn would_allow(&self) -> bool {
        let now = Instant::now();
        let cutoff = now - self.window;
        let active = self.call_times.iter().filter(|&&t| t > cutoff).count() as u32;
        active < self.max_calls
    }

    /// Prune expired timestamps from the window.
    fn prune(&mut self, now: Instant) {
        let cutoff = now - self.window;
        while self
            .call_times
            .front()
            .is_some_and(|&t| t <= cutoff)
        {
            self.call_times.pop_front();
        }
    }

    /// Returns the number of calls remaining in the current window.
    pub fn remaining(&self) -> u32 {
        let now = Instant::now();
        let cutoff = now - self.window;
        let active = self.call_times.iter().filter(|&&t| t > cutoff).count() as u32;
        self.max_calls.saturating_sub(active)
    }

    /// Returns total rejected calls.
    pub fn rejected_count(&self) -> u64 {
        self.rejected_count
    }

    /// Reset the rate limiter state.
    pub fn reset(&mut self) {
        self.call_times.clear();
        self.rejected_count = 0;
    }
}

// ---------------------------------------------------------------------------
// PendingResponse
// ---------------------------------------------------------------------------

/// A pending response for an RPC call that expects a return value.
#[derive(Debug)]
pub struct PendingResponse {
    /// The response ID (matches the request).
    pub response_id: u32,
    /// When the request was sent.
    pub sent_at: Instant,
    /// The response data, once received.
    pub result: Option<Vec<u8>>,
    /// Whether this response has timed out.
    pub timed_out: bool,
}

impl PendingResponse {
    /// Create a new pending response.
    pub fn new(response_id: u32) -> Self {
        Self {
            response_id,
            sent_at: Instant::now(),
            result: None,
            timed_out: false,
        }
    }

    /// Check if this response has timed out.
    pub fn is_timed_out(&self) -> bool {
        self.timed_out || Instant::now().duration_since(self.sent_at) > RPC_RESPONSE_TIMEOUT
    }

    /// Returns true if the response has been received.
    pub fn is_complete(&self) -> bool {
        self.result.is_some()
    }

    /// Set the response data.
    pub fn complete(&mut self, data: Vec<u8>) {
        self.result = Some(data);
    }
}

// ---------------------------------------------------------------------------
// RpcDispatchResult
// ---------------------------------------------------------------------------

/// Result of dispatching an incoming RPC.
#[derive(Debug)]
pub enum RpcDispatchResult {
    /// RPC was handled successfully, with optional response data.
    Ok(Option<Vec<u8>>),
    /// RPC handler not found for the given ID.
    HandlerNotFound(u16),
    /// RPC was rate-limited.
    RateLimited(u32),
    /// Handler returned an error.
    Error(String),
}

// ---------------------------------------------------------------------------
// RpcStats
// ---------------------------------------------------------------------------

/// Statistics for the RPC system.
#[derive(Debug, Clone, Default)]
pub struct RpcStats {
    /// Total RPCs sent.
    pub rpcs_sent: u64,
    /// Total RPCs received.
    pub rpcs_received: u64,
    /// Total RPCs dispatched successfully.
    pub rpcs_dispatched: u64,
    /// Total RPCs failed (handler error).
    pub rpcs_failed: u64,
    /// Total RPCs rate-limited.
    pub rpcs_rate_limited: u64,
    /// Total RPC batches sent.
    pub batches_sent: u64,
    /// Total RPC batches received.
    pub batches_received: u64,
    /// Total bytes sent for RPCs.
    pub bytes_sent: u64,
    /// Total bytes received for RPCs.
    pub bytes_received: u64,
    /// Total request/response pairs completed.
    pub responses_completed: u64,
    /// Total request/response pairs timed out.
    pub responses_timed_out: u64,
}

// ---------------------------------------------------------------------------
// RpcManager
// ---------------------------------------------------------------------------

/// Central manager for the RPC system.
///
/// Handles registration of RPC functions, dispatching incoming calls,
/// batching outgoing calls, rate limiting, and the request/response pattern.
pub struct RpcManager {
    /// Registered RPC handlers, keyed by numeric ID.
    handlers: HashMap<u16, RpcRegistration>,
    /// Name to ID mapping for quick lookup.
    name_to_id: HashMap<String, u16>,
    /// Next RPC ID to assign.
    next_rpc_id: u16,
    /// Outgoing RPC queue, partitioned by target.
    outgoing_queue: VecDeque<RpcCall>,
    /// Per-client rate limiters.
    rate_limiters: HashMap<u32, RateLimiter>,
    /// Default rate limit for new clients.
    default_rate_limit: u32,
    /// Pending responses waiting for data.
    pending_responses: HashMap<u32, PendingResponse>,
    /// Next response ID to assign.
    next_response_id: u32,
    /// Statistics.
    stats: RpcStats,
    /// Maximum outgoing queue size before flushing.
    max_queue_size: usize,
    /// Received RPC calls waiting to be dispatched.
    incoming_queue: VecDeque<(u32, RpcCall)>,
}

impl RpcManager {
    /// Creates a new RPC manager.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            name_to_id: HashMap::new(),
            next_rpc_id: 1, // 0 is reserved
            outgoing_queue: VecDeque::new(),
            rate_limiters: HashMap::new(),
            default_rate_limit: DEFAULT_RATE_LIMIT,
            pending_responses: HashMap::new(),
            next_response_id: 1,
            stats: RpcStats::default(),
            max_queue_size: MAX_RPCS_PER_BATCH * 4,
            incoming_queue: VecDeque::new(),
        }
    }

    /// Register an RPC handler by name.
    ///
    /// Returns the assigned RPC ID. If a handler with this name already exists,
    /// it is replaced and the existing ID is reused.
    pub fn register_rpc(
        &mut self,
        name: &str,
        handler: Box<dyn RpcHandler>,
    ) -> u16 {
        self.register_rpc_with_reliability(name, handler, RpcReliability::Reliable)
    }

    /// Register an RPC handler with a specific default reliability.
    pub fn register_rpc_with_reliability(
        &mut self,
        name: &str,
        handler: Box<dyn RpcHandler>,
        default_reliability: RpcReliability,
    ) -> u16 {
        // Check if already registered.
        if let Some(&existing_id) = self.name_to_id.get(name) {
            self.handlers.insert(
                existing_id,
                RpcRegistration {
                    id: existing_id,
                    name: name.to_string(),
                    handler,
                    default_reliability,
                },
            );
            log::debug!("RPC '{}' re-registered with ID {}", name, existing_id);
            return existing_id;
        }

        let id = self.next_rpc_id;
        self.next_rpc_id += 1;

        self.name_to_id.insert(name.to_string(), id);
        self.handlers.insert(
            id,
            RpcRegistration {
                id,
                name: name.to_string(),
                handler,
                default_reliability,
            },
        );

        log::debug!("RPC '{}' registered with ID {}", name, id);
        id
    }

    /// Unregister an RPC handler by name.
    pub fn unregister_rpc(&mut self, name: &str) -> bool {
        if let Some(id) = self.name_to_id.remove(name) {
            self.handlers.remove(&id);
            true
        } else {
            false
        }
    }

    /// Returns the RPC ID for a given name, if registered.
    pub fn get_rpc_id(&self, name: &str) -> Option<u16> {
        self.name_to_id.get(name).copied()
    }

    /// Returns the RPC name for a given ID, if registered.
    pub fn get_rpc_name(&self, id: u16) -> Option<&str> {
        self.handlers.get(&id).map(|r| r.name.as_str())
    }

    /// Returns the number of registered RPCs.
    pub fn registered_count(&self) -> usize {
        self.handlers.len()
    }

    /// Queue an RPC call for sending.
    ///
    /// The call will be batched and sent during the next `flush()`.
    pub fn call_rpc(&mut self, target: RpcTarget, name: &str, args: &[u8]) -> EngineResult<()> {
        let rpc_id = self.name_to_id.get(name).copied().ok_or_else(|| {
            EngineError::NotFound(format!("RPC '{}' not registered", name))
        })?;

        let reliability = self
            .handlers
            .get(&rpc_id)
            .map(|r| r.default_reliability)
            .unwrap_or(RpcReliability::Reliable);

        let call = RpcCall::new(rpc_id, target, args.to_vec(), reliability);
        self.outgoing_queue.push_back(call);
        self.stats.rpcs_sent += 1;

        Ok(())
    }

    /// Queue an RPC call with a specific reliability level.
    pub fn call_rpc_with_reliability(
        &mut self,
        target: RpcTarget,
        name: &str,
        args: &[u8],
        reliability: RpcReliability,
    ) -> EngineResult<()> {
        let rpc_id = self.name_to_id.get(name).copied().ok_or_else(|| {
            EngineError::NotFound(format!("RPC '{}' not registered", name))
        })?;

        let call = RpcCall::new(rpc_id, target, args.to_vec(), reliability);
        self.outgoing_queue.push_back(call);
        self.stats.rpcs_sent += 1;

        Ok(())
    }

    /// Queue an RPC call that expects a response.
    ///
    /// Returns the response_id which can be used to poll for the result.
    pub fn call_rpc_with_response(
        &mut self,
        target: RpcTarget,
        name: &str,
        args: &[u8],
    ) -> EngineResult<u32> {
        let rpc_id = self.name_to_id.get(name).copied().ok_or_else(|| {
            EngineError::NotFound(format!("RPC '{}' not registered", name))
        })?;

        let response_id = self.next_response_id;
        self.next_response_id += 1;

        let call = RpcCall::with_response(
            rpc_id,
            target,
            args.to_vec(),
            RpcReliability::Reliable,
            response_id,
        );
        self.outgoing_queue.push_back(call);
        self.stats.rpcs_sent += 1;

        // Create the pending response slot.
        self.pending_responses
            .insert(response_id, PendingResponse::new(response_id));

        Ok(response_id)
    }

    /// Poll for a response to a previously sent RPC.
    ///
    /// Returns `Some(data)` if the response has arrived, `None` if still pending.
    /// Returns an error if the response has timed out.
    pub fn poll_response(&mut self, response_id: u32) -> EngineResult<Option<Vec<u8>>> {
        let pending = self.pending_responses.get_mut(&response_id).ok_or_else(|| {
            EngineError::NotFound(format!(
                "no pending response with ID {}",
                response_id
            ))
        })?;

        if pending.is_timed_out() {
            pending.timed_out = true;
            self.stats.responses_timed_out += 1;
            let _ = self.pending_responses.remove(&response_id);
            return Err(EngineError::Timeout(RPC_RESPONSE_TIMEOUT));
        }

        if let Some(data) = pending.result.take() {
            self.stats.responses_completed += 1;
            let _ = self.pending_responses.remove(&response_id);
            return Ok(Some(data));
        }

        Ok(None)
    }

    /// Flush the outgoing queue and produce batched RPC packets.
    ///
    /// Returns a list of `(RpcTarget, encoded_batch)` pairs ready to be sent
    /// over the network transport.
    pub fn flush(&mut self) -> Vec<(RpcTarget, Vec<u8>)> {
        if self.outgoing_queue.is_empty() {
            return Vec::new();
        }

        // Group calls by target.
        let mut by_target: HashMap<String, (RpcTarget, Vec<RpcCall>)> = HashMap::new();
        while let Some(call) = self.outgoing_queue.pop_front() {
            let key = format!("{:?}", call.target);
            let entry = by_target
                .entry(key)
                .or_insert_with(|| (call.target.clone(), Vec::new()));
            entry.1.push(call);
        }

        let mut results = Vec::new();
        for (_key, (target, calls)) in by_target {
            // Split into batches of MAX_RPCS_PER_BATCH.
            for chunk in calls.chunks(MAX_RPCS_PER_BATCH) {
                let mut batch = RpcBatch::new();
                for call in chunk {
                    batch.add(call.clone());
                }
                let encoded = batch.encode();
                self.stats.bytes_sent += encoded.len() as u64;
                self.stats.batches_sent += 1;
                results.push((target.clone(), encoded));
            }
        }

        results
    }

    /// Receive an encoded RPC batch from the network.
    ///
    /// Decodes and queues the calls for dispatch. `sender_id` identifies the
    /// peer who sent this batch (0 for server).
    pub fn receive_batch(&mut self, sender_id: u32, data: &[u8]) -> EngineResult<usize> {
        let batch = RpcBatch::decode(data).ok_or_else(|| {
            EngineError::InvalidArgument("failed to decode RPC batch".into())
        })?;

        self.stats.bytes_received += data.len() as u64;
        self.stats.batches_received += 1;

        let count = batch.calls.len();
        for call in batch.calls {
            self.stats.rpcs_received += 1;
            self.incoming_queue.push_back((sender_id, call));
        }

        Ok(count)
    }

    /// Dispatch all queued incoming RPCs through their registered handlers.
    ///
    /// Returns a list of dispatch results and any response RPCs that need sending.
    pub fn dispatch_incoming(&mut self) -> Vec<RpcDispatchResult> {
        let mut results = Vec::new();
        let mut response_calls = Vec::new();

        while let Some((sender_id, call)) = self.incoming_queue.pop_front() {
            // Check if this is a response to a pending request.
            if call.is_response {
                if let Some(rid) = call.response_id {
                    if let Some(pending) = self.pending_responses.get_mut(&rid) {
                        pending.complete(call.args.clone());
                    }
                }
                results.push(RpcDispatchResult::Ok(None));
                continue;
            }

            // Rate limit check.
            let limiter = self
                .rate_limiters
                .entry(sender_id)
                .or_insert_with(|| RateLimiter::new(self.default_rate_limit, RATE_LIMIT_WINDOW));

            if !limiter.check_and_record() {
                self.stats.rpcs_rate_limited += 1;
                results.push(RpcDispatchResult::RateLimited(sender_id));
                continue;
            }

            // Find the handler.
            let handler_result = if let Some(reg) = self.handlers.get(&call.rpc_id) {
                match reg.handler.handle(sender_id, &call.args) {
                    Ok(response_data) => {
                        self.stats.rpcs_dispatched += 1;

                        // If the call expects a response, queue one.
                        if let Some(rid) = call.response_id {
                            if let Some(data) = &response_data {
                                response_calls.push(RpcCall::response(
                                    call.rpc_id,
                                    RpcTarget::Client(sender_id),
                                    data.clone(),
                                    rid,
                                ));
                            }
                        }

                        RpcDispatchResult::Ok(response_data)
                    }
                    Err(e) => {
                        self.stats.rpcs_failed += 1;
                        RpcDispatchResult::Error(e.to_string())
                    }
                }
            } else {
                RpcDispatchResult::HandlerNotFound(call.rpc_id)
            };

            results.push(handler_result);
        }

        // Queue response RPCs for sending.
        for resp in response_calls {
            self.outgoing_queue.push_back(resp);
        }

        results
    }

    /// Set the rate limit for a specific client.
    pub fn set_client_rate_limit(&mut self, client_id: u32, max_calls: u32) {
        self.rate_limiters
            .insert(client_id, RateLimiter::new(max_calls, RATE_LIMIT_WINDOW));
    }

    /// Set the default rate limit for new clients.
    pub fn set_default_rate_limit(&mut self, max_calls: u32) {
        self.default_rate_limit = max_calls;
    }

    /// Remove rate limiting state for a disconnected client.
    pub fn remove_client(&mut self, client_id: u32) {
        self.rate_limiters.remove(&client_id);
    }

    /// Clean up timed-out pending responses.
    pub fn cleanup_timeouts(&mut self) {
        let timed_out: Vec<u32> = self
            .pending_responses
            .iter()
            .filter(|(_, p)| p.is_timed_out())
            .map(|(&id, _)| id)
            .collect();

        for id in timed_out {
            self.pending_responses.remove(&id);
            self.stats.responses_timed_out += 1;
        }
    }

    /// Returns the current RPC statistics.
    pub fn stats(&self) -> &RpcStats {
        &self.stats
    }

    /// Reset statistics counters.
    pub fn reset_stats(&mut self) {
        self.stats = RpcStats::default();
    }

    /// Returns the number of pending outgoing RPCs.
    pub fn outgoing_queue_len(&self) -> usize {
        self.outgoing_queue.len()
    }

    /// Returns the number of pending incoming RPCs.
    pub fn incoming_queue_len(&self) -> usize {
        self.incoming_queue.len()
    }

    /// Returns the number of pending responses.
    pub fn pending_response_count(&self) -> usize {
        self.pending_responses.len()
    }
}

impl Default for RpcManager {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // RpcTarget
    // -----------------------------------------------------------------------

    #[test]
    fn test_rpc_target_encode_decode() {
        let targets = vec![
            RpcTarget::Server,
            RpcTarget::Client(42),
            RpcTarget::AllClients,
            RpcTarget::AllClientsExcept(7),
            RpcTarget::Broadcast,
        ];

        for target in targets {
            let encoded = target.encode();
            let (decoded, _consumed) = RpcTarget::decode(&encoded).unwrap();
            assert_eq!(decoded, target);
        }
    }

    #[test]
    fn test_rpc_target_includes_client() {
        assert!(!RpcTarget::Server.includes_client(1));
        assert!(RpcTarget::Client(1).includes_client(1));
        assert!(!RpcTarget::Client(2).includes_client(1));
        assert!(RpcTarget::AllClients.includes_client(1));
        assert!(RpcTarget::AllClientsExcept(2).includes_client(1));
        assert!(!RpcTarget::AllClientsExcept(1).includes_client(1));
        assert!(RpcTarget::Broadcast.includes_client(1));
    }

    #[test]
    fn test_rpc_target_includes_server() {
        assert!(RpcTarget::Server.includes_server());
        assert!(!RpcTarget::Client(1).includes_server());
        assert!(!RpcTarget::AllClients.includes_server());
        assert!(!RpcTarget::AllClientsExcept(1).includes_server());
        assert!(RpcTarget::Broadcast.includes_server());
    }

    // -----------------------------------------------------------------------
    // RpcReliability
    // -----------------------------------------------------------------------

    #[test]
    fn test_rpc_reliability_roundtrip() {
        let reliabilities = [
            RpcReliability::Unreliable,
            RpcReliability::Reliable,
            RpcReliability::ReliableOrdered,
        ];
        for r in reliabilities {
            assert_eq!(RpcReliability::from_u8(r.as_u8()), Some(r));
        }
        assert!(RpcReliability::from_u8(99).is_none());
    }

    // -----------------------------------------------------------------------
    // RpcCall encoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_rpc_call_encode_decode_simple() {
        let call = RpcCall::new(42, RpcTarget::Server, vec![1, 2, 3], RpcReliability::Reliable);
        let encoded = call.encode_payload();
        let (rpc_id, flags, response_id, args, consumed) =
            RpcCall::decode_payload(&encoded).unwrap();

        assert_eq!(rpc_id, 42);
        assert_eq!(flags, 0);
        assert!(response_id.is_none());
        assert_eq!(args, vec![1, 2, 3]);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn test_rpc_call_encode_decode_with_response() {
        let call = RpcCall::with_response(
            10,
            RpcTarget::Client(5),
            vec![10, 20],
            RpcReliability::Reliable,
            99,
        );
        let encoded = call.encode_payload();
        let (rpc_id, flags, response_id, args, _consumed) =
            RpcCall::decode_payload(&encoded).unwrap();

        assert_eq!(rpc_id, 10);
        assert_eq!(flags & FLAG_HAS_RESPONSE_ID, FLAG_HAS_RESPONSE_ID);
        assert_eq!(flags & FLAG_IS_RESPONSE, 0);
        assert_eq!(response_id, Some(99));
        assert_eq!(args, vec![10, 20]);
    }

    #[test]
    fn test_rpc_call_encode_decode_response() {
        let call = RpcCall::response(10, RpcTarget::Client(5), vec![42], 99);
        let encoded = call.encode_payload();
        let (rpc_id, flags, response_id, args, _consumed) =
            RpcCall::decode_payload(&encoded).unwrap();

        assert_eq!(rpc_id, 10);
        assert_eq!(flags & FLAG_HAS_RESPONSE_ID, FLAG_HAS_RESPONSE_ID);
        assert_eq!(flags & FLAG_IS_RESPONSE, FLAG_IS_RESPONSE);
        assert_eq!(response_id, Some(99));
        assert_eq!(args, vec![42]);
    }

    #[test]
    fn test_rpc_call_empty_payload() {
        let call = RpcCall::new(1, RpcTarget::Server, vec![], RpcReliability::Unreliable);
        let encoded = call.encode_payload();
        let (rpc_id, _flags, _response_id, args, _consumed) =
            RpcCall::decode_payload(&encoded).unwrap();
        assert_eq!(rpc_id, 1);
        assert!(args.is_empty());
    }

    // -----------------------------------------------------------------------
    // RpcBatch
    // -----------------------------------------------------------------------

    #[test]
    fn test_rpc_batch_encode_decode() {
        let mut batch = RpcBatch::new();
        batch.add(RpcCall::new(1, RpcTarget::Server, vec![10], RpcReliability::Reliable));
        batch.add(RpcCall::new(2, RpcTarget::Server, vec![20, 30], RpcReliability::Unreliable));
        batch.add(RpcCall::new(3, RpcTarget::Server, vec![], RpcReliability::ReliableOrdered));

        assert_eq!(batch.len(), 3);

        let encoded = batch.encode();
        let decoded = RpcBatch::decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded.calls[0].rpc_id, 1);
        assert_eq!(decoded.calls[0].args, vec![10]);
        assert_eq!(decoded.calls[1].rpc_id, 2);
        assert_eq!(decoded.calls[1].args, vec![20, 30]);
        assert_eq!(decoded.calls[2].rpc_id, 3);
        assert!(decoded.calls[2].args.is_empty());
    }

    #[test]
    fn test_rpc_batch_empty() {
        let batch = RpcBatch::new();
        assert!(batch.is_empty());
        let encoded = batch.encode();
        let decoded = RpcBatch::decode(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    // -----------------------------------------------------------------------
    // RateLimiter
    // -----------------------------------------------------------------------

    #[test]
    fn test_rate_limiter_allows_within_limit() {
        let mut limiter = RateLimiter::new(5, Duration::from_secs(1));
        for _ in 0..5 {
            assert!(limiter.check_and_record());
        }
        // 6th call should be rejected.
        assert!(!limiter.check_and_record());
        assert_eq!(limiter.rejected_count(), 1);
    }

    #[test]
    fn test_rate_limiter_remaining() {
        let mut limiter = RateLimiter::new(10, Duration::from_secs(1));
        assert_eq!(limiter.remaining(), 10);
        limiter.check_and_record();
        limiter.check_and_record();
        assert_eq!(limiter.remaining(), 8);
    }

    #[test]
    fn test_rate_limiter_reset() {
        let mut limiter = RateLimiter::new(5, Duration::from_secs(1));
        for _ in 0..5 {
            limiter.check_and_record();
        }
        assert!(!limiter.check_and_record());
        limiter.reset();
        assert!(limiter.check_and_record());
    }

    #[test]
    fn test_rate_limiter_would_allow() {
        let mut limiter = RateLimiter::new(2, Duration::from_secs(1));
        assert!(limiter.would_allow());
        limiter.check_and_record();
        assert!(limiter.would_allow());
        limiter.check_and_record();
        assert!(!limiter.would_allow());
    }

    // -----------------------------------------------------------------------
    // RpcManager registration
    // -----------------------------------------------------------------------

    #[test]
    fn test_rpc_manager_register() {
        let mut mgr = RpcManager::new();
        let handler = FnRpcHandler::new("test_func", |_caller, _args| Ok(None));
        let id = mgr.register_rpc("test_func", Box::new(handler));

        assert_eq!(id, 1);
        assert_eq!(mgr.registered_count(), 1);
        assert_eq!(mgr.get_rpc_id("test_func"), Some(1));
        assert_eq!(mgr.get_rpc_name(1), Some("test_func"));
    }

    #[test]
    fn test_rpc_manager_register_multiple() {
        let mut mgr = RpcManager::new();
        let h1 = FnRpcHandler::new("func_a", |_, _| Ok(None));
        let h2 = FnRpcHandler::new("func_b", |_, _| Ok(None));
        let h3 = FnRpcHandler::new("func_c", |_, _| Ok(None));

        let id1 = mgr.register_rpc("func_a", Box::new(h1));
        let id2 = mgr.register_rpc("func_b", Box::new(h2));
        let id3 = mgr.register_rpc("func_c", Box::new(h3));

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
        assert_eq!(mgr.registered_count(), 3);
    }

    #[test]
    fn test_rpc_manager_re_register() {
        let mut mgr = RpcManager::new();
        let h1 = FnRpcHandler::new("func", |_, _| Ok(None));
        let id1 = mgr.register_rpc("func", Box::new(h1));

        let h2 = FnRpcHandler::new("func", |_, _| Ok(Some(vec![42])));
        let id2 = mgr.register_rpc("func", Box::new(h2));

        // Same ID, handler replaced.
        assert_eq!(id1, id2);
        assert_eq!(mgr.registered_count(), 1);
    }

    #[test]
    fn test_rpc_manager_unregister() {
        let mut mgr = RpcManager::new();
        let h = FnRpcHandler::new("func", |_, _| Ok(None));
        mgr.register_rpc("func", Box::new(h));

        assert!(mgr.unregister_rpc("func"));
        assert_eq!(mgr.registered_count(), 0);
        assert!(mgr.get_rpc_id("func").is_none());

        assert!(!mgr.unregister_rpc("nonexistent"));
    }

    // -----------------------------------------------------------------------
    // RpcManager call and dispatch
    // -----------------------------------------------------------------------

    #[test]
    fn test_rpc_manager_call_and_flush() {
        let mut mgr = RpcManager::new();
        let h = FnRpcHandler::new("ping", |_, _| Ok(None));
        mgr.register_rpc("ping", Box::new(h));

        mgr.call_rpc(RpcTarget::Server, "ping", &[1, 2, 3]).unwrap();
        mgr.call_rpc(RpcTarget::Server, "ping", &[4, 5, 6]).unwrap();

        let flushed = mgr.flush();
        assert_eq!(flushed.len(), 1); // all same target, one batch
        assert_eq!(mgr.outgoing_queue_len(), 0);
    }

    #[test]
    fn test_rpc_manager_call_unknown() {
        let mut mgr = RpcManager::new();
        let result = mgr.call_rpc(RpcTarget::Server, "nonexistent", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_rpc_manager_dispatch() {
        let mut mgr = RpcManager::new();
        let h = FnRpcHandler::new("echo", |_caller, args| {
            Ok(Some(args.to_vec()))
        });
        let rpc_id = mgr.register_rpc("echo", Box::new(h));

        // Simulate receiving a batch.
        let mut batch = RpcBatch::new();
        batch.add(RpcCall::new(rpc_id, RpcTarget::Server, vec![42], RpcReliability::Reliable));
        let encoded = batch.encode();

        mgr.receive_batch(1, &encoded).unwrap();
        assert_eq!(mgr.incoming_queue_len(), 1);

        let results = mgr.dispatch_incoming();
        assert_eq!(results.len(), 1);
        match &results[0] {
            RpcDispatchResult::Ok(Some(data)) => assert_eq!(data, &vec![42]),
            _ => panic!("Expected Ok with data"),
        }
    }

    #[test]
    fn test_rpc_manager_dispatch_handler_not_found() {
        let mut mgr = RpcManager::new();

        // Send an RPC with an unregistered ID.
        let mut batch = RpcBatch::new();
        batch.add(RpcCall::new(999, RpcTarget::Server, vec![], RpcReliability::Reliable));
        let encoded = batch.encode();

        mgr.receive_batch(1, &encoded).unwrap();
        let results = mgr.dispatch_incoming();
        match &results[0] {
            RpcDispatchResult::HandlerNotFound(id) => assert_eq!(*id, 999),
            _ => panic!("Expected HandlerNotFound"),
        }
    }

    #[test]
    fn test_rpc_manager_rate_limiting() {
        let mut mgr = RpcManager::new();
        mgr.set_client_rate_limit(1, 2);

        let h = FnRpcHandler::new("func", |_, _| Ok(None));
        let rpc_id = mgr.register_rpc("func", Box::new(h));

        // Send 3 calls from client 1, only 2 should be dispatched.
        let mut batch = RpcBatch::new();
        batch.add(RpcCall::new(rpc_id, RpcTarget::Server, vec![], RpcReliability::Reliable));
        batch.add(RpcCall::new(rpc_id, RpcTarget::Server, vec![], RpcReliability::Reliable));
        batch.add(RpcCall::new(rpc_id, RpcTarget::Server, vec![], RpcReliability::Reliable));
        let encoded = batch.encode();

        mgr.receive_batch(1, &encoded).unwrap();
        let results = mgr.dispatch_incoming();

        let ok_count = results
            .iter()
            .filter(|r| matches!(r, RpcDispatchResult::Ok(_)))
            .count();
        let limited_count = results
            .iter()
            .filter(|r| matches!(r, RpcDispatchResult::RateLimited(_)))
            .count();

        assert_eq!(ok_count, 2);
        assert_eq!(limited_count, 1);
    }

    #[test]
    fn test_rpc_manager_request_response() {
        let mut mgr = RpcManager::new();
        let h = FnRpcHandler::new("add", |_caller, args| {
            if args.len() >= 2 {
                let result = args[0].wrapping_add(args[1]);
                Ok(Some(vec![result]))
            } else {
                Ok(None)
            }
        });
        mgr.register_rpc("add", Box::new(h));

        // Send request.
        let response_id = mgr
            .call_rpc_with_response(RpcTarget::Server, "add", &[3, 4])
            .unwrap();
        assert_eq!(mgr.pending_response_count(), 1);

        // Poll before response arrives: should be None.
        let result = mgr.poll_response(response_id).unwrap();
        assert!(result.is_none());

        // Simulate the response arriving.
        let mut batch = RpcBatch::new();
        batch.add(RpcCall::response(1, RpcTarget::Server, vec![7], response_id));
        let encoded = batch.encode();
        mgr.receive_batch(0, &encoded).unwrap();
        mgr.dispatch_incoming();

        // Now poll should return the response.
        let result = mgr.poll_response(response_id).unwrap();
        assert_eq!(result, Some(vec![7]));
        assert_eq!(mgr.pending_response_count(), 0);
    }

    #[test]
    fn test_rpc_manager_stats() {
        let mut mgr = RpcManager::new();
        let h = FnRpcHandler::new("noop", |_, _| Ok(None));
        mgr.register_rpc("noop", Box::new(h));

        mgr.call_rpc(RpcTarget::Server, "noop", &[1]).unwrap();
        assert_eq!(mgr.stats().rpcs_sent, 1);

        let flushed = mgr.flush();
        assert_eq!(mgr.stats().batches_sent, 1);
        assert!(mgr.stats().bytes_sent > 0);

        // Receive the batch back.
        mgr.receive_batch(0, &flushed[0].1).unwrap();
        assert_eq!(mgr.stats().rpcs_received, 1);
        assert_eq!(mgr.stats().batches_received, 1);

        mgr.dispatch_incoming();
        assert_eq!(mgr.stats().rpcs_dispatched, 1);
    }

    #[test]
    fn test_rpc_manager_remove_client() {
        let mut mgr = RpcManager::new();
        mgr.set_client_rate_limit(1, 100);
        mgr.remove_client(1);
        // Should not panic.
    }

    #[test]
    fn test_rpc_manager_call_with_reliability() {
        let mut mgr = RpcManager::new();
        let h = FnRpcHandler::new("func", |_, _| Ok(None));
        mgr.register_rpc("func", Box::new(h));

        mgr.call_rpc_with_reliability(
            RpcTarget::Server,
            "func",
            &[],
            RpcReliability::Unreliable,
        )
        .unwrap();

        assert_eq!(mgr.outgoing_queue_len(), 1);
    }

    #[test]
    fn test_rpc_manager_multiple_targets() {
        let mut mgr = RpcManager::new();
        let h = FnRpcHandler::new("func", |_, _| Ok(None));
        mgr.register_rpc("func", Box::new(h));

        mgr.call_rpc(RpcTarget::Server, "func", &[1]).unwrap();
        mgr.call_rpc(RpcTarget::Client(1), "func", &[2]).unwrap();
        mgr.call_rpc(RpcTarget::Client(2), "func", &[3]).unwrap();

        let flushed = mgr.flush();
        // Should produce batches per unique target.
        assert!(flushed.len() >= 2); // at least Server and Client(x) groups
    }

    #[test]
    fn test_rpc_manager_handler_error() {
        let mut mgr = RpcManager::new();
        let h = FnRpcHandler::new("fail", |_, _| {
            Err(EngineError::InvalidArgument("bad args".into()))
        });
        let rpc_id = mgr.register_rpc("fail", Box::new(h));

        let mut batch = RpcBatch::new();
        batch.add(RpcCall::new(rpc_id, RpcTarget::Server, vec![], RpcReliability::Reliable));
        let encoded = batch.encode();

        mgr.receive_batch(1, &encoded).unwrap();
        let results = mgr.dispatch_incoming();

        match &results[0] {
            RpcDispatchResult::Error(msg) => assert!(msg.contains("bad args")),
            _ => panic!("Expected error result"),
        }
        assert_eq!(mgr.stats().rpcs_failed, 1);
    }
}
