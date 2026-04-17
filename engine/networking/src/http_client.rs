//! Simple HTTP/1.1 client for the Genovo networking module.
//!
//! Provides a lightweight, zero-dependency HTTP client built directly on TCP
//! sockets. Designed for use within a game engine where pulling in a full HTTP
//! library (hyper, reqwest) is undesirable.
//!
//! # Features
//!
//! - HTTP/1.1 request building (GET, POST, PUT, DELETE, PATCH, HEAD)
//! - Header parsing with case-insensitive matching
//! - Chunked transfer-encoding decoding
//! - Content-Length based body reading
//! - Simple JSON body parsing (values, objects, arrays)
//! - Configurable timeouts (connect, read, total)
//! - Automatic redirect following (3xx) with configurable limit
//! - Connection pooling with idle timeout eviction
//! - Basic and Bearer authentication helpers
//! - Query string building
//!
//! # Example
//!
//! ```ignore
//! let mut client = HttpClient::new();
//! client.set_timeout(Duration::from_secs(10));
//!
//! let response = client.get("http://api.example.com/status")?;
//! assert_eq!(response.status_code, 200);
//! println!("{}", response.body_text());
//! ```

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default connect timeout in seconds.
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 30;

/// Default read timeout in seconds.
const DEFAULT_READ_TIMEOUT_SECS: u64 = 60;

/// Default maximum number of redirects to follow.
const DEFAULT_MAX_REDIRECTS: u32 = 10;

/// Default maximum response body size (16 MiB).
const DEFAULT_MAX_BODY_SIZE: usize = 16 * 1024 * 1024;

/// Maximum number of pooled connections per host.
const DEFAULT_POOL_SIZE_PER_HOST: usize = 4;

/// Idle timeout for pooled connections in seconds.
const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 90;

/// Buffer size for reading from sockets.
const READ_BUFFER_SIZE: usize = 8192;

/// Maximum header size (64 KiB).
const MAX_HEADER_SIZE: usize = 64 * 1024;

/// HTTP version string.
const HTTP_VERSION: &str = "HTTP/1.1";

/// Default User-Agent header value.
const DEFAULT_USER_AGENT: &str = "GenovoEngine/1.0";

/// Line ending used in HTTP.
const CRLF: &str = "\r\n";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during HTTP operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpError {
    /// Failed to resolve the hostname to an address.
    DnsResolutionFailed(String),
    /// TCP connection failed.
    ConnectionFailed(String),
    /// Connection timed out.
    ConnectTimeout,
    /// Read timed out waiting for response data.
    ReadTimeout,
    /// The total request timeout was exceeded.
    TotalTimeout,
    /// The response headers exceeded the maximum allowed size.
    HeadersTooLarge,
    /// The response body exceeded the maximum allowed size.
    BodyTooLarge,
    /// The server returned a malformed response.
    MalformedResponse(String),
    /// A malformed URL was provided.
    InvalidUrl(String),
    /// Too many redirects were followed.
    TooManyRedirects,
    /// Chunked transfer encoding was malformed.
    InvalidChunkedEncoding(String),
    /// An I/O error occurred.
    IoError(String),
    /// JSON parsing failed.
    JsonParseError(String),
    /// The request was cancelled.
    Cancelled,
    /// Connection was closed unexpectedly.
    ConnectionClosed,
    /// Invalid HTTP method.
    InvalidMethod(String),
    /// SSL/TLS is not supported in this implementation.
    TlsNotSupported,
}

impl fmt::Display for HttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DnsResolutionFailed(host) => write!(f, "DNS resolution failed for '{host}'"),
            Self::ConnectionFailed(msg) => write!(f, "connection failed: {msg}"),
            Self::ConnectTimeout => write!(f, "connect timeout"),
            Self::ReadTimeout => write!(f, "read timeout"),
            Self::TotalTimeout => write!(f, "total request timeout exceeded"),
            Self::HeadersTooLarge => write!(f, "response headers too large"),
            Self::BodyTooLarge => write!(f, "response body too large"),
            Self::MalformedResponse(msg) => write!(f, "malformed response: {msg}"),
            Self::InvalidUrl(url) => write!(f, "invalid URL: {url}"),
            Self::TooManyRedirects => write!(f, "too many redirects"),
            Self::InvalidChunkedEncoding(msg) => write!(f, "invalid chunked encoding: {msg}"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
            Self::JsonParseError(msg) => write!(f, "JSON parse error: {msg}"),
            Self::Cancelled => write!(f, "request cancelled"),
            Self::ConnectionClosed => write!(f, "connection closed unexpectedly"),
            Self::InvalidMethod(m) => write!(f, "invalid HTTP method: {m}"),
            Self::TlsNotSupported => write!(f, "TLS/HTTPS not supported"),
        }
    }
}

/// Result type alias for HTTP operations.
pub type HttpResult<T> = Result<T, HttpError>;

// ---------------------------------------------------------------------------
// HTTP Method
// ---------------------------------------------------------------------------

/// HTTP request methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    /// GET — retrieve a resource.
    Get,
    /// POST — create or submit data.
    Post,
    /// PUT — replace a resource.
    Put,
    /// DELETE — remove a resource.
    Delete,
    /// PATCH — partially update a resource.
    Patch,
    /// HEAD — like GET but without a response body.
    Head,
    /// OPTIONS — describe communication options.
    Options,
}

impl HttpMethod {
    /// Returns the method as an HTTP-compliant string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Patch => "PATCH",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
        }
    }

    /// Returns `true` if the method typically carries a request body.
    pub fn has_body(&self) -> bool {
        matches!(self, Self::Post | Self::Put | Self::Patch)
    }

    /// Parse a method from a string.
    pub fn from_str(s: &str) -> HttpResult<Self> {
        match s.to_uppercase().as_str() {
            "GET" => Ok(Self::Get),
            "POST" => Ok(Self::Post),
            "PUT" => Ok(Self::Put),
            "DELETE" => Ok(Self::Delete),
            "PATCH" => Ok(Self::Patch),
            "HEAD" => Ok(Self::Head),
            "OPTIONS" => Ok(Self::Options),
            other => Err(HttpError::InvalidMethod(other.to_string())),
        }
    }
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// URL Parsing
// ---------------------------------------------------------------------------

/// A parsed URL broken into its components.
#[derive(Debug, Clone)]
pub struct ParsedUrl {
    /// The scheme (http or https).
    pub scheme: String,
    /// The hostname.
    pub host: String,
    /// The port number (defaults to 80 for http, 443 for https).
    pub port: u16,
    /// The path component (starts with /).
    pub path: String,
    /// Optional query string (without the leading ?).
    pub query: Option<String>,
    /// Optional fragment (without the leading #).
    pub fragment: Option<String>,
}

impl ParsedUrl {
    /// Parse a URL string into components.
    pub fn parse(url: &str) -> HttpResult<Self> {
        let url = url.trim();

        // Extract scheme.
        let (scheme, rest) = if let Some(idx) = url.find("://") {
            let s = url[..idx].to_lowercase();
            (s, &url[idx + 3..])
        } else {
            return Err(HttpError::InvalidUrl(
                "missing scheme (expected http:// or https://)".to_string(),
            ));
        };

        if scheme != "http" && scheme != "https" {
            return Err(HttpError::InvalidUrl(format!(
                "unsupported scheme: {scheme}"
            )));
        }

        // Extract fragment.
        let (rest, fragment) = if let Some(idx) = rest.find('#') {
            (&rest[..idx], Some(rest[idx + 1..].to_string()))
        } else {
            (rest, None)
        };

        // Extract query.
        let (rest, query) = if let Some(idx) = rest.find('?') {
            (&rest[..idx], Some(rest[idx + 1..].to_string()))
        } else {
            (rest, None)
        };

        // Extract host and path.
        let (authority, path) = if let Some(idx) = rest.find('/') {
            (&rest[..idx], rest[idx..].to_string())
        } else {
            (rest, "/".to_string())
        };

        // Extract host and port from authority.
        let (host, port) = if let Some(idx) = authority.rfind(':') {
            let host_part = &authority[..idx];
            let port_str = &authority[idx + 1..];
            let port: u16 = port_str.parse().map_err(|_| {
                HttpError::InvalidUrl(format!("invalid port: {port_str}"))
            })?;
            (host_part.to_string(), port)
        } else {
            let default_port = if scheme == "https" { 443 } else { 80 };
            (authority.to_string(), default_port)
        };

        if host.is_empty() {
            return Err(HttpError::InvalidUrl("empty hostname".to_string()));
        }

        Ok(Self {
            scheme,
            host,
            port,
            path,
            query,
            fragment,
        })
    }

    /// Returns the authority string (host:port or just host if default port).
    pub fn authority(&self) -> String {
        let default_port = if self.scheme == "https" { 443 } else { 80 };
        if self.port == default_port {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }

    /// Returns the full request path including query string.
    pub fn request_path(&self) -> String {
        match &self.query {
            Some(q) => format!("{}?{}", self.path, q),
            None => self.path.clone(),
        }
    }

    /// Returns the host:port key used for connection pooling.
    pub fn pool_key(&self) -> String {
        format!("{}:{}:{}", self.scheme, self.host, self.port)
    }
}

impl fmt::Display for ParsedUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}://{}{}", self.scheme, self.authority(), self.path)?;
        if let Some(ref q) = self.query {
            write!(f, "?{q}")?;
        }
        if let Some(ref frag) = self.fragment {
            write!(f, "#{frag}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Headers
// ---------------------------------------------------------------------------

/// A collection of HTTP headers with case-insensitive key lookup.
#[derive(Debug, Clone, Default)]
pub struct HeaderMap {
    /// Stored as (original_name, value) pairs.
    entries: Vec<(String, String)>,
}

impl HeaderMap {
    /// Create a new empty header map.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Insert a header. If the header already exists, it is replaced.
    pub fn set(&mut self, name: &str, value: &str) {
        let lower = name.to_lowercase();
        for entry in &mut self.entries {
            if entry.0.to_lowercase() == lower {
                entry.1 = value.to_string();
                return;
            }
        }
        self.entries.push((name.to_string(), value.to_string()));
    }

    /// Append a header value (allows multiple values for the same key).
    pub fn append(&mut self, name: &str, value: &str) {
        self.entries.push((name.to_string(), value.to_string()));
    }

    /// Get the first value for a header (case-insensitive).
    pub fn get(&self, name: &str) -> Option<&str> {
        let lower = name.to_lowercase();
        self.entries
            .iter()
            .find(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
    }

    /// Get all values for a header (case-insensitive).
    pub fn get_all(&self, name: &str) -> Vec<&str> {
        let lower = name.to_lowercase();
        self.entries
            .iter()
            .filter(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    /// Remove all entries for a header name (case-insensitive).
    pub fn remove(&mut self, name: &str) {
        let lower = name.to_lowercase();
        self.entries.retain(|(k, _)| k.to_lowercase() != lower);
    }

    /// Returns `true` if the header exists (case-insensitive).
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Returns the number of header entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if there are no headers.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all (name, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Serialize headers into the HTTP wire format.
    pub fn to_wire_format(&self) -> String {
        let mut out = String::new();
        for (name, value) in &self.entries {
            out.push_str(name);
            out.push_str(": ");
            out.push_str(value);
            out.push_str(CRLF);
        }
        out
    }

    /// Parse headers from raw header text (after the status line).
    pub fn parse(raw: &str) -> HttpResult<Self> {
        let mut map = Self::new();
        for line in raw.lines() {
            if line.is_empty() {
                continue;
            }
            if let Some(colon_idx) = line.find(':') {
                let name = line[..colon_idx].trim().to_string();
                let value = line[colon_idx + 1..].trim().to_string();
                if name.is_empty() {
                    return Err(HttpError::MalformedResponse(
                        "empty header name".to_string(),
                    ));
                }
                map.entries.push((name, value));
            } else {
                return Err(HttpError::MalformedResponse(format!(
                    "invalid header line: {line}"
                )));
            }
        }
        Ok(map)
    }

    /// Clear all headers.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ---------------------------------------------------------------------------
// Query string builder
// ---------------------------------------------------------------------------

/// Builder for URL query strings with proper percent-encoding.
#[derive(Debug, Clone, Default)]
pub struct QueryBuilder {
    params: Vec<(String, String)>,
}

impl QueryBuilder {
    /// Create an empty query builder.
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    /// Add a key-value parameter.
    pub fn param(mut self, key: &str, value: &str) -> Self {
        self.params.push((key.to_string(), value.to_string()));
        self
    }

    /// Build the query string (without leading '?').
    pub fn build(&self) -> String {
        self.params
            .iter()
            .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
            .collect::<Vec<_>>()
            .join("&")
    }

    /// Returns `true` if no parameters have been added.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Returns the number of parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }
}

/// Percent-encode a string for use in a URL query parameter.
pub fn percent_encode(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                output.push(byte as char);
            }
            _ => {
                output.push_str(&format!("%{byte:02X}"));
            }
        }
    }
    output
}

/// Percent-decode a URL-encoded string.
pub fn percent_decode(input: &str) -> String {
    let mut output = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(
                &input[i + 1..i + 3],
                16,
            ) {
                output.push(byte);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            output.push(b' ');
        } else {
            output.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&output).into_owned()
}

// ---------------------------------------------------------------------------
// HTTP Request
// ---------------------------------------------------------------------------

/// An HTTP request to be sent.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// The HTTP method.
    pub method: HttpMethod,
    /// The target URL.
    pub url: String,
    /// Request headers.
    pub headers: HeaderMap,
    /// Optional request body.
    pub body: Option<Vec<u8>>,
    /// Connect timeout override.
    pub connect_timeout: Option<Duration>,
    /// Read timeout override.
    pub read_timeout: Option<Duration>,
    /// Maximum redirects to follow (None = use client default).
    pub max_redirects: Option<u32>,
}

impl HttpRequest {
    /// Create a new request with the given method and URL.
    pub fn new(method: HttpMethod, url: &str) -> Self {
        Self {
            method,
            url: url.to_string(),
            headers: HeaderMap::new(),
            body: None,
            connect_timeout: None,
            read_timeout: None,
            max_redirects: None,
        }
    }

    /// Set a header on the request.
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.set(name, value);
        self
    }

    /// Set the request body as raw bytes.
    pub fn body_bytes(mut self, data: Vec<u8>) -> Self {
        self.body = Some(data);
        self
    }

    /// Set the request body as a UTF-8 string.
    pub fn body_text(mut self, text: &str) -> Self {
        self.body = Some(text.as_bytes().to_vec());
        self
    }

    /// Set the request body as JSON text and set Content-Type.
    pub fn body_json(mut self, json: &str) -> Self {
        self.headers.set("Content-Type", "application/json");
        self.body = Some(json.as_bytes().to_vec());
        self
    }

    /// Set Basic authentication header.
    pub fn basic_auth(mut self, username: &str, password: &str) -> Self {
        let credentials = format!("{username}:{password}");
        let encoded = base64_encode(credentials.as_bytes());
        self.headers.set("Authorization", &format!("Basic {encoded}"));
        self
    }

    /// Set Bearer token authentication header.
    pub fn bearer_auth(mut self, token: &str) -> Self {
        self.headers.set("Authorization", &format!("Bearer {token}"));
        self
    }

    /// Set the connect timeout for this request.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Set the read timeout for this request.
    pub fn with_read_timeout(mut self, timeout: Duration) -> Self {
        self.read_timeout = Some(timeout);
        self
    }

    /// Build the raw HTTP request bytes.
    fn build_raw(&self, parsed_url: &ParsedUrl) -> Vec<u8> {
        let request_path = parsed_url.request_path();
        let mut out = String::new();

        // Request line.
        out.push_str(&format!(
            "{} {} {}{CRLF}",
            self.method.as_str(),
            request_path,
            HTTP_VERSION
        ));

        // Host header (always required for HTTP/1.1).
        if !self.headers.contains("Host") {
            out.push_str(&format!("Host: {}{CRLF}", parsed_url.authority()));
        }

        // User-Agent if not set.
        if !self.headers.contains("User-Agent") {
            out.push_str(&format!("User-Agent: {DEFAULT_USER_AGENT}{CRLF}"));
        }

        // Content-Length if there is a body.
        if let Some(ref body) = self.body {
            if !self.headers.contains("Content-Length") {
                out.push_str(&format!("Content-Length: {}{CRLF}", body.len()));
            }
        }

        // Connection keep-alive.
        if !self.headers.contains("Connection") {
            out.push_str(&format!("Connection: keep-alive{CRLF}"));
        }

        // Accept if not set.
        if !self.headers.contains("Accept") {
            out.push_str(&format!("Accept: */*{CRLF}"));
        }

        // Custom headers.
        out.push_str(&self.headers.to_wire_format());

        // End of headers.
        out.push_str(CRLF);

        let mut bytes = out.into_bytes();

        // Append body.
        if let Some(ref body) = self.body {
            bytes.extend_from_slice(body);
        }

        bytes
    }
}

// ---------------------------------------------------------------------------
// HTTP Response
// ---------------------------------------------------------------------------

/// An HTTP response received from a server.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// The HTTP version string from the status line (e.g. "HTTP/1.1").
    pub http_version: String,
    /// The numeric status code (e.g. 200, 404).
    pub status_code: u16,
    /// The reason phrase (e.g. "OK", "Not Found").
    pub reason_phrase: String,
    /// Response headers.
    pub headers: HeaderMap,
    /// The response body bytes.
    pub body: Vec<u8>,
    /// The total time taken for the request.
    pub elapsed: Duration,
    /// The final URL (after any redirects).
    pub final_url: String,
    /// Number of redirects that were followed.
    pub redirect_count: u32,
}

impl HttpResponse {
    /// Returns the body decoded as UTF-8 text (lossy).
    pub fn body_text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// Returns `true` if the status code is in the 2xx range.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status_code)
    }

    /// Returns `true` if the status code is in the 3xx range.
    pub fn is_redirect(&self) -> bool {
        (300..400).contains(&self.status_code)
    }

    /// Returns `true` if the status code is in the 4xx range.
    pub fn is_client_error(&self) -> bool {
        (400..500).contains(&self.status_code)
    }

    /// Returns `true` if the status code is in the 5xx range.
    pub fn is_server_error(&self) -> bool {
        (500..600).contains(&self.status_code)
    }

    /// Returns the content length from headers, if present.
    pub fn content_length(&self) -> Option<usize> {
        self.headers
            .get("Content-Length")
            .and_then(|v| v.parse().ok())
    }

    /// Returns the Content-Type header value.
    pub fn content_type(&self) -> Option<&str> {
        self.headers.get("Content-Type")
    }

    /// Parse the body as a JSON value.
    pub fn json(&self) -> HttpResult<JsonValue> {
        let text = self.body_text();
        parse_json(&text)
    }
}

impl fmt::Display for HttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {} ({:.1}ms, {} bytes)",
            self.http_version,
            self.status_code,
            self.reason_phrase,
            self.elapsed.as_secs_f64() * 1000.0,
            self.body.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Connection Pool
// ---------------------------------------------------------------------------

/// A single pooled TCP connection.
struct PooledConnection {
    stream: TcpStream,
    created_at: Instant,
    last_used: Instant,
}

/// Connection pool for reusing TCP connections across requests.
pub struct ConnectionPool {
    /// Map from pool key (scheme:host:port) to a list of idle connections.
    connections: HashMap<String, Vec<PooledConnection>>,
    /// Maximum connections per host.
    max_per_host: usize,
    /// Idle timeout after which a connection is evicted.
    idle_timeout: Duration,
    /// Total number of connections currently pooled.
    total_connections: usize,
    /// Statistics: total connections created.
    stats_created: u64,
    /// Statistics: total connections reused.
    stats_reused: u64,
    /// Statistics: total connections evicted.
    stats_evicted: u64,
}

impl ConnectionPool {
    /// Create a new connection pool with default settings.
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
            max_per_host: DEFAULT_POOL_SIZE_PER_HOST,
            idle_timeout: Duration::from_secs(DEFAULT_POOL_IDLE_TIMEOUT_SECS),
            total_connections: 0,
            stats_created: 0,
            stats_reused: 0,
            stats_evicted: 0,
        }
    }

    /// Create a connection pool with custom settings.
    pub fn with_config(max_per_host: usize, idle_timeout: Duration) -> Self {
        Self {
            connections: HashMap::new(),
            max_per_host,
            idle_timeout,
            total_connections: 0,
            stats_created: 0,
            stats_reused: 0,
            stats_evicted: 0,
        }
    }

    /// Try to acquire a pooled connection for the given key.
    pub fn acquire(&mut self, key: &str) -> Option<TcpStream> {
        self.evict_expired();

        if let Some(conns) = self.connections.get_mut(key) {
            if let Some(pooled) = conns.pop() {
                self.total_connections -= 1;
                self.stats_reused += 1;
                // Try to clone the stream to verify it's still alive.
                match pooled.stream.try_clone() {
                    Ok(stream) => return Some(stream),
                    Err(_) => {
                        // Connection is dead, try next one.
                        return self.acquire(key);
                    }
                }
            }
        }
        None
    }

    /// Return a connection to the pool.
    pub fn release(&mut self, key: &str, stream: TcpStream) {
        self.evict_expired();

        let conns = self.connections.entry(key.to_string()).or_default();
        if conns.len() < self.max_per_host {
            conns.push(PooledConnection {
                stream,
                created_at: Instant::now(),
                last_used: Instant::now(),
            });
            self.total_connections += 1;
        }
        // If at capacity, just drop the connection.
    }

    /// Evict all expired idle connections.
    fn evict_expired(&mut self) {
        let now = Instant::now();
        let idle_timeout = self.idle_timeout;
        let mut evicted = 0u64;

        for conns in self.connections.values_mut() {
            let before = conns.len();
            conns.retain(|c| now.duration_since(c.last_used) < idle_timeout);
            let after = conns.len();
            evicted += (before - after) as u64;
        }

        self.total_connections -= evicted as usize;
        self.stats_evicted += evicted;

        // Remove empty keys.
        self.connections.retain(|_, v| !v.is_empty());
    }

    /// Clear all pooled connections.
    pub fn clear(&mut self) {
        let total = self.total_connections;
        self.connections.clear();
        self.total_connections = 0;
        self.stats_evicted += total as u64;
    }

    /// Returns pool statistics.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            active_connections: self.total_connections,
            total_created: self.stats_created,
            total_reused: self.stats_reused,
            total_evicted: self.stats_evicted,
            hosts: self.connections.len(),
        }
    }
}

impl Default for ConnectionPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the connection pool.
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    /// Number of connections currently pooled.
    pub active_connections: usize,
    /// Total connections ever created.
    pub total_created: u64,
    /// Total connections reused from pool.
    pub total_reused: u64,
    /// Total connections evicted due to idle timeout.
    pub total_evicted: u64,
    /// Number of distinct hosts with pooled connections.
    pub hosts: usize,
}

impl fmt::Display for PoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pool[active={}, created={}, reused={}, evicted={}, hosts={}]",
            self.active_connections,
            self.total_created,
            self.total_reused,
            self.total_evicted,
            self.hosts
        )
    }
}

// ---------------------------------------------------------------------------
// HTTP Client
// ---------------------------------------------------------------------------

/// Configuration for the HTTP client.
#[derive(Debug, Clone)]
pub struct HttpClientConfig {
    /// Connect timeout.
    pub connect_timeout: Duration,
    /// Read timeout.
    pub read_timeout: Duration,
    /// Maximum number of redirects to follow.
    pub max_redirects: u32,
    /// Maximum response body size in bytes.
    pub max_body_size: usize,
    /// Default headers sent with every request.
    pub default_headers: HeaderMap,
    /// Whether to automatically follow redirects.
    pub follow_redirects: bool,
    /// Whether to use the connection pool.
    pub use_connection_pool: bool,
    /// Pool size per host.
    pub pool_size_per_host: usize,
    /// Pool idle timeout.
    pub pool_idle_timeout: Duration,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS),
            read_timeout: Duration::from_secs(DEFAULT_READ_TIMEOUT_SECS),
            max_redirects: DEFAULT_MAX_REDIRECTS,
            max_body_size: DEFAULT_MAX_BODY_SIZE,
            default_headers: HeaderMap::new(),
            follow_redirects: true,
            use_connection_pool: true,
            pool_size_per_host: DEFAULT_POOL_SIZE_PER_HOST,
            pool_idle_timeout: Duration::from_secs(DEFAULT_POOL_IDLE_TIMEOUT_SECS),
        }
    }
}

/// A simple HTTP/1.1 client.
pub struct HttpClient {
    /// Client configuration.
    config: HttpClientConfig,
    /// Connection pool.
    pool: ConnectionPool,
    /// Total requests made.
    request_count: u64,
    /// Total bytes sent.
    bytes_sent: u64,
    /// Total bytes received.
    bytes_received: u64,
}

impl HttpClient {
    /// Create a new HTTP client with default configuration.
    pub fn new() -> Self {
        Self {
            config: HttpClientConfig::default(),
            pool: ConnectionPool::new(),
            request_count: 0,
            bytes_sent: 0,
            bytes_received: 0,
        }
    }

    /// Create a new HTTP client with custom configuration.
    pub fn with_config(config: HttpClientConfig) -> Self {
        let pool = ConnectionPool::with_config(
            config.pool_size_per_host,
            config.pool_idle_timeout,
        );
        Self {
            config,
            pool,
            request_count: 0,
            bytes_sent: 0,
            bytes_received: 0,
        }
    }

    /// Set the connect timeout.
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.config.connect_timeout = timeout;
        self.config.read_timeout = timeout;
    }

    /// Set the connect timeout separately.
    pub fn set_connect_timeout(&mut self, timeout: Duration) {
        self.config.connect_timeout = timeout;
    }

    /// Set the read timeout separately.
    pub fn set_read_timeout(&mut self, timeout: Duration) {
        self.config.read_timeout = timeout;
    }

    /// Set whether to follow redirects.
    pub fn set_follow_redirects(&mut self, follow: bool) {
        self.config.follow_redirects = follow;
    }

    /// Set the maximum number of redirects.
    pub fn set_max_redirects(&mut self, max: u32) {
        self.config.max_redirects = max;
    }

    /// Set a default header sent with every request.
    pub fn set_default_header(&mut self, name: &str, value: &str) {
        self.config.default_headers.set(name, value);
    }

    /// Perform a GET request.
    pub fn get(&mut self, url: &str) -> HttpResult<HttpResponse> {
        let req = HttpRequest::new(HttpMethod::Get, url);
        self.execute(req)
    }

    /// Perform a POST request with a body.
    pub fn post(&mut self, url: &str, body: &str) -> HttpResult<HttpResponse> {
        let req = HttpRequest::new(HttpMethod::Post, url).body_text(body);
        self.execute(req)
    }

    /// Perform a POST request with a JSON body.
    pub fn post_json(&mut self, url: &str, json: &str) -> HttpResult<HttpResponse> {
        let req = HttpRequest::new(HttpMethod::Post, url).body_json(json);
        self.execute(req)
    }

    /// Perform a PUT request with a body.
    pub fn put(&mut self, url: &str, body: &str) -> HttpResult<HttpResponse> {
        let req = HttpRequest::new(HttpMethod::Put, url).body_text(body);
        self.execute(req)
    }

    /// Perform a PUT request with a JSON body.
    pub fn put_json(&mut self, url: &str, json: &str) -> HttpResult<HttpResponse> {
        let req = HttpRequest::new(HttpMethod::Put, url).body_json(json);
        self.execute(req)
    }

    /// Perform a DELETE request.
    pub fn delete(&mut self, url: &str) -> HttpResult<HttpResponse> {
        let req = HttpRequest::new(HttpMethod::Delete, url);
        self.execute(req)
    }

    /// Perform a PATCH request with a body.
    pub fn patch(&mut self, url: &str, body: &str) -> HttpResult<HttpResponse> {
        let req = HttpRequest::new(HttpMethod::Patch, url).body_text(body);
        self.execute(req)
    }

    /// Perform a HEAD request.
    pub fn head(&mut self, url: &str) -> HttpResult<HttpResponse> {
        let req = HttpRequest::new(HttpMethod::Head, url);
        self.execute(req)
    }

    /// Execute an arbitrary HTTP request.
    pub fn execute(&mut self, request: HttpRequest) -> HttpResult<HttpResponse> {
        let start = Instant::now();
        self.request_count += 1;

        let max_redirects = request
            .max_redirects
            .unwrap_or(self.config.max_redirects);

        let mut current_url = request.url.clone();
        let mut redirect_count = 0u32;

        loop {
            let parsed = ParsedUrl::parse(&current_url)?;

            if parsed.scheme == "https" {
                return Err(HttpError::TlsNotSupported);
            }

            // Build the request with merged headers.
            let mut merged_headers = self.config.default_headers.clone();
            for (name, value) in request.headers.iter() {
                merged_headers.set(name, value);
            }

            let mut effective_request = HttpRequest {
                method: request.method,
                url: current_url.clone(),
                headers: merged_headers,
                body: request.body.clone(),
                connect_timeout: request.connect_timeout,
                read_timeout: request.read_timeout,
                max_redirects: request.max_redirects,
            };

            let raw = effective_request.build_raw(&parsed);
            self.bytes_sent += raw.len() as u64;

            // Get or create a connection.
            let connect_timeout = effective_request
                .connect_timeout
                .unwrap_or(self.config.connect_timeout);
            let read_timeout = effective_request
                .read_timeout
                .unwrap_or(self.config.read_timeout);

            let mut stream = if self.config.use_connection_pool {
                if let Some(s) = self.pool.acquire(&parsed.pool_key()) {
                    s
                } else {
                    self.connect(&parsed, connect_timeout)?
                }
            } else {
                self.connect(&parsed, connect_timeout)?
            };

            stream
                .set_read_timeout(Some(read_timeout))
                .map_err(|e| HttpError::IoError(e.to_string()))?;

            // Send the request.
            stream
                .write_all(&raw)
                .map_err(|e| HttpError::IoError(e.to_string()))?;
            stream
                .flush()
                .map_err(|e| HttpError::IoError(e.to_string()))?;

            // Read the response.
            let response = self.read_response(
                &mut stream,
                request.method == HttpMethod::Head,
                &current_url,
                redirect_count,
                start,
            )?;

            self.bytes_received += response.body.len() as u64;

            // Return connection to pool if keep-alive.
            if self.config.use_connection_pool {
                let connection_header = response
                    .headers
                    .get("Connection")
                    .unwrap_or("keep-alive")
                    .to_lowercase();
                if connection_header != "close" {
                    if let Ok(cloned) = stream.try_clone() {
                        self.pool.release(&parsed.pool_key(), cloned);
                    }
                }
            }

            // Handle redirects.
            if self.config.follow_redirects && response.is_redirect() {
                if redirect_count >= max_redirects {
                    return Err(HttpError::TooManyRedirects);
                }
                if let Some(location) = response.headers.get("Location") {
                    current_url = resolve_redirect_url(&current_url, location)?;
                    redirect_count += 1;
                    continue;
                }
            }

            return Ok(response);
        }
    }

    /// Establish a TCP connection to the target host.
    fn connect(&mut self, url: &ParsedUrl, timeout: Duration) -> HttpResult<TcpStream> {
        let addr_str = format!("{}:{}", url.host, url.port);
        let addrs: Vec<SocketAddr> = addr_str
            .to_socket_addrs()
            .map_err(|_| HttpError::DnsResolutionFailed(url.host.clone()))?
            .collect();

        if addrs.is_empty() {
            return Err(HttpError::DnsResolutionFailed(url.host.clone()));
        }

        let mut last_err = None;
        for addr in &addrs {
            match TcpStream::connect_timeout(addr, timeout) {
                Ok(stream) => {
                    self.pool.stats_created += 1;
                    return Ok(stream);
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        match last_err {
            Some(e) if e.kind() == io::ErrorKind::TimedOut => Err(HttpError::ConnectTimeout),
            Some(e) => Err(HttpError::ConnectionFailed(e.to_string())),
            None => Err(HttpError::ConnectionFailed("no addresses to connect to".to_string())),
        }
    }

    /// Read and parse an HTTP response from a stream.
    fn read_response(
        &self,
        stream: &mut TcpStream,
        head_only: bool,
        url: &str,
        redirect_count: u32,
        start: Instant,
    ) -> HttpResult<HttpResponse> {
        let mut buf = Vec::with_capacity(READ_BUFFER_SIZE);
        let mut tmp = [0u8; READ_BUFFER_SIZE];

        // Read until we find the end of headers (\r\n\r\n).
        let header_end;
        loop {
            let n = stream
                .read(&mut tmp)
                .map_err(|e| {
                    if e.kind() == io::ErrorKind::TimedOut
                        || e.kind() == io::ErrorKind::WouldBlock
                    {
                        HttpError::ReadTimeout
                    } else {
                        HttpError::IoError(e.to_string())
                    }
                })?;
            if n == 0 {
                if buf.is_empty() {
                    return Err(HttpError::ConnectionClosed);
                }
                break;
            }
            buf.extend_from_slice(&tmp[..n]);

            if buf.len() > MAX_HEADER_SIZE {
                return Err(HttpError::HeadersTooLarge);
            }

            if let Some(pos) = find_header_end(&buf) {
                header_end = pos;
                break;
            }

            // If we haven't found the header end and the buffer is growing large.
            if buf.len() > MAX_HEADER_SIZE {
                return Err(HttpError::HeadersTooLarge);
            }

            continue;
        }

        // This is safe because if we break out of the loop without setting
        // header_end (the n==0 case with non-empty buf), we need a fallback.
        let header_end = if let Some(pos) = find_header_end(&buf) {
            pos
        } else {
            return Err(HttpError::MalformedResponse(
                "end of headers not found".to_string(),
            ));
        };

        // Parse the header block.
        let header_bytes = &buf[..header_end];
        let header_text = String::from_utf8_lossy(header_bytes).into_owned();

        // Split status line and headers.
        let (status_line, header_block) =
            if let Some(idx) = header_text.find("\r\n") {
                (&header_text[..idx], &header_text[idx + 2..])
            } else if let Some(idx) = header_text.find('\n') {
                (&header_text[..idx], &header_text[idx + 1..])
            } else {
                (header_text.as_str(), "")
            };

        // Parse the status line: "HTTP/1.1 200 OK"
        let (http_version, status_code, reason_phrase) = parse_status_line(status_line)?;

        // Parse headers.
        let headers = HeaderMap::parse(header_block)?;

        // Read body based on transfer encoding and content length.
        let body_start = header_end + 4; // skip \r\n\r\n
        let initial_body = if body_start < buf.len() {
            buf[body_start..].to_vec()
        } else {
            Vec::new()
        };

        let body = if head_only || status_code == 204 || status_code == 304 {
            Vec::new()
        } else if let Some(te) = headers.get("Transfer-Encoding") {
            if te.to_lowercase().contains("chunked") {
                self.read_chunked_body(stream, initial_body)?
            } else {
                self.read_body_to_end(stream, initial_body)?
            }
        } else if let Some(cl) = headers.get("Content-Length") {
            let content_length: usize = cl
                .trim()
                .parse()
                .map_err(|_| HttpError::MalformedResponse("invalid Content-Length".to_string()))?;
            if content_length > self.config.max_body_size {
                return Err(HttpError::BodyTooLarge);
            }
            self.read_exact_body(stream, initial_body, content_length)?
        } else {
            // No Content-Length and no chunked encoding — read until connection close
            // or use what we have if keep-alive.
            let connection = headers.get("Connection").unwrap_or("keep-alive").to_lowercase();
            if connection == "close" {
                self.read_body_to_end(stream, initial_body)?
            } else {
                initial_body
            }
        };

        Ok(HttpResponse {
            http_version,
            status_code,
            reason_phrase,
            headers,
            body,
            elapsed: start.elapsed(),
            final_url: url.to_string(),
            redirect_count,
        })
    }

    /// Read a body with a known Content-Length.
    fn read_exact_body(
        &self,
        stream: &mut TcpStream,
        initial: Vec<u8>,
        content_length: usize,
    ) -> HttpResult<Vec<u8>> {
        let mut body = initial;
        let mut tmp = [0u8; READ_BUFFER_SIZE];

        while body.len() < content_length {
            let n = stream
                .read(&mut tmp)
                .map_err(|e| {
                    if e.kind() == io::ErrorKind::TimedOut {
                        HttpError::ReadTimeout
                    } else {
                        HttpError::IoError(e.to_string())
                    }
                })?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&tmp[..n]);

            if body.len() > self.config.max_body_size {
                return Err(HttpError::BodyTooLarge);
            }
        }

        body.truncate(content_length);
        Ok(body)
    }

    /// Read a chunked transfer-encoded body.
    fn read_chunked_body(
        &self,
        stream: &mut TcpStream,
        initial: Vec<u8>,
    ) -> HttpResult<Vec<u8>> {
        let mut raw = initial;
        let mut tmp = [0u8; READ_BUFFER_SIZE];
        let mut decoded = Vec::new();
        let mut pos = 0;

        loop {
            // Ensure we have enough data to parse the next chunk size.
            while !has_line_at(&raw, pos) {
                let n = stream
                    .read(&mut tmp)
                    .map_err(|e| HttpError::IoError(e.to_string()))?;
                if n == 0 {
                    return Err(HttpError::InvalidChunkedEncoding(
                        "unexpected end of stream".to_string(),
                    ));
                }
                raw.extend_from_slice(&tmp[..n]);
            }

            // Parse chunk size line.
            let line_end = find_line_end(&raw, pos).unwrap();
            let size_str = String::from_utf8_lossy(&raw[pos..line_end]).trim().to_string();
            // Strip chunk extensions (;ext=val).
            let size_hex = size_str.split(';').next().unwrap_or("").trim();
            let chunk_size = usize::from_str_radix(size_hex, 16).map_err(|_| {
                HttpError::InvalidChunkedEncoding(format!("invalid chunk size: {size_hex}"))
            })?;

            pos = line_end + 2; // skip \r\n after size line

            if chunk_size == 0 {
                // Terminal chunk — skip trailing \r\n.
                break;
            }

            if decoded.len() + chunk_size > self.config.max_body_size {
                return Err(HttpError::BodyTooLarge);
            }

            // Ensure we have the full chunk data + trailing \r\n.
            let needed = pos + chunk_size + 2;
            while raw.len() < needed {
                let n = stream
                    .read(&mut tmp)
                    .map_err(|e| HttpError::IoError(e.to_string()))?;
                if n == 0 {
                    return Err(HttpError::InvalidChunkedEncoding(
                        "unexpected end of chunk data".to_string(),
                    ));
                }
                raw.extend_from_slice(&tmp[..n]);
            }

            decoded.extend_from_slice(&raw[pos..pos + chunk_size]);
            pos += chunk_size + 2; // skip chunk data + \r\n
        }

        Ok(decoded)
    }

    /// Read body until connection closes.
    fn read_body_to_end(
        &self,
        stream: &mut TcpStream,
        initial: Vec<u8>,
    ) -> HttpResult<Vec<u8>> {
        let mut body = initial;
        let mut tmp = [0u8; READ_BUFFER_SIZE];

        loop {
            let n = match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::TimedOut => break,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) => return Err(HttpError::IoError(e.to_string())),
            };
            body.extend_from_slice(&tmp[..n]);
            if body.len() > self.config.max_body_size {
                return Err(HttpError::BodyTooLarge);
            }
        }

        Ok(body)
    }

    /// Get the connection pool statistics.
    pub fn pool_stats(&self) -> PoolStats {
        self.pool.stats()
    }

    /// Clear the connection pool.
    pub fn clear_pool(&mut self) {
        self.pool.clear();
    }

    /// Get total request count.
    pub fn request_count(&self) -> u64 {
        self.request_count
    }

    /// Get total bytes sent.
    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent
    }

    /// Get total bytes received.
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// JSON Parser (minimal, self-contained)
// ---------------------------------------------------------------------------

/// A simple JSON value representation.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// JSON null.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// JSON number (stored as f64).
    Number(f64),
    /// JSON string.
    String(String),
    /// JSON array.
    Array(Vec<JsonValue>),
    /// JSON object (preserves insertion order).
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Returns `true` if this value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    /// Try to get this value as a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get this value as a number.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Try to get this value as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get this value as an array.
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Try to get this value as an object.
    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(entries) => Some(entries),
            _ => None,
        }
    }

    /// Get a field from a JSON object by key.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(entries) => {
                entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
            }
            _ => None,
        }
    }

    /// Get an element from a JSON array by index.
    pub fn index(&self, idx: usize) -> Option<&JsonValue> {
        match self {
            JsonValue::Array(arr) => arr.get(idx),
            _ => None,
        }
    }

    /// Serialize this value to a compact JSON string.
    pub fn to_json_string(&self) -> String {
        match self {
            JsonValue::Null => "null".to_string(),
            JsonValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            JsonValue::Number(n) => {
                if n.fract() == 0.0 && n.abs() < i64::MAX as f64 {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
            JsonValue::String(s) => format!("\"{}\"", escape_json_string(s)),
            JsonValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_json_string()).collect();
                format!("[{}]", items.join(","))
            }
            JsonValue::Object(entries) => {
                let items: Vec<String> = entries
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", escape_json_string(k), v.to_json_string()))
                    .collect();
                format!("{{{}}}", items.join(","))
            }
        }
    }
}

impl fmt::Display for JsonValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_json_string())
    }
}

/// Escape special characters in a JSON string.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Parse a JSON string into a `JsonValue`.
pub fn parse_json(input: &str) -> HttpResult<JsonValue> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(HttpError::JsonParseError("empty input".to_string()));
    }
    let chars: Vec<char> = trimmed.chars().collect();
    let mut pos = 0;
    let result = parse_json_value(&chars, &mut pos)?;
    skip_whitespace(&chars, &mut pos);
    if pos < chars.len() {
        return Err(HttpError::JsonParseError(format!(
            "unexpected trailing content at position {pos}"
        )));
    }
    Ok(result)
}

fn skip_whitespace(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn parse_json_value(chars: &[char], pos: &mut usize) -> HttpResult<JsonValue> {
    skip_whitespace(chars, pos);
    if *pos >= chars.len() {
        return Err(HttpError::JsonParseError("unexpected end of input".to_string()));
    }
    match chars[*pos] {
        '"' => parse_json_string(chars, pos).map(JsonValue::String),
        '{' => parse_json_object(chars, pos),
        '[' => parse_json_array(chars, pos),
        't' | 'f' => parse_json_bool(chars, pos),
        'n' => parse_json_null(chars, pos),
        '-' | '0'..='9' => parse_json_number(chars, pos),
        c => Err(HttpError::JsonParseError(format!(
            "unexpected character '{c}' at position {}",
            *pos
        ))),
    }
}

fn parse_json_string(chars: &[char], pos: &mut usize) -> HttpResult<String> {
    if chars[*pos] != '"' {
        return Err(HttpError::JsonParseError("expected '\"'".to_string()));
    }
    *pos += 1;
    let mut s = String::new();
    while *pos < chars.len() {
        let c = chars[*pos];
        *pos += 1;
        match c {
            '"' => return Ok(s),
            '\\' => {
                if *pos >= chars.len() {
                    return Err(HttpError::JsonParseError("unexpected end in escape".to_string()));
                }
                let esc = chars[*pos];
                *pos += 1;
                match esc {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    '/' => s.push('/'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    'b' => s.push('\u{0008}'),
                    'f' => s.push('\u{000C}'),
                    'u' => {
                        let mut hex = String::new();
                        for _ in 0..4 {
                            if *pos >= chars.len() {
                                return Err(HttpError::JsonParseError(
                                    "unexpected end in unicode escape".to_string(),
                                ));
                            }
                            hex.push(chars[*pos]);
                            *pos += 1;
                        }
                        let code = u32::from_str_radix(&hex, 16).map_err(|_| {
                            HttpError::JsonParseError(format!("invalid unicode escape: \\u{hex}"))
                        })?;
                        if let Some(ch) = char::from_u32(code) {
                            s.push(ch);
                        } else {
                            s.push('\u{FFFD}');
                        }
                    }
                    _ => {
                        return Err(HttpError::JsonParseError(format!(
                            "invalid escape character: \\{esc}"
                        )));
                    }
                }
            }
            _ => s.push(c),
        }
    }
    Err(HttpError::JsonParseError("unterminated string".to_string()))
}

fn parse_json_number(chars: &[char], pos: &mut usize) -> HttpResult<JsonValue> {
    let start = *pos;
    if *pos < chars.len() && chars[*pos] == '-' {
        *pos += 1;
    }
    while *pos < chars.len() && chars[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos < chars.len() && chars[*pos] == '.' {
        *pos += 1;
        while *pos < chars.len() && chars[*pos].is_ascii_digit() {
            *pos += 1;
        }
    }
    if *pos < chars.len() && (chars[*pos] == 'e' || chars[*pos] == 'E') {
        *pos += 1;
        if *pos < chars.len() && (chars[*pos] == '+' || chars[*pos] == '-') {
            *pos += 1;
        }
        while *pos < chars.len() && chars[*pos].is_ascii_digit() {
            *pos += 1;
        }
    }
    let num_str: String = chars[start..*pos].iter().collect();
    let n: f64 = num_str.parse().map_err(|_| {
        HttpError::JsonParseError(format!("invalid number: {num_str}"))
    })?;
    Ok(JsonValue::Number(n))
}

fn parse_json_bool(chars: &[char], pos: &mut usize) -> HttpResult<JsonValue> {
    if chars[*pos..].starts_with(&['t', 'r', 'u', 'e']) {
        *pos += 4;
        Ok(JsonValue::Bool(true))
    } else if chars[*pos..].starts_with(&['f', 'a', 'l', 's', 'e']) {
        *pos += 5;
        Ok(JsonValue::Bool(false))
    } else {
        Err(HttpError::JsonParseError(format!(
            "unexpected token at position {}",
            *pos
        )))
    }
}

fn parse_json_null(chars: &[char], pos: &mut usize) -> HttpResult<JsonValue> {
    if chars[*pos..].starts_with(&['n', 'u', 'l', 'l']) {
        *pos += 4;
        Ok(JsonValue::Null)
    } else {
        Err(HttpError::JsonParseError(format!(
            "unexpected token at position {}",
            *pos
        )))
    }
}

fn parse_json_array(chars: &[char], pos: &mut usize) -> HttpResult<JsonValue> {
    *pos += 1; // skip [
    skip_whitespace(chars, pos);
    let mut arr = Vec::new();
    if *pos < chars.len() && chars[*pos] == ']' {
        *pos += 1;
        return Ok(JsonValue::Array(arr));
    }
    loop {
        let val = parse_json_value(chars, pos)?;
        arr.push(val);
        skip_whitespace(chars, pos);
        if *pos >= chars.len() {
            return Err(HttpError::JsonParseError("unterminated array".to_string()));
        }
        if chars[*pos] == ']' {
            *pos += 1;
            return Ok(JsonValue::Array(arr));
        }
        if chars[*pos] != ',' {
            return Err(HttpError::JsonParseError(format!(
                "expected ',' or ']' in array at position {}",
                *pos
            )));
        }
        *pos += 1;
    }
}

fn parse_json_object(chars: &[char], pos: &mut usize) -> HttpResult<JsonValue> {
    *pos += 1; // skip {
    skip_whitespace(chars, pos);
    let mut entries = Vec::new();
    if *pos < chars.len() && chars[*pos] == '}' {
        *pos += 1;
        return Ok(JsonValue::Object(entries));
    }
    loop {
        skip_whitespace(chars, pos);
        if *pos >= chars.len() || chars[*pos] != '"' {
            return Err(HttpError::JsonParseError("expected string key in object".to_string()));
        }
        let key = parse_json_string(chars, pos)?;
        skip_whitespace(chars, pos);
        if *pos >= chars.len() || chars[*pos] != ':' {
            return Err(HttpError::JsonParseError("expected ':' in object".to_string()));
        }
        *pos += 1;
        let val = parse_json_value(chars, pos)?;
        entries.push((key, val));
        skip_whitespace(chars, pos);
        if *pos >= chars.len() {
            return Err(HttpError::JsonParseError("unterminated object".to_string()));
        }
        if chars[*pos] == '}' {
            *pos += 1;
            return Ok(JsonValue::Object(entries));
        }
        if chars[*pos] != ',' {
            return Err(HttpError::JsonParseError(format!(
                "expected ',' or '}}' in object at position {}",
                *pos
            )));
        }
        *pos += 1;
    }
}

// ---------------------------------------------------------------------------
// Base64 encoder (minimal, for auth headers)
// ---------------------------------------------------------------------------

const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode bytes to base64.
pub fn base64_encode(input: &[u8]) -> String {
    let mut output = String::with_capacity((input.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 2 < input.len() {
        let b0 = input[i] as u32;
        let b1 = input[i + 1] as u32;
        let b2 = input[i + 2] as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        output.push(BASE64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        output.push(BASE64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        output.push(BASE64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        output.push(BASE64_CHARS[(triple & 0x3F) as usize] as char);
        i += 3;
    }
    let remaining = input.len() - i;
    if remaining == 1 {
        let b0 = input[i] as u32;
        output.push(BASE64_CHARS[((b0 >> 2) & 0x3F) as usize] as char);
        output.push(BASE64_CHARS[((b0 << 4) & 0x3F) as usize] as char);
        output.push('=');
        output.push('=');
    } else if remaining == 2 {
        let b0 = input[i] as u32;
        let b1 = input[i + 1] as u32;
        output.push(BASE64_CHARS[((b0 >> 2) & 0x3F) as usize] as char);
        output.push(BASE64_CHARS[(((b0 << 4) | (b1 >> 4)) & 0x3F) as usize] as char);
        output.push(BASE64_CHARS[((b1 << 2) & 0x3F) as usize] as char);
        output.push('=');
    }
    output
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the position of the \r\n\r\n separator in a byte buffer.
fn find_header_end(buf: &[u8]) -> Option<usize> {
    for i in 0..buf.len().saturating_sub(3) {
        if buf[i] == b'\r' && buf[i + 1] == b'\n' && buf[i + 2] == b'\r' && buf[i + 3] == b'\n' {
            return Some(i);
        }
    }
    None
}

/// Check if there is a complete line (ending with \r\n) at the given position.
fn has_line_at(buf: &[u8], pos: usize) -> bool {
    find_line_end(buf, pos).is_some()
}

/// Find the position of the \r\n after the given starting position.
fn find_line_end(buf: &[u8], pos: usize) -> Option<usize> {
    for i in pos..buf.len().saturating_sub(1) {
        if buf[i] == b'\r' && buf[i + 1] == b'\n' {
            return Some(i);
        }
    }
    None
}

/// Parse an HTTP status line: "HTTP/1.1 200 OK" -> (version, code, reason).
fn parse_status_line(line: &str) -> HttpResult<(String, u16, String)> {
    let mut parts = line.splitn(3, ' ');
    let version = parts
        .next()
        .ok_or_else(|| HttpError::MalformedResponse("empty status line".to_string()))?
        .to_string();
    let code_str = parts
        .next()
        .ok_or_else(|| HttpError::MalformedResponse("missing status code".to_string()))?;
    let code: u16 = code_str.parse().map_err(|_| {
        HttpError::MalformedResponse(format!("invalid status code: {code_str}"))
    })?;
    let reason = parts.next().unwrap_or("").to_string();
    Ok((version, code, reason))
}

/// Resolve a redirect Location header against the current URL.
fn resolve_redirect_url(current: &str, location: &str) -> HttpResult<String> {
    // Absolute URL.
    if location.starts_with("http://") || location.starts_with("https://") {
        return Ok(location.to_string());
    }

    let parsed = ParsedUrl::parse(current)?;

    if location.starts_with("//") {
        // Protocol-relative URL.
        return Ok(format!("{}:{}", parsed.scheme, location));
    }

    if location.starts_with('/') {
        // Absolute path.
        return Ok(format!(
            "{}://{}{}",
            parsed.scheme,
            parsed.authority(),
            location
        ));
    }

    // Relative path — append to current directory.
    let base_path = if let Some(idx) = parsed.path.rfind('/') {
        &parsed.path[..idx + 1]
    } else {
        "/"
    };
    Ok(format!(
        "{}://{}{}{}",
        parsed.scheme,
        parsed.authority(),
        base_path,
        location
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_parsing_basic() {
        let url = ParsedUrl::parse("http://example.com/path?q=1#frag").unwrap();
        assert_eq!(url.scheme, "http");
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, 80);
        assert_eq!(url.path, "/path");
        assert_eq!(url.query.as_deref(), Some("q=1"));
        assert_eq!(url.fragment.as_deref(), Some("frag"));
    }

    #[test]
    fn test_url_parsing_with_port() {
        let url = ParsedUrl::parse("http://localhost:8080/api/v1").unwrap();
        assert_eq!(url.host, "localhost");
        assert_eq!(url.port, 8080);
        assert_eq!(url.path, "/api/v1");
    }

    #[test]
    fn test_url_parsing_no_path() {
        let url = ParsedUrl::parse("http://example.com").unwrap();
        assert_eq!(url.path, "/");
    }

    #[test]
    fn test_url_parsing_https_default_port() {
        let url = ParsedUrl::parse("https://secure.example.com/login").unwrap();
        assert_eq!(url.port, 443);
    }

    #[test]
    fn test_url_invalid() {
        assert!(ParsedUrl::parse("not-a-url").is_err());
        assert!(ParsedUrl::parse("ftp://example.com").is_err());
    }

    #[test]
    fn test_header_map() {
        let mut headers = HeaderMap::new();
        headers.set("Content-Type", "application/json");
        headers.set("Accept", "text/html");

        assert_eq!(headers.get("content-type"), Some("application/json"));
        assert_eq!(headers.get("ACCEPT"), Some("text/html"));
        assert!(headers.contains("Content-Type"));
        assert_eq!(headers.len(), 2);
    }

    #[test]
    fn test_header_replace() {
        let mut headers = HeaderMap::new();
        headers.set("X-Custom", "first");
        headers.set("x-custom", "second");
        assert_eq!(headers.get("X-Custom"), Some("second"));
        assert_eq!(headers.len(), 1);
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::new()
            .param("name", "John Doe")
            .param("age", "30")
            .build();
        assert_eq!(query, "name=John%20Doe&age=30");
    }

    #[test]
    fn test_percent_encode() {
        assert_eq!(percent_encode("hello world"), "hello%20world");
        assert_eq!(percent_encode("a+b=c"), "a%2Bb%3Dc");
        assert_eq!(percent_encode("safe-string_here.txt"), "safe-string_here.txt");
    }

    #[test]
    fn test_percent_decode() {
        assert_eq!(percent_decode("hello%20world"), "hello world");
        assert_eq!(percent_decode("a%2Bb%3Dc"), "a+b=c");
    }

    #[test]
    fn test_json_parse_null() {
        let v = parse_json("null").unwrap();
        assert!(v.is_null());
    }

    #[test]
    fn test_json_parse_bool() {
        assert_eq!(parse_json("true").unwrap().as_bool(), Some(true));
        assert_eq!(parse_json("false").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn test_json_parse_number() {
        assert_eq!(parse_json("42").unwrap().as_number(), Some(42.0));
        assert_eq!(parse_json("-3.14").unwrap().as_number(), Some(-3.14));
        assert_eq!(parse_json("1e10").unwrap().as_number(), Some(1e10));
    }

    #[test]
    fn test_json_parse_string() {
        assert_eq!(parse_json("\"hello\"").unwrap().as_str(), Some("hello"));
        assert_eq!(
            parse_json("\"line\\nbreak\"").unwrap().as_str(),
            Some("line\nbreak")
        );
    }

    #[test]
    fn test_json_parse_array() {
        let v = parse_json("[1, 2, 3]").unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_number(), Some(1.0));
    }

    #[test]
    fn test_json_parse_object() {
        let v = parse_json("{\"name\": \"test\", \"value\": 42}").unwrap();
        assert_eq!(v.get("name").unwrap().as_str(), Some("test"));
        assert_eq!(v.get("value").unwrap().as_number(), Some(42.0));
    }

    #[test]
    fn test_json_nested() {
        let v = parse_json("{\"data\": [1, {\"inner\": true}]}").unwrap();
        let data = v.get("data").unwrap().as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[1].get("inner").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn test_json_roundtrip() {
        let original = "{\"name\":\"test\",\"values\":[1,2,3],\"nested\":{\"ok\":true}}";
        let parsed = parse_json(original).unwrap();
        let serialized = parsed.to_json_string();
        let reparsed = parse_json(&serialized).unwrap();
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
        assert_eq!(base64_encode(b"Hi"), "SGk=");
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"abc"), "YWJj");
    }

    #[test]
    fn test_http_method() {
        assert_eq!(HttpMethod::Get.as_str(), "GET");
        assert!(HttpMethod::Post.has_body());
        assert!(!HttpMethod::Get.has_body());
        assert_eq!(HttpMethod::from_str("delete").unwrap(), HttpMethod::Delete);
    }

    #[test]
    fn test_redirect_resolution() {
        let base = "http://example.com/api/v1/users";
        assert_eq!(
            resolve_redirect_url(base, "http://other.com/new").unwrap(),
            "http://other.com/new"
        );
        assert_eq!(
            resolve_redirect_url(base, "/absolute/path").unwrap(),
            "http://example.com/absolute/path"
        );
        assert_eq!(
            resolve_redirect_url(base, "relative").unwrap(),
            "http://example.com/api/v1/relative"
        );
    }

    #[test]
    fn test_pool_stats_default() {
        let pool = ConnectionPool::new();
        let stats = pool.stats();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_created, 0);
    }

    #[test]
    fn test_http_response_status_categories() {
        let response = HttpResponse {
            http_version: "HTTP/1.1".to_string(),
            status_code: 200,
            reason_phrase: "OK".to_string(),
            headers: HeaderMap::new(),
            body: Vec::new(),
            elapsed: Duration::from_millis(50),
            final_url: "http://example.com".to_string(),
            redirect_count: 0,
        };
        assert!(response.is_success());
        assert!(!response.is_redirect());
        assert!(!response.is_client_error());
        assert!(!response.is_server_error());
    }

    #[test]
    fn test_header_map_parse() {
        let raw = "Content-Type: application/json\r\nX-Request-Id: abc123";
        let headers = HeaderMap::parse(raw).unwrap();
        assert_eq!(headers.get("Content-Type"), Some("application/json"));
        assert_eq!(headers.get("x-request-id"), Some("abc123"));
    }
}
