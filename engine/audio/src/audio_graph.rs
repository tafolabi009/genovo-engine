//! Audio graph / routing system with node-based signal processing.
//!
//! Provides a directed acyclic graph (DAG) of audio processing nodes.
//! Audio flows from source nodes through processing nodes (gain, filter,
//! delay, etc.) to the output. Nodes are connected by edges and processed
//! in topologically sorted order.
//!
//! Includes:
//! - `AudioNode` trait for custom processing nodes
//! - Built-in nodes: `MixerNode`, `GainNode`, `PanNode`, `FilterNode`,
//!   `DelayNode`, `SplitterNode`, `AnalyzerNode`
//! - `AudioGraphBuilder` for fluent graph construction
//! - `AudioGraph::process()` for per-frame graph evaluation

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default internal buffer size (samples per channel).
const DEFAULT_BUFFER_SIZE: usize = 512;
/// Maximum delay time in seconds.
const MAX_DELAY_TIME: f32 = 4.0;

// ---------------------------------------------------------------------------
// Node handle
// ---------------------------------------------------------------------------

/// Opaque handle to a node in the audio graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeHandle(pub u64);

// ---------------------------------------------------------------------------
// AudioNode trait
// ---------------------------------------------------------------------------

/// Trait for audio processing nodes in the graph.
///
/// Each node receives zero or more input buffers and writes to a single
/// output buffer. The graph evaluates nodes in topological order so that
/// all inputs are available when `process` is called.
pub trait AudioNode: Send + Sync {
    /// Process audio. `inputs` contains one buffer per input connection.
    /// `output` is the buffer to write the result into.
    /// `sample_rate` is the current output sample rate.
    fn process(&mut self, inputs: &[&[f32]], output: &mut [f32], sample_rate: u32);

    /// The name of this node type (for debugging / UI).
    fn name(&self) -> &str;

    /// Reset internal state.
    fn reset(&mut self);

    /// Number of expected input connections (0 for source nodes).
    fn input_count(&self) -> usize {
        1
    }
}

// ===========================================================================
// Built-in nodes
// ===========================================================================

// ---------------------------------------------------------------------------
// MixerNode
// ---------------------------------------------------------------------------

/// Sums N inputs with per-input volume control.
pub struct MixerNode {
    /// Volume per input [0, 1+].
    pub input_volumes: Vec<f32>,
    /// Number of inputs.
    num_inputs: usize,
}

impl MixerNode {
    /// Create a new mixer node with the given number of inputs.
    pub fn new(num_inputs: usize) -> Self {
        Self {
            input_volumes: vec![1.0; num_inputs],
            num_inputs,
        }
    }

    /// Set the volume for a specific input.
    pub fn set_input_volume(&mut self, input_index: usize, volume: f32) {
        if input_index < self.input_volumes.len() {
            self.input_volumes[input_index] = volume;
        }
    }
}

impl AudioNode for MixerNode {
    fn process(&mut self, inputs: &[&[f32]], output: &mut [f32], _sample_rate: u32) {
        // Zero the output
        for s in output.iter_mut() {
            *s = 0.0;
        }

        // Sum all inputs with their volumes
        for (i, input) in inputs.iter().enumerate() {
            let vol = if i < self.input_volumes.len() {
                self.input_volumes[i]
            } else {
                1.0
            };

            for (j, &sample) in input.iter().enumerate() {
                if j < output.len() {
                    output[j] += sample * vol;
                }
            }
        }
    }

    fn name(&self) -> &str {
        "MixerNode"
    }

    fn reset(&mut self) {}

    fn input_count(&self) -> usize {
        self.num_inputs
    }
}

// ---------------------------------------------------------------------------
// GainNode
// ---------------------------------------------------------------------------

/// Multiplies the signal by a constant gain factor.
pub struct GainNode {
    /// Gain multiplier.
    pub gain: f32,
}

impl GainNode {
    /// Create a new gain node.
    pub fn new(gain: f32) -> Self {
        Self { gain }
    }
}

impl AudioNode for GainNode {
    fn process(&mut self, inputs: &[&[f32]], output: &mut [f32], _sample_rate: u32) {
        if let Some(input) = inputs.first() {
            for (i, &s) in input.iter().enumerate() {
                if i < output.len() {
                    output[i] = s * self.gain;
                }
            }
        } else {
            for s in output.iter_mut() {
                *s = 0.0;
            }
        }
    }

    fn name(&self) -> &str {
        "GainNode"
    }

    fn reset(&mut self) {}
}

// ---------------------------------------------------------------------------
// PanNode
// ---------------------------------------------------------------------------

/// Stereo panning node. Takes a mono input and outputs stereo.
///
/// Uses constant-power panning:
///   left  = cos(angle) * input
///   right = sin(angle) * input
/// where angle = (pan + 1) / 2 * pi/2
pub struct PanNode {
    /// Pan position [-1 left, 0 center, +1 right].
    pub pan: f32,
}

impl PanNode {
    /// Create a new pan node.
    pub fn new(pan: f32) -> Self {
        Self {
            pan: pan.clamp(-1.0, 1.0),
        }
    }

    /// Set the pan position.
    pub fn set_pan(&mut self, pan: f32) {
        self.pan = pan.clamp(-1.0, 1.0);
    }
}

impl AudioNode for PanNode {
    fn process(&mut self, inputs: &[&[f32]], output: &mut [f32], _sample_rate: u32) {
        let angle = ((self.pan + 1.0) * 0.5) * std::f32::consts::FRAC_PI_2;
        let gain_l = angle.cos();
        let gain_r = angle.sin();

        if let Some(input) = inputs.first() {
            // Output is stereo interleaved: [L, R, L, R, ...]
            let frames = output.len() / 2;
            for i in 0..frames {
                let mono_sample = if i < input.len() { input[i] } else { 0.0 };
                output[i * 2] = mono_sample * gain_l;
                output[i * 2 + 1] = mono_sample * gain_r;
            }
        } else {
            for s in output.iter_mut() {
                *s = 0.0;
            }
        }
    }

    fn name(&self) -> &str {
        "PanNode"
    }

    fn reset(&mut self) {}
}

// ---------------------------------------------------------------------------
// FilterNode
// ---------------------------------------------------------------------------

/// Filter type for the FilterNode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    LowPass,
    HighPass,
}

/// Simple first-order IIR filter node.
pub struct FilterNode {
    /// Filter type.
    pub filter_type: FilterType,
    /// Cutoff frequency in Hz.
    pub cutoff: f32,
    /// Previous output (filter state).
    prev_output: f32,
    /// Previous input (for high-pass).
    prev_input: f32,
}

impl FilterNode {
    /// Create a new filter node.
    pub fn new(filter_type: FilterType, cutoff: f32) -> Self {
        Self {
            filter_type,
            cutoff: cutoff.max(1.0),
            prev_output: 0.0,
            prev_input: 0.0,
        }
    }
}

impl AudioNode for FilterNode {
    fn process(&mut self, inputs: &[&[f32]], output: &mut [f32], sample_rate: u32) {
        let input = match inputs.first() {
            Some(i) => *i,
            None => {
                for s in output.iter_mut() {
                    *s = 0.0;
                }
                return;
            }
        };

        let tau = std::f32::consts::TAU;
        let dt = 1.0 / sample_rate as f32;
        let rc = 1.0 / (tau * self.cutoff);

        match self.filter_type {
            FilterType::LowPass => {
                let alpha = dt / (rc + dt);
                for (i, &s) in input.iter().enumerate() {
                    if i < output.len() {
                        self.prev_output = alpha * s + (1.0 - alpha) * self.prev_output;
                        output[i] = self.prev_output;
                    }
                }
            }
            FilterType::HighPass => {
                let alpha = rc / (rc + dt);
                for (i, &s) in input.iter().enumerate() {
                    if i < output.len() {
                        self.prev_output = alpha * (self.prev_output + s - self.prev_input);
                        self.prev_input = s;
                        output[i] = self.prev_output;
                    }
                }
            }
        }
    }

    fn name(&self) -> &str {
        match self.filter_type {
            FilterType::LowPass => "LowPassFilterNode",
            FilterType::HighPass => "HighPassFilterNode",
        }
    }

    fn reset(&mut self) {
        self.prev_output = 0.0;
        self.prev_input = 0.0;
    }
}

// ---------------------------------------------------------------------------
// DelayNode
// ---------------------------------------------------------------------------

/// Circular buffer delay node.
pub struct DelayNode {
    /// Delay time in seconds.
    pub delay_time: f32,
    /// Feedback amount [0, 1).
    pub feedback: f32,
    /// Wet/dry mix [0, 1].
    pub mix: f32,
    /// Circular buffer.
    buffer: Vec<f32>,
    /// Write position.
    write_pos: usize,
}

impl DelayNode {
    /// Create a new delay node.
    pub fn new(delay_time: f32, feedback: f32, mix: f32) -> Self {
        let max_samples = (MAX_DELAY_TIME * 48000.0) as usize;
        Self {
            delay_time: delay_time.clamp(0.0, MAX_DELAY_TIME),
            feedback: feedback.clamp(0.0, 0.99),
            mix: mix.clamp(0.0, 1.0),
            buffer: vec![0.0; max_samples],
            write_pos: 0,
        }
    }
}

impl AudioNode for DelayNode {
    fn process(&mut self, inputs: &[&[f32]], output: &mut [f32], sample_rate: u32) {
        let input = match inputs.first() {
            Some(i) => *i,
            None => {
                for s in output.iter_mut() {
                    *s = 0.0;
                }
                return;
            }
        };

        let delay_samples = (self.delay_time * sample_rate as f32) as usize;
        let buf_len = self.buffer.len();

        if delay_samples == 0 || delay_samples >= buf_len {
            // Pass through
            for (i, &s) in input.iter().enumerate() {
                if i < output.len() {
                    output[i] = s;
                }
            }
            return;
        }

        for (i, &s) in input.iter().enumerate() {
            if i >= output.len() {
                break;
            }

            let read_pos = (self.write_pos + buf_len - delay_samples) % buf_len;
            let delayed = self.buffer[read_pos];

            self.buffer[self.write_pos] = s + delayed * self.feedback;
            output[i] = s * (1.0 - self.mix) + delayed * self.mix;

            self.write_pos = (self.write_pos + 1) % buf_len;
        }
    }

    fn name(&self) -> &str {
        "DelayNode"
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// SplitterNode
// ---------------------------------------------------------------------------

/// Duplicates its input to the output without modification.
/// Multiple edges from a splitter's output connect it to multiple consumers.
pub struct SplitterNode;

impl SplitterNode {
    pub fn new() -> Self {
        Self
    }
}

impl AudioNode for SplitterNode {
    fn process(&mut self, inputs: &[&[f32]], output: &mut [f32], _sample_rate: u32) {
        if let Some(input) = inputs.first() {
            for (i, &s) in input.iter().enumerate() {
                if i < output.len() {
                    output[i] = s;
                }
            }
        } else {
            for s in output.iter_mut() {
                *s = 0.0;
            }
        }
    }

    fn name(&self) -> &str {
        "SplitterNode"
    }

    fn reset(&mut self) {}
}

// ---------------------------------------------------------------------------
// AnalyzerNode
// ---------------------------------------------------------------------------

/// Analyzer node that passes audio through while computing RMS, peak, and
/// optionally a simple FFT spectrum.
pub struct AnalyzerNode {
    /// Current RMS level.
    pub rms: f32,
    /// Current peak level.
    pub peak: f32,
    /// FFT spectrum (magnitude bins). Length = fft_size / 2.
    pub spectrum: Vec<f32>,
    /// FFT size (power of 2).
    pub fft_size: usize,
    /// Accumulation buffer for FFT.
    fft_buffer: Vec<f32>,
    /// Write position in the FFT buffer.
    fft_write_pos: usize,
    /// Whether to compute FFT each frame.
    pub compute_fft: bool,
}

impl AnalyzerNode {
    /// Create a new analyzer node.
    pub fn new(fft_size: usize) -> Self {
        let fft_size = fft_size.next_power_of_two().max(64);
        Self {
            rms: 0.0,
            peak: 0.0,
            spectrum: vec![0.0; fft_size / 2],
            fft_size,
            fft_buffer: vec![0.0; fft_size],
            fft_write_pos: 0,
            compute_fft: false,
        }
    }

    /// Compute a simple DFT magnitude spectrum (not a proper FFT, but
    /// sufficient for visualization at small sizes).
    fn compute_dft(&mut self) {
        let n = self.fft_size;
        let half = n / 2;

        for k in 0..half {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            let freq = std::f32::consts::TAU * k as f32 / n as f32;

            for i in 0..n {
                let angle = freq * i as f32;
                real += self.fft_buffer[i] * angle.cos();
                imag -= self.fft_buffer[i] * angle.sin();
            }

            self.spectrum[k] = (real * real + imag * imag).sqrt() / n as f32;
        }
    }
}

impl AudioNode for AnalyzerNode {
    fn process(&mut self, inputs: &[&[f32]], output: &mut [f32], _sample_rate: u32) {
        let input = match inputs.first() {
            Some(i) => *i,
            None => {
                for s in output.iter_mut() {
                    *s = 0.0;
                }
                self.rms = 0.0;
                self.peak = 0.0;
                return;
            }
        };

        // Pass through
        let mut sum_sq = 0.0f32;
        let mut peak = 0.0f32;

        for (i, &s) in input.iter().enumerate() {
            if i < output.len() {
                output[i] = s;
            }
            sum_sq += s * s;
            let abs = s.abs();
            if abs > peak {
                peak = abs;
            }

            // Fill FFT buffer
            if self.compute_fft {
                self.fft_buffer[self.fft_write_pos] = s;
                self.fft_write_pos = (self.fft_write_pos + 1) % self.fft_size;
            }
        }

        let count = input.len().max(1) as f32;
        self.rms = (sum_sq / count).sqrt();
        self.peak = peak;

        // Compute FFT if buffer is full
        if self.compute_fft && self.fft_write_pos == 0 {
            self.compute_dft();
        }
    }

    fn name(&self) -> &str {
        "AnalyzerNode"
    }

    fn reset(&mut self) {
        self.rms = 0.0;
        self.peak = 0.0;
        self.fft_buffer.fill(0.0);
        self.fft_write_pos = 0;
        self.spectrum.fill(0.0);
    }
}

// ===========================================================================
// AudioGraph
// ===========================================================================

/// An edge in the audio graph (connection between nodes).
#[derive(Debug, Clone)]
struct GraphEdge {
    /// Source node handle.
    from: NodeHandle,
    /// Destination node handle.
    to: NodeHandle,
    /// Input index on the destination node.
    input_index: usize,
}

/// Internal node data.
struct GraphNode {
    handle: NodeHandle,
    node: Box<dyn AudioNode>,
    /// Output buffer (filled during processing).
    output_buffer: Vec<f32>,
}

/// Node-based audio routing graph.
///
/// Nodes are connected in a directed acyclic graph (DAG). When `process()`
/// is called, nodes are evaluated in topologically sorted order so that
/// each node's inputs are available before it runs.
pub struct AudioGraph {
    /// All nodes in the graph.
    nodes: HashMap<NodeHandle, GraphNode>,
    /// All edges (connections).
    edges: Vec<GraphEdge>,
    /// The output node (final destination).
    output_node: Option<NodeHandle>,
    /// Next handle ID.
    next_id: u64,
    /// Buffer size for internal processing.
    buffer_size: usize,
    /// Sample rate.
    pub sample_rate: u32,
    /// Cached topological order (invalidated when graph changes).
    topo_order: Vec<NodeHandle>,
    /// Whether the topology needs to be recomputed.
    dirty: bool,
}

impl AudioGraph {
    /// Create a new audio graph.
    pub fn new(sample_rate: u32, buffer_size: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            output_node: None,
            next_id: 1,
            buffer_size,
            sample_rate,
            topo_order: Vec::new(),
            dirty: true,
        }
    }

    /// Add a node to the graph and return its handle.
    pub fn add_node(&mut self, node: Box<dyn AudioNode>) -> NodeHandle {
        let handle = NodeHandle(self.next_id);
        self.next_id += 1;

        let graph_node = GraphNode {
            handle,
            node,
            output_buffer: vec![0.0; self.buffer_size],
        };

        self.nodes.insert(handle, graph_node);
        self.dirty = true;
        handle
    }

    /// Remove a node and all its connections.
    pub fn remove_node(&mut self, handle: NodeHandle) {
        self.nodes.remove(&handle);
        self.edges.retain(|e| e.from != handle && e.to != handle);
        if self.output_node == Some(handle) {
            self.output_node = None;
        }
        self.dirty = true;
    }

    /// Connect two nodes: `from`'s output feeds into `to`'s input at `input_index`.
    pub fn connect(&mut self, from: NodeHandle, to: NodeHandle, input_index: usize) {
        self.edges.push(GraphEdge {
            from,
            to,
            input_index,
        });
        self.dirty = true;
    }

    /// Disconnect a specific edge.
    pub fn disconnect(&mut self, from: NodeHandle, to: NodeHandle) {
        self.edges.retain(|e| !(e.from == from && e.to == to));
        self.dirty = true;
    }

    /// Set the output (final destination) node.
    pub fn set_output(&mut self, handle: NodeHandle) {
        self.output_node = Some(handle);
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get a reference to a node by handle.
    pub fn get_node(&self, handle: NodeHandle) -> Option<&dyn AudioNode> {
        self.nodes.get(&handle).map(|n| &*n.node)
    }

    // -----------------------------------------------------------------------
    // Topological sort
    // -----------------------------------------------------------------------

    /// Compute the topological evaluation order using Kahn's algorithm.
    fn topological_sort(&mut self) {
        let mut in_degree: HashMap<NodeHandle, usize> = HashMap::new();
        let mut adj: HashMap<NodeHandle, Vec<NodeHandle>> = HashMap::new();

        // Initialize
        for &handle in self.nodes.keys() {
            in_degree.insert(handle, 0);
            adj.insert(handle, Vec::new());
        }

        // Build adjacency and in-degree
        for edge in &self.edges {
            if let Some(neighbors) = adj.get_mut(&edge.from) {
                neighbors.push(edge.to);
            }
            if let Some(deg) = in_degree.get_mut(&edge.to) {
                *deg += 1;
            }
        }

        // Find all nodes with in-degree 0
        let mut queue: VecDeque<NodeHandle> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(handle, _)| *handle)
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(node) = queue.pop_front() {
            order.push(node);

            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(&neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        self.topo_order = order;
        self.dirty = false;
    }

    // -----------------------------------------------------------------------
    // Process
    // -----------------------------------------------------------------------

    /// Process the entire audio graph for one frame.
    ///
    /// Evaluates all nodes in topological order and writes the result
    /// to `output_buffer`.
    pub fn process(&mut self, output_buffer: &mut [f32]) {
        if self.dirty {
            self.topological_sort();
        }

        let buffer_size = self.buffer_size;
        let sample_rate = self.sample_rate;

        // Process each node in topological order
        let order = self.topo_order.clone();
        for &handle in &order {
            // Collect input buffers for this node
            let input_edges: Vec<(usize, NodeHandle)> = self
                .edges
                .iter()
                .filter(|e| e.to == handle)
                .map(|e| (e.input_index, e.from))
                .collect();

            // Build the input references
            // We need to collect the data into owned vecs first to avoid
            // borrow conflicts with self.nodes
            let input_data: Vec<Vec<f32>> = input_edges
                .iter()
                .map(|(_, from_handle)| {
                    self.nodes
                        .get(from_handle)
                        .map(|n| n.output_buffer.clone())
                        .unwrap_or_else(|| vec![0.0; buffer_size])
                })
                .collect();

            let input_refs: Vec<&[f32]> = input_data.iter().map(|v| v.as_slice()).collect();

            // Process the node
            if let Some(graph_node) = self.nodes.get_mut(&handle) {
                graph_node.output_buffer.resize(buffer_size, 0.0);
                graph_node
                    .node
                    .process(&input_refs, &mut graph_node.output_buffer, sample_rate);
            }
        }

        // Copy the output node's buffer to the output
        if let Some(output_handle) = self.output_node {
            if let Some(output_node) = self.nodes.get(&output_handle) {
                let len = output_buffer.len().min(output_node.output_buffer.len());
                output_buffer[..len].copy_from_slice(&output_node.output_buffer[..len]);
            }
        } else {
            for s in output_buffer.iter_mut() {
                *s = 0.0;
            }
        }
    }

    /// Reset all nodes in the graph.
    pub fn reset(&mut self) {
        for node in self.nodes.values_mut() {
            node.node.reset();
            node.output_buffer.fill(0.0);
        }
    }
}

// ===========================================================================
// AudioGraphBuilder
// ===========================================================================

/// Fluent builder API for constructing audio graphs.
pub struct AudioGraphBuilder {
    graph: AudioGraph,
}

impl AudioGraphBuilder {
    /// Start building a new audio graph.
    pub fn new(sample_rate: u32, buffer_size: usize) -> Self {
        Self {
            graph: AudioGraph::new(sample_rate, buffer_size),
        }
    }

    /// Add a node and return its handle.
    pub fn add_node(&mut self, node: Box<dyn AudioNode>) -> NodeHandle {
        self.graph.add_node(node)
    }

    /// Connect two nodes.
    pub fn connect(
        &mut self,
        from: NodeHandle,
        to: NodeHandle,
        input_index: usize,
    ) -> &mut Self {
        self.graph.connect(from, to, input_index);
        self
    }

    /// Set the output node.
    pub fn set_output(&mut self, handle: NodeHandle) -> &mut Self {
        self.graph.set_output(handle);
        self
    }

    /// Build and return the audio graph.
    pub fn build(self) -> AudioGraph {
        self.graph
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Source node that generates a constant value.
    struct ConstantSource {
        value: f32,
    }

    impl ConstantSource {
        fn new(value: f32) -> Self {
            Self { value }
        }
    }

    impl AudioNode for ConstantSource {
        fn process(&mut self, _inputs: &[&[f32]], output: &mut [f32], _sample_rate: u32) {
            for s in output.iter_mut() {
                *s = self.value;
            }
        }

        fn name(&self) -> &str {
            "ConstantSource"
        }

        fn reset(&mut self) {}

        fn input_count(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_gain_node() {
        let mut gain = GainNode::new(0.5);
        let input = vec![1.0f32; 10];
        let mut output = vec![0.0f32; 10];

        gain.process(&[&input], &mut output, 44100);

        for &s in &output {
            assert!((s - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mixer_node() {
        let mut mixer = MixerNode::new(2);
        let input_a = vec![0.3f32; 10];
        let input_b = vec![0.7f32; 10];
        let mut output = vec![0.0f32; 10];

        mixer.process(&[&input_a, &input_b], &mut output, 44100);

        for &s in &output {
            assert!((s - 1.0).abs() < 1e-6, "Expected 1.0, got {}", s);
        }
    }

    #[test]
    fn test_mixer_node_with_volumes() {
        let mut mixer = MixerNode::new(2);
        mixer.set_input_volume(0, 0.5);
        mixer.set_input_volume(1, 0.5);

        let input_a = vec![1.0f32; 10];
        let input_b = vec![1.0f32; 10];
        let mut output = vec![0.0f32; 10];

        mixer.process(&[&input_a, &input_b], &mut output, 44100);

        for &s in &output {
            assert!((s - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_pan_node_center() {
        let mut pan = PanNode::new(0.0);
        let input = vec![1.0f32; 5];
        let mut output = vec![0.0f32; 10]; // Stereo

        pan.process(&[&input], &mut output, 44100);

        // At center pan, both channels should have equal energy
        let left = output[0];
        let right = output[1];
        assert!(
            (left - right).abs() < 0.01,
            "Center pan: L={}, R={}",
            left,
            right
        );
    }

    #[test]
    fn test_pan_node_hard_left() {
        let mut pan = PanNode::new(-1.0);
        let input = vec![1.0f32; 5];
        let mut output = vec![0.0f32; 10];

        pan.process(&[&input], &mut output, 44100);

        assert!(output[0] > 0.9, "Left channel should be loud");
        assert!(output[1] < 0.01, "Right channel should be silent");
    }

    #[test]
    fn test_filter_node_lowpass() {
        let mut filter = FilterNode::new(FilterType::LowPass, 100.0);

        // Generate a high-frequency signal
        let sr = 44100;
        let input: Vec<f32> = (0..1000)
            .map(|i| (i as f32 / sr as f32 * 5000.0 * std::f32::consts::TAU).sin())
            .collect();
        let mut output = vec![0.0f32; 1000];

        filter.process(&[&input], &mut output, sr);

        // Output energy should be less than input energy
        let input_energy: f32 = input.iter().map(|s| s * s).sum();
        let output_energy: f32 = output.iter().map(|s| s * s).sum();
        assert!(output_energy < input_energy * 0.5);
    }

    #[test]
    fn test_delay_node() {
        let mut delay = DelayNode::new(0.01, 0.0, 1.0);
        let mut input = vec![0.0f32; 1000];
        input[0] = 1.0;
        let mut output = vec![0.0f32; 1000];

        delay.process(&[&input], &mut output, 44100);

        // Delayed impulse should appear at ~441 samples
        assert!(output[441].abs() > 0.5, "Delayed signal at 441: {}", output[441]);
    }

    #[test]
    fn test_analyzer_node() {
        let mut analyzer = AnalyzerNode::new(256);
        let input: Vec<f32> = (0..256)
            .map(|i| (i as f32 / 256.0 * std::f32::consts::TAU * 10.0).sin() * 0.8)
            .collect();
        let mut output = vec![0.0f32; 256];

        analyzer.process(&[&input], &mut output, 44100);

        assert!(analyzer.rms > 0.0);
        assert!(analyzer.peak > 0.0);
        assert!(analyzer.peak <= 0.8 + 0.01);

        // Output should be pass-through
        for (i, &s) in input.iter().enumerate() {
            assert!((output[i] - s).abs() < 1e-6);
        }
    }

    #[test]
    fn test_splitter_node() {
        let mut splitter = SplitterNode::new();
        let input = vec![0.5f32; 10];
        let mut output = vec![0.0f32; 10];

        splitter.process(&[&input], &mut output, 44100);

        for &s in &output {
            assert!((s - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simple_graph() {
        let mut builder = AudioGraphBuilder::new(44100, 256);

        // Source -> Gain -> Output
        let source = builder.add_node(Box::new(ConstantSource::new(1.0)));
        let gain = builder.add_node(Box::new(GainNode::new(0.5)));

        builder.connect(source, gain, 0);
        builder.set_output(gain);

        let mut graph = builder.build();
        let mut output = vec![0.0f32; 256];

        graph.process(&mut output);

        for &s in &output {
            assert!((s - 0.5).abs() < 1e-4, "Expected 0.5, got {}", s);
        }
    }

    #[test]
    fn test_graph_mixer() {
        let mut builder = AudioGraphBuilder::new(44100, 128);

        let src_a = builder.add_node(Box::new(ConstantSource::new(0.3)));
        let src_b = builder.add_node(Box::new(ConstantSource::new(0.7)));
        let mixer = builder.add_node(Box::new(MixerNode::new(2)));

        builder.connect(src_a, mixer, 0);
        builder.connect(src_b, mixer, 1);
        builder.set_output(mixer);

        let mut graph = builder.build();
        let mut output = vec![0.0f32; 128];

        graph.process(&mut output);

        for &s in &output {
            assert!((s - 1.0).abs() < 1e-4, "Expected 1.0, got {}", s);
        }
    }

    #[test]
    fn test_graph_chain() {
        let mut builder = AudioGraphBuilder::new(44100, 64);

        // Source -> Gain(2.0) -> Gain(0.25) -> Output
        // Result should be 1.0 * 2.0 * 0.25 = 0.5
        let src = builder.add_node(Box::new(ConstantSource::new(1.0)));
        let gain1 = builder.add_node(Box::new(GainNode::new(2.0)));
        let gain2 = builder.add_node(Box::new(GainNode::new(0.25)));

        builder.connect(src, gain1, 0);
        builder.connect(gain1, gain2, 0);
        builder.set_output(gain2);

        let mut graph = builder.build();
        let mut output = vec![0.0f32; 64];

        graph.process(&mut output);

        for &s in &output {
            assert!((s - 0.5).abs() < 1e-4);
        }
    }

    #[test]
    fn test_graph_reset() {
        let mut graph = AudioGraph::new(44100, 64);
        let src = graph.add_node(Box::new(ConstantSource::new(1.0)));
        graph.set_output(src);

        let mut output = vec![0.0f32; 64];
        graph.process(&mut output);
        assert!(output[0] > 0.9);

        graph.reset();
        // Should still work after reset
        graph.process(&mut output);
        assert!(output[0] > 0.9);
    }

    #[test]
    fn test_graph_node_count() {
        let mut graph = AudioGraph::new(44100, 64);
        assert_eq!(graph.node_count(), 0);

        let h = graph.add_node(Box::new(GainNode::new(1.0)));
        assert_eq!(graph.node_count(), 1);

        graph.remove_node(h);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_topological_sort_order() {
        let mut graph = AudioGraph::new(44100, 32);

        let a = graph.add_node(Box::new(ConstantSource::new(1.0)));
        let b = graph.add_node(Box::new(GainNode::new(1.0)));
        let c = graph.add_node(Box::new(GainNode::new(1.0)));

        graph.connect(a, b, 0);
        graph.connect(b, c, 0);

        graph.topological_sort();

        // a should come before b, b before c
        let pos_a = graph.topo_order.iter().position(|&h| h == a).unwrap();
        let pos_b = graph.topo_order.iter().position(|&h| h == b).unwrap();
        let pos_c = graph.topo_order.iter().position(|&h| h == c).unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }
}
