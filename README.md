# Genovo Engine

A AAA-tier game engine built in Rust — **491,000+ lines** across 28+ modules with real implementations of rendering, physics, AI, networking, scripting, and more.

## Quick Start

```rust
use genovo::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = Engine::new(EngineConfig {
        app_name: "My Game".to_string(),
        ..Default::default()
    })?;

    let world = engine.world_mut();
    world.spawn_entity()
        .with(TransformComponent::default())
        .build();

    engine.run();
    Ok(())
}
```

## Architecture

```
genovo_engine/
├── engine/
│   ├── core/           # Math, SIMD (SSE/AVX/NEON), memory, threading, noise, color, curves, spatial structures
│   ├── ecs/            # Archetype-based ECS with change detection, parallel scheduling
│   ├── scene/          # Scene graph, hierarchical transforms, prefabs
│   ├── render/         # PBR, deferred, ray tracing, post-processing, particles, terrain, ocean, sky
│   ├── platform/       # Cross-platform windowing (winit), input action mapping, gesture detection
│   ├── physics/        # Rigid body, cloth, soft body, fluid, vehicles, ragdoll, destruction
│   ├── audio/          # Software mixer, spatial audio, DSP effects, adaptive music, audio graph
│   ├── animation/      # Skeletal, IK solvers, blend trees, state machines, morph targets
│   ├── assets/         # glTF 2.0, TGA, HDR, DDS, OBJ, WAV, TrueType fonts, asset pipeline
│   ├── scripting/      # Custom bytecode VM with lexer/parser/compiler, debugger, optimizer
│   ├── networking/     # Reliable UDP, replication, prediction, RPC, lobby, matchmaking
│   ├── ai/             # A*, behavior trees, navmesh, GOAP, utility AI, steering, influence maps
│   ├── editor/         # Viewport, node graph, material editor, curve editor, terrain tools
│   ├── ui/             # 24 widgets, flexbox/grid layout, text rendering, theming
│   ├── debug/          # CPU/GPU profiler, developer console, debug draw, memory profiler
│   ├── terrain/        # Heightmap, hydraulic erosion, CDLOD LOD, splatmap texturing, vegetation
│   ├── procgen/        # Wave Function Collapse, dungeon generation, L-systems, mazes, name gen
│   ├── cinematics/     # Sequencer, cinematic camera, timeline editor
│   ├── localization/   # ICU pluralization (30+ languages), number/date formatting
│   ├── gameplay/       # Character controller, inventory, quests, dialogue, damage system
│   ├── world/          # World partition, async streaming, LOD management, splines
│   ├── replay/         # Frame-accurate replay recording/playback with kill cam
│   ├── save_system/    # Save slots, auto-save, checksum verification
│   └── genovo/         # Top-level engine facade integrating all subsystems
├── crates/
│   └── genovo-ffi/     # C/C++ FFI (54+ extern functions), Python/C# binding generators
├── tools/              # Shader compiler, asset cooker, platform build scripts
├── tests/              # Integration tests and benchmarks
├── examples/           # Demo applications
└── docs/               # Architecture documentation
```

## Module Highlights

### Rendering (55,000+ lines)
- **PBR**: Cook-Torrance BRDF, GGX NDF, Schlick Fresnel, metallic-roughness workflow
- **Deferred pipeline**: G-Buffer, light volumes, stencil optimization
- **Shadows**: Cascaded shadow maps, PCF, PCSS, Variance shadow maps, Virtual shadow maps with page management
- **Post-processing**: Bloom, SSAO, SSR, Depth of Field, Motion Blur, FXAA, TAA, Color Grading, Film Grain
- **Ray tracing**: BVH with SAH construction, Moller-Trumbore intersection, screen probes, path tracer
- **Lumen-style GI**: Screen-space probes, surface cache, multi-bounce, spherical harmonics
- **Virtual geometry**: Nanite-style mesh clusters, QEM simplification, streaming
- **GPU-driven**: Indirect draw, Hi-Z occlusion culling, depth pyramid
- **Atmosphere**: Rayleigh/Mie scattering, transmittance LUT, day/night cycle, stars, clouds
- **Ocean**: FFT-based (Tessendorf/Phillips spectrum), Gerstner waves, foam, underwater effects
- **Particles**: SoA pool, 7 force fields, curl noise, collision, trails, decals, volumetric fog
- **Upscaling**: FSR-style spatial (EASU/RCAS) + temporal upscaling
- **Advanced shading**: Subsurface scattering (skin), Marschner hair model

### Physics (18,000+ lines)
- Rigid body dynamics with SAT collision and sequential impulse solver
- Cloth simulation (Verlet integration, Jakobsen constraints, tearing)
- Soft body (Finite Element Method with co-rotational elasticity)
- SPH fluid simulation with marching cubes surface extraction
- Vehicle physics (Pacejka tire model, Ackermann steering, ABS/TCS)
- Ragdoll with humanoid preset and partial blend
- Voronoi fracture destruction
- Rope/chain physics, buoyancy, continuous collision detection

### AI (11,600+ lines)
- A* and HPA* pathfinding with grid graphs
- Behavior trees (15 node types, fluent builder API)
- Navigation mesh with funnel algorithm
- ORCA crowd simulation
- Utility AI with response curves
- GOAP (Goal-Oriented Action Planning) with A* world-state search
- Influence maps with strategic queries
- 14 steering behaviors (seek, flee, arrive, pursue, wander, flock, hide...)

### Scripting (9,800+ lines)
- Custom stack-based bytecode VM with 40+ opcodes
- Lexer, recursive-descent parser, single-pass compiler
- 80+ standard library functions (math, string, array, map)
- Script debugger with breakpoints, stepping, variable inspection
- Bytecode optimizer (constant folding, dead code elimination, peephole)

### Networking (14,500+ lines)
- Reliable UDP transport with EWMA RTT estimation
- Delta-compressed state replication
- Client-side prediction with rollback reconciliation
- Lag compensation with historical state rewind
- RPC system with batching and rate limiting
- Lobby system with host migration
- ELO-based matchmaking
- Area-of-interest spatial sync

## Language Mix

| Language | Lines | Purpose |
|----------|-------|---------|
| Rust | 253,000+ | Core engine — all 26 modules |
| WGSL | Embedded | GPU shaders (PBR, post-processing, compute) |
| C FFI | 8,100+ | 54+ `extern "C"` functions for C/C++ interop |
| SIMD ASM | ~2,000 | SSE4.1/AVX2/NEON intrinsics for math hot paths |

## Building

```bash
# Check entire workspace (26 crates)
cargo check --workspace

# Full build
cargo build --workspace

# Release build
cargo build --workspace --release

# Run tests
cargo test --workspace

# Run example
cargo run -p genovo-examples --example hello_triangle

# Generate C/C++ headers (FFI)
cargo build -p genovo-ffi
```

## Platform Support

| Platform | Status |
|----------|--------|
| Windows | Primary (winit + wgpu) |
| macOS | Supported (winit + wgpu/Metal) |
| Linux | Supported (winit + wgpu/Vulkan) |
| iOS | Stubs with UIKit FFI signatures |
| Android | Stubs with NDK FFI signatures |
| Xbox | Stub (requires GDK) |
| PlayStation | Stub (requires PS SDK under NDA) |

## License

Dual-licensed under MIT and Apache 2.0.
