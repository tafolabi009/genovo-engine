# Genovo Engine - Development Guide

## Project Overview
AAA-tier game engine built primarily in Rust with C++ interop. Cargo workspace with 14+ crates.

## Build Commands
```bash
cargo build                    # Dev build
cargo build --release          # Release build
cargo test --workspace         # Run all tests
cargo bench -p genovo-benchmarks  # Run benchmarks
cargo clippy --workspace       # Lint
cargo doc --workspace --no-deps   # Generate docs
```

## Architecture
- `engine/` - Core engine crates (Rust)
- `crates/genovo-ffi/` - C++ interop layer
- `tools/` - Build tools (shader compiler, asset cooker)
- `tests/` - Integration tests and benchmarks
- `examples/` - Demo applications

## Conventions
- TODO markers: `// TODO(CATEGORY): Description - Timeline`
- Categories: CORE, RENDER, PLATFORM, PHYSICS, AUDIO, ANIMATION, ASSETS, SCRIPTING, NETWORKING, AI, EDITOR, FFI, BUILD, OPTIMIZATION
- All public traits must have full method signatures
- Use `thiserror` for error types
- Use `profiling` crate for instrumentation
- Prefer `parking_lot` over `std::sync`

## Module Dependencies (directed, no cycles)
core → (no deps)
ecs → core
scene → core, ecs
render → core, ecs
platform → core, render
physics → core, ecs
audio → core, ecs
animation → core, ecs
assets → core
scripting → core, ecs
networking → core, ecs
ai → core, ecs
editor → core, ecs, scene, render, assets
ffi → core, ecs, physics, audio
