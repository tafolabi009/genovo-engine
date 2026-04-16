# Getting Started with Genovo Engine

This guide walks you through setting up the development environment, building the engine, and running the example projects.

## Prerequisites

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| **Rust** | 1.85+ (2024 edition) | Primary language toolchain |
| **CMake** | 3.20+ | C++ FFI layer build system |
| **Python** | 3.10+ | Build tools (shader compiler, asset cooker) |
| **Git** | 2.x | Version control |

### Platform-Specific Requirements

**Windows:**
- Visual Studio 2022 Build Tools (MSVC toolchain)
- Vulkan SDK (https://vulkan.lunarg.com/sdk/home)

**macOS:**
- Xcode Command Line Tools (`xcode-select --install`)
- MoltenVK (included in Vulkan SDK for macOS)

**Linux:**
- GCC 12+ or Clang 15+
- Vulkan development headers (`libvulkan-dev` on Debian/Ubuntu)
- X11/Wayland development headers

### Optional Tools

| Tool | Purpose |
|------|---------|
| Vulkan SDK | Shader compilation (glslangValidator), validation layers |
| NVIDIA PhysX SDK | Physics middleware (requires separate download) |
| FMOD Studio | Audio middleware (requires license) |

## Installation

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
rustup update
```

Verify:
```bash
rustc --version   # Should be 1.85.0 or later
cargo --version
```

### 2. Clone the Repository

```bash
git clone https://github.com/genovo/genovo-engine.git
cd genovo-engine
```

### 3. Build the Engine

```bash
# Debug build (faster compilation, slower runtime)
cargo build

# Release build (slower compilation, optimized runtime)
cargo build --release

# Release with debug info (for profiling)
cargo build --profile release-with-debug
```

Or use the platform build scripts:

```bash
# Windows (from Git Bash or WSL)
./tools/build_scripts/build_windows.sh

# macOS
./tools/build_scripts/build_macos.sh

# Linux
./tools/build_scripts/build_linux.sh
```

### 4. Run Tests

```bash
# Run all tests
cargo test --workspace

# Run specific test crate
cargo test -p genovo-integration-tests

# Run benchmarks
cargo bench -p genovo-benchmarks
```

## Running Examples

### Hello Triangle

Renders a single colored triangle -- the "hello world" of graphics programming.

```bash
cargo run --example hello_triangle
```

### ECS Demo

Demonstrates creating entities, attaching components, and running systems.

```bash
cargo run --example ecs_demo
```

## Project Structure

```
genovo-engine/
  Cargo.toml              # Workspace root
  engine/
    core/                  # Math, memory, reflection, threading
    ecs/                   # Entity Component System
    render/                # GPU rendering
    platform/              # Window, input, OS abstraction
    scene/                 # Scene graph
    physics/               # Physics simulation
    audio/                 # Audio engine
    animation/             # Skeletal and property animation
    assets/                # Asset loading pipeline
    scripting/             # Scripting runtime
    networking/            # Multiplayer networking
    ai/                    # AI (behavior trees, pathfinding)
    editor/                # Level editor
    genovo/                # Facade crate (re-exports everything)
  crates/
    genovo-ffi/            # C/C++ interop layer
  tools/
    shader_compiler/       # GLSL/HLSL to SPIR-V compiler
    asset_cooker/          # Asset processing pipeline
    build_scripts/         # Platform build scripts
  tests/
    integration/           # Integration tests
    benchmarks/            # Performance benchmarks
  examples/                # Example applications
  docs/                    # Documentation
```

## Creating a New Game Project

A Genovo game project is a Cargo project that depends on the `genovo` facade crate:

```toml
[package]
name = "my-game"
version = "0.1.0"
edition = "2024"

[dependencies]
genovo = { path = "../genovo-engine/engine/genovo" }
```

```rust
use genovo::prelude::*;

fn main() {
    let app = App::new()
        .add_plugin(DefaultPlugins)
        .add_system(my_game_system)
        .run();
}

fn my_game_system(query: Query<(&Position, &mut Velocity)>) {
    // Game logic here
}
```

## Next Steps

- Read the [Architecture Overview](../architecture/overview.md) to understand the engine design
- Explore the [Module Documentation](../modules/) for detailed API information
- Check the [Contributing Guide](contributing.md) to contribute to the engine
