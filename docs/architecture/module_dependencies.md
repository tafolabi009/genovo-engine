# Module Dependency Graph

This document describes the dependency relationships between Genovo engine modules. Dependencies flow downward -- modules may only depend on modules listed below them in the hierarchy.

## Dependency Diagram

```
                       genovo (facade)
                            |
         +------------------+------------------+
         |                  |                  |
      editor           scripting          networking
         |                  |                  |
    +----+----+        +----+----+        +----+
    |    |    |        |         |        |
  scene  |  assets   scene     ecs      ecs
    |    |    |        |                  |
    |  render |      +---+              core
    |    |    |      |   |
    |    |    +------+   |
    |    |               |
    +----+---+       animation
         |    |          |
         |  physics      |
         |    |          |
         |    +----+-----+
         |         |
         |        ecs
         |         |
       platform  core
         |         |
         +----+----+
              |
         (external)
```

## Per-Module Dependencies

### Foundation

| Module | Dependencies | External Crates |
|--------|-------------|-----------------|
| `genovo-core` | *(none)* | `glam`, `parking_lot`, `crossbeam`, `rayon`, `smallvec`, `bytemuck`, `profiling`, `log`, `thiserror`, `bitflags` |
| `genovo-ecs` | `genovo-core` | `log`, `thiserror`, `rayon` |

### Runtime

| Module | Dependencies | External Crates |
|--------|-------------|-----------------|
| `genovo-platform` | `genovo-core` | `winit`, `raw-window-handle`, `log` |
| `genovo-render` | `genovo-core`, `genovo-ecs`, `genovo-platform` | `ash`, `wgpu`, `gpu-allocator`, `bytemuck`, `log` |
| `genovo-assets` | `genovo-core` | `serde`, `serde_json`, `ron`, `uuid`, `log` |

### Simulation

| Module | Dependencies | External Crates |
|--------|-------------|-----------------|
| `genovo-scene` | `genovo-core`, `genovo-ecs` | `serde`, `uuid`, `log` |
| `genovo-physics` | `genovo-core`, `genovo-ecs` | `glam`, `log` |
| `genovo-audio` | `genovo-core`, `genovo-ecs` | `log` |
| `genovo-animation` | `genovo-core`, `genovo-ecs` | `log` |

### Game

| Module | Dependencies | External Crates |
|--------|-------------|-----------------|
| `genovo-ai` | `genovo-core`, `genovo-ecs` | `log` |
| `genovo-networking` | `genovo-core`, `genovo-ecs` | `serde`, `log` |
| `genovo-scripting` | `genovo-core`, `genovo-ecs`, `genovo-scene` | `log` |

### Tooling

| Module | Dependencies | External Crates |
|--------|-------------|-----------------|
| `genovo-editor` | `genovo-core`, `genovo-ecs`, `genovo-scene`, `genovo-render`, `genovo-assets` | `serde`, `serde_json`, `uuid`, `log`, `thiserror` |
| `genovo-ffi` | `genovo-core`, `genovo-ecs`, `genovo-physics`, `genovo-audio` | `log`, `cbindgen` (build) |

### Facade

| Module | Dependencies |
|--------|-------------|
| `genovo` | All engine crates (re-exports) |

## Dependency Rules

1. **No circular dependencies** - The crate graph must be a DAG. If module A depends on module B, module B must not depend on module A.
2. **Core is foundational** - `genovo-core` has no internal dependencies and should remain lightweight.
3. **ECS is the backbone** - Most simulation and game modules depend on `genovo-ecs` for component storage and system scheduling.
4. **Platform is isolated** - Only `genovo-render` and `genovo-platform` directly interact with OS/windowing APIs.
5. **FFI is a leaf** - `genovo-ffi` depends on engine crates but nothing depends on it. It is purely an outward-facing interface.
6. **Editor is optional** - The editor depends on many modules but can be excluded from runtime builds entirely.

## Build Order

Cargo resolves build order automatically from the dependency graph. The approximate order is:

1. `genovo-core`
2. `genovo-ecs`, `genovo-platform`, `genovo-assets`
3. `genovo-render`, `genovo-scene`, `genovo-physics`, `genovo-audio`, `genovo-animation`
4. `genovo-ai`, `genovo-networking`, `genovo-scripting`
5. `genovo-editor`, `genovo-ffi`
6. `genovo` (facade)
7. `genovo-examples`, `genovo-integration-tests`, `genovo-benchmarks`
