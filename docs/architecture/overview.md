# Genovo Engine - Architecture Overview

## Design Philosophy

Genovo is a modular, data-oriented game engine written in Rust with C++ interoperability for industry-standard middleware (PhysX, FMOD/Wwise). The engine prioritizes:

- **Safety without sacrifice** - Rust's ownership model eliminates classes of bugs common in C++ engines (use-after-free, data races) while maintaining the performance characteristics required for real-time applications.
- **Data-oriented design** - The ECS (Entity Component System) architecture keeps data in cache-friendly layouts. Components are stored in contiguous arrays, not scattered across heap-allocated objects.
- **Modularity** - Each subsystem is an independent crate with well-defined interfaces. Games link only the subsystems they need.
- **Cross-platform** - A platform abstraction layer isolates OS-specific code. The engine targets Windows, macOS, Linux, iOS, Android, and consoles.

## High-Level Architecture

```
+---------------------------------------------------------------+
|                        Game / Application                      |
+---------------------------------------------------------------+
|                        genovo (facade crate)                   |
+---------------------------------------------------------------+
|                                                                |
|  +----------+  +----------+  +---------+  +----------+        |
|  |  Editor   |  | Scripting|  |   AI    |  |Networking|        |
|  +----------+  +----------+  +---------+  +----------+        |
|                                                                |
|  +----------+  +----------+  +---------+  +----------+        |
|  |  Scene   |  | Animation|  | Physics |  |  Audio   |        |
|  +----------+  +----------+  +---------+  +----------+        |
|                                                                |
|  +----------+  +----------+  +---------+                       |
|  |  Render  |  |  Assets  |  | Platform|                       |
|  +----------+  +----------+  +---------+                       |
|                                                                |
|  +----------+  +----------+                                    |
|  |   ECS    |  |   Core   |                                    |
|  +----------+  +----------+                                    |
|                                                                |
+---------------------------------------------------------------+
|              genovo-ffi (C++ interop layer)                     |
+---------------------------------------------------------------+
|         PhysX SDK  |  FMOD / Wwise  |  Platform SDKs          |
+---------------------------------------------------------------+
```

## Layer Responsibilities

### Foundation Layer

- **Core** (`genovo-core`) - Math library (leveraging `glam`), memory utilities (arena allocators, pool allocators), reflection/type metadata, threading primitives (job system, task graph), and profiling hooks.
- **ECS** (`genovo-ecs`) - Entity Component System with archetypal storage. Handles entity lifecycle, component storage, system scheduling with automatic parallelism, and change detection.

### Runtime Layer

- **Platform** (`genovo-platform`) - Window management (via `winit`), input handling, file I/O, and platform-specific abstractions. Produces `RawWindowHandle` for the render backend.
- **Render** (`genovo-render`) - GPU abstraction over Vulkan (`ash`/`wgpu`), render graph, material system, mesh rendering, lighting, post-processing pipeline.
- **Assets** (`genovo-assets`) - Async asset loading, format-specific importers, hot-reloading, reference-counted handles, and dependency tracking.

### Simulation Layer

- **Scene** (`genovo-scene`) - Scene graph with parent-child transform hierarchy, prefab/template system, and scene serialization.
- **Physics** (`genovo-physics`) - Physics simulation with pluggable backends (built-in or PhysX via FFI). Collision detection, rigid body dynamics, raycasting, and trigger volumes.
- **Audio** (`genovo-audio`) - Spatial audio, mixer with channel groups, audio source components, and middleware integration (FMOD/Wwise via FFI).
- **Animation** (`genovo-animation`) - Skeletal animation, animation blending/state machines, IK solvers, and property animation (tweens).

### Game Layer

- **AI** (`genovo-ai`) - Behavior trees, navigation meshes with A* pathfinding, and utility AI framework.
- **Networking** (`genovo-networking`) - Client-server transport (UDP with reliability layer), state replication, client-side prediction, and lag compensation.
- **Scripting** (`genovo-scripting`) - Embedded scripting runtime for gameplay logic (Lua or custom DSL).

### Tooling Layer

- **Editor** (`genovo-editor`) - Full-featured level editor with 3D viewport, property inspector, asset browser, scene hierarchy, and project management.
- **FFI** (`genovo-ffi`) - C-compatible foreign function interface for interop with C++ middleware libraries.

## Data Flow

A typical frame follows this flow:

1. **Input** - Platform polls OS events, updates input state
2. **Scripting** - Gameplay scripts execute, modifying ECS components
3. **AI** - Behavior trees tick, pathfinding updates
4. **Physics** - Fixed-timestep simulation step, collision callbacks fire
5. **Animation** - Skeletal animation evaluates, blend trees update
6. **Scene** - Transform hierarchy propagates world transforms
7. **Render** - Render graph executes: culling, draw call submission, post-processing
8. **Audio** - Audio sources update, 3D spatialization recalculated
9. **Present** - Swapchain presents the final frame

## Threading Model

The engine uses a job-based threading model:

- **Main thread** - Owns the window, processes OS events, orchestrates frame
- **Render thread** - Submits GPU command buffers, manages swapchain
- **Worker threads** - Execute jobs from the task graph (physics, animation, culling, etc.)
- **IO thread** - Handles async file and network I/O

The ECS system scheduler automatically parallelizes systems whose component access patterns do not conflict.

## Memory Model

- **Frame allocator** - Bump allocator reset each frame for transient data
- **Pool allocator** - Fixed-size block allocator for frequently allocated/freed objects
- **Arena allocator** - Long-lived allocations grouped by lifetime (level, session)
- **GPU memory** - Managed via `gpu-allocator` with sub-allocation strategies
