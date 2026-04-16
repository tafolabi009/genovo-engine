# genovo-ecs

Archetypal Entity Component System providing high-performance entity management, component storage, and parallel system scheduling.

## Core Concepts

### Entity

An entity is a lightweight identifier (64-bit: 32-bit index + 32-bit generation) that serves as a handle to a collection of components. Entities have no behavior on their own.

### Component

A component is a plain data struct (POD). Components are stored contiguously in memory grouped by archetype for cache efficiency.

```rust
#[derive(Component)]
struct Position { x: f32, y: f32, z: f32 }

#[derive(Component)]
struct Velocity { x: f32, y: f32, z: f32 }

#[derive(Component)]
struct Health { current: f32, max: f32 }
```

### Archetype

An archetype is a unique combination of component types. All entities with the same set of components share an archetype. Component data is stored in a struct-of-arrays (SoA) layout within each archetype for optimal iteration.

### System

A system is a function that operates on entities matching a specific component query. Systems declare their data access patterns, enabling the scheduler to parallelize non-conflicting systems.

```rust
fn movement_system(query: Query<(&mut Position, &Velocity)>, dt: f32) {
    for (pos, vel) in query.iter() {
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;
    }
}
```

### World

The world is the top-level container holding all entities, components, and systems. It orchestrates system execution each frame.

## API Overview

```rust
// Create the world
let mut world = World::new();

// Spawn entities
let player = world.spawn()
    .insert(Position { x: 0.0, y: 0.0, z: 0.0 })
    .insert(Velocity { x: 1.0, y: 0.0, z: 0.0 })
    .insert(Health { current: 100.0, max: 100.0 })
    .id();

// Query components
for (pos, vel) in world.query::<(&Position, &Velocity)>() {
    println!("Entity at ({}, {}, {}) moving at ({}, {}, {})",
        pos.x, pos.y, pos.z, vel.x, vel.y, vel.z);
}

// Remove a component
world.remove::<Velocity>(player);

// Despawn an entity
world.despawn(player);
```

## System Scheduling

Systems are organized into stages that execute in order. Within each stage, systems with non-conflicting access patterns run in parallel automatically.

```
Stage: PreUpdate
  |-- InputSystem (reads: Input)
  |-- NetworkReceiveSystem (reads: Network, writes: Position, Health)

Stage: Update
  |-- MovementSystem (reads: Velocity, writes: Position)  -- parallel -->
  |-- AISystem (reads: Position, writes: AIState)         -- parallel -->
  |-- AnimationSystem (reads: AnimState, writes: Transform)

Stage: PostUpdate
  |-- PhysicsSystem (reads/writes: Position, Velocity, Collider)
  |-- TransformPropagation (reads: Parent, writes: GlobalTransform)
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Spawn entity | O(1) amortized | May trigger archetype allocation |
| Despawn entity | O(n components) | Swap-remove from archetype |
| Add component | O(n components) | Entity moves to new archetype |
| Remove component | O(n components) | Entity moves to new archetype |
| Query iteration | O(matching entities) | Cache-friendly SoA traversal |
| System scheduling | O(n systems) | Graph-based dependency resolution |

## Status

- Entity storage and generation tracking (Month 2)
- Archetypal component storage (Month 2-3)
- Basic queries (Month 2-3)
- System scheduler with parallel execution (Month 3)
- Change detection (Month 4)
- Command buffers for deferred operations (Month 3-4)
