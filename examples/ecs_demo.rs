//! # ECS Demo
//!
//! Demonstrates the Entity Component System by creating entities with
//! components, iterating over them manually, and running a simple movement
//! simulation.
//!
//! Run with:
//! ```
//! cargo run --example ecs_demo
//! ```

use genovo::prelude::*;

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Position in 2D space.
#[derive(Debug, Clone)]
struct Position {
    x: f32,
    y: f32,
}
impl Component for Position {}

/// Velocity vector.
#[derive(Debug, Clone)]
struct Velocity {
    x: f32,
    y: f32,
}
impl Component for Velocity {}

/// Health component.
#[derive(Debug, Clone)]
struct Health {
    current: f32,
    max: f32,
}
impl Component for Health {}

/// Tag component marking an entity as a player.
#[derive(Debug, Clone)]
struct Player;
impl Component for Player {}

/// Tag component marking an entity as an enemy.
#[derive(Debug, Clone)]
struct Enemy;
impl Component for Enemy {}

/// Name component for display purposes.
#[derive(Debug, Clone)]
struct Name(String);
impl Component for Name {}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("Genovo ECS Demo");
    println!("===============");
    println!();

    let mut world = World::new();

    // -----------------------------------------------------------------------
    // Step 1: Spawn entities
    // -----------------------------------------------------------------------

    // Player entity
    let hero = world
        .spawn_entity()
        .with(Name("Hero".to_string()))
        .with(Position { x: 0.0, y: 0.0 })
        .with(Velocity { x: 1.0, y: 0.5 })
        .with(Health {
            current: 100.0,
            max: 100.0,
        })
        .with(Player)
        .build();

    // Enemy entities
    let mut enemies = Vec::new();
    for i in 0..5 {
        let enemy = world
            .spawn_entity()
            .with(Name(format!("Goblin {}", i)))
            .with(Position {
                x: 10.0 + i as f32 * 2.0,
                y: 5.0,
            })
            .with(Velocity { x: -0.5, y: 0.0 })
            .with(Health {
                current: 30.0,
                max: 30.0,
            })
            .with(Enemy)
            .build();
        enemies.push(enemy);
    }

    // Static scenery (position only, no velocity)
    let _tree = world
        .spawn_entity()
        .with(Name("Tree".to_string()))
        .with(Position { x: 5.0, y: 3.0 })
        .build();

    println!("Spawned {} entities", world.entity_count());

    // -----------------------------------------------------------------------
    // Step 2: Query and display initial state
    // -----------------------------------------------------------------------

    println!("\n--- Initial State ---");
    print_named_positions(&world, &[hero], "Player");
    print_named_positions(&world, &enemies, "Enemies");

    // -----------------------------------------------------------------------
    // Step 3: Run movement simulation (60 frames)
    // -----------------------------------------------------------------------

    let dt = 1.0 / 60.0_f32;
    let num_frames = 60;

    println!("\nSimulating {} entities for {} frames...", world.entity_count(), num_frames);

    // Collect all entities that have both Position and Velocity.
    // Since the ECS uses HashMap-based storage with entity IDs, we iterate
    // over known entities and check for components.
    let all_entities: Vec<Entity> = {
        let mut ents = vec![hero];
        ents.extend_from_slice(&enemies);
        ents
    };

    for frame in 0..num_frames {
        // Movement system: apply velocity to position
        for &entity in &all_entities {
            let vel = world.get_component::<Velocity>(entity).cloned();
            if let Some(vel) = vel {
                if let Some(pos) = world.get_component_mut::<Position>(entity) {
                    pos.x += vel.x * dt;
                    pos.y += vel.y * dt;
                }
            }
        }

        // Damage system: reduce enemy health over time
        for &entity in &enemies {
            if world.has_component::<Enemy>(entity) {
                if let Some(health) = world.get_component_mut::<Health>(entity) {
                    health.current -= 10.0 * dt;
                    if health.current < 0.0 {
                        health.current = 0.0;
                    }
                }
            }
        }

        // Print state at specific frames
        if frame == 0 || frame == 29 || frame == 59 {
            println!("\n--- Frame {} (t = {:.3}s) ---", frame, frame as f32 * dt);
            if let (Some(name), Some(pos)) = (
                world.get_component::<Name>(hero),
                world.get_component::<Position>(hero),
            ) {
                println!("  {} at ({:.1}, {:.1})", name.0, pos.x, pos.y);
            }
            for &enemy in &enemies {
                if let (Some(name), Some(pos), Some(hp)) = (
                    world.get_component::<Name>(enemy),
                    world.get_component::<Position>(enemy),
                    world.get_component::<Health>(enemy),
                ) {
                    println!(
                        "  {} at ({:.1}, {:.1}) HP={:.1}/{:.1}",
                        name.0, pos.x, pos.y, hp.current, hp.max,
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: Cleanup dead enemies
    // -----------------------------------------------------------------------

    println!("\n--- Cleanup ---");
    let mut despawned = 0;
    for &entity in &enemies {
        if let Some(health) = world.get_component::<Health>(entity) {
            if health.current <= 0.0 {
                if let Some(name) = world.get_component::<Name>(entity) {
                    println!("  Despawning dead entity: {}", name.0);
                }
                world.despawn(entity);
                despawned += 1;
            }
        }
    }
    println!("  Despawned {} entities, {} remaining", despawned, world.entity_count());

    // -----------------------------------------------------------------------
    // Step 5: Resource demo
    // -----------------------------------------------------------------------

    println!("\n--- Resource Demo ---");
    world.add_resource(42.0_f64);
    println!(
        "  Added resource f64 = {:?}",
        world.get_resource::<f64>()
    );
    *world.get_resource_mut::<f64>().unwrap() = 99.0;
    println!(
        "  Modified resource f64 = {:?}",
        world.get_resource::<f64>()
    );

    println!("\nECS demo complete! {} frames of {} entities simulated.", num_frames, all_entities.len() + 1);
}

/// Helper: print positions for a set of entities.
fn print_named_positions(world: &World, entities: &[Entity], label: &str) {
    println!("  {}:", label);
    for &entity in entities {
        if let (Some(name), Some(pos)) = (
            world.get_component::<Name>(entity),
            world.get_component::<Position>(entity),
        ) {
            println!("    {} at ({:.1}, {:.1})", name.0, pos.x, pos.y);
        }
    }
}
