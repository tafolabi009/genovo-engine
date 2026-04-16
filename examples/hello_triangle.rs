//! # Hello Triangle
//!
//! Minimal example demonstrating the Genovo engine facade by creating an
//! engine instance, spawning entities with transforms and physics bodies,
//! and running a short simulation.
//!
//! Run with:
//! ```
//! cargo run --example hello_triangle
//! ```

use genovo::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig {
        app_name: "Hello Triangle".to_string(),
        ..Default::default()
    };

    let mut engine = Engine::new(config)?;

    // -----------------------------------------------------------------------
    // Step 1: Spawn ECS entities with transform components
    // -----------------------------------------------------------------------

    let world = engine.world_mut();

    let _ground = world
        .spawn_entity()
        .with(TransformComponent::from_position(Vec3::new(0.0, -1.0, 0.0)))
        .build();

    let _ball = world
        .spawn_entity()
        .with(TransformComponent::from_position(Vec3::new(0.0, 10.0, 0.0)))
        .build();

    let _camera = world
        .spawn_entity()
        .with(TransformComponent::from_position(Vec3::new(0.0, 5.0, 15.0)))
        .build();

    println!("Genovo Engine initialized successfully!");
    println!("World has {} entities", engine.world().entity_count());

    // -----------------------------------------------------------------------
    // Step 2: Add physics bodies
    // -----------------------------------------------------------------------

    // Static ground plane
    let ground_body = engine
        .physics_mut()
        .add_body(&RigidBodyDesc {
            body_type: BodyType::Static,
            position: Vec3::new(0.0, -1.0, 0.0),
            ..Default::default()
        })
        .expect("failed to add ground body");

    engine
        .physics_mut()
        .add_collider(
            ground_body,
            &ColliderDesc {
                shape: CollisionShape::Box {
                    half_extents: Vec3::new(50.0, 1.0, 50.0),
                },
                ..Default::default()
            },
        )
        .expect("failed to add ground collider");

    // Dynamic ball
    let ball_body = engine
        .physics_mut()
        .add_body(&RigidBodyDesc {
            body_type: BodyType::Dynamic,
            mass: 1.0,
            position: Vec3::new(0.0, 10.0, 0.0),
            restitution: 0.6,
            ..Default::default()
        })
        .expect("failed to add ball body");

    engine
        .physics_mut()
        .add_collider(
            ball_body,
            &ColliderDesc {
                shape: CollisionShape::Sphere { radius: 0.5 },
                material: PhysicsMaterial {
                    restitution: 0.6,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("failed to add ball collider");

    println!(
        "Physics world: {} bodies, gravity = {:?}",
        engine.physics().body_count(),
        engine.physics().gravity(),
    );

    // -----------------------------------------------------------------------
    // Step 3: Load a test audio clip into the mixer
    // -----------------------------------------------------------------------

    let test_clip = AudioClip::sine_wave(440.0, 0.5, 44100);
    engine.audio_mut().load_clip(test_clip);
    println!("Audio mixer: {} clips loaded", engine.audio().clip_count());

    // -----------------------------------------------------------------------
    // Step 4: Run a 100-frame simulation
    // -----------------------------------------------------------------------

    println!("\nSimulating 100 physics frames at 60 Hz...");

    let initial_pos = engine
        .physics()
        .get_position(ball_body)
        .expect("ball body missing");
    println!(
        "  Ball start position: ({:.2}, {:.2}, {:.2})",
        initial_pos.x, initial_pos.y, initial_pos.z,
    );

    // Step physics directly at a fixed 60 Hz rate (headless demo -- real-time
    // clock accumulation would produce near-zero dt since we run in a tight
    // loop without any rendering or vsync).
    let fixed_dt = 1.0 / 60.0_f32;

    for frame in 0..100 {
        engine
            .physics_world
            .step(fixed_dt)
            .expect("physics step failed");

        // Log ball position periodically
        if frame % 25 == 0 || frame == 99 {
            let pos = engine
                .physics()
                .get_position(ball_body)
                .expect("ball body missing");
            println!(
                "  Frame {:>3} (t={:.3}s): ball at ({:.2}, {:.2}, {:.2})",
                frame,
                (frame + 1) as f32 * fixed_dt,
                pos.x,
                pos.y,
                pos.z,
            );
        }
    }

    let final_pos = engine
        .physics()
        .get_position(ball_body)
        .expect("ball body missing");
    println!(
        "\nBall fell from y={:.2} to y={:.2} in {:.2}s",
        initial_pos.y,
        final_pos.y,
        100.0 * fixed_dt,
    );
    println!("Simulation complete!");

    Ok(())
}
