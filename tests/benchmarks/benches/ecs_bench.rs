//! # ECS Benchmarks
//!
//! Performance benchmarks for the ECS subsystem using the Criterion framework.
//! Run with: `cargo bench -p genovo-benchmarks`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

// TODO(BENCH): Enable benchmarks once genovo-ecs crate has implementations - Month 3-4
// use genovo_ecs::*;

// ---------------------------------------------------------------------------
// Component types for benchmarking
// ---------------------------------------------------------------------------

// #[derive(Clone, Copy, Debug, Default)]
// struct Position { x: f32, y: f32, z: f32 }
//
// #[derive(Clone, Copy, Debug, Default)]
// struct Velocity { x: f32, y: f32, z: f32 }
//
// #[derive(Clone, Copy, Debug, Default)]
// struct Health { current: f32, max: f32 }

// ---------------------------------------------------------------------------
// Spawn benchmarks
// ---------------------------------------------------------------------------

fn bench_entity_spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_spawn");

    for count in [100, 1_000, 10_000, 100_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.iter(|| {
                // TODO(BENCH): Spawn `count` entities with Position component
                // let mut world = World::new();
                // for _ in 0..count {
                //     world.spawn().insert(Position::default());
                // }
                // black_box(&world);
                black_box(count);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Query benchmarks
// ---------------------------------------------------------------------------

fn bench_simple_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_query");

    for count in [1_000, 10_000, 100_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            // TODO(BENCH): Pre-populate world with `count` entities
            // let mut world = World::new();
            // for _ in 0..count {
            //     world.spawn()
            //         .insert(Position::default())
            //         .insert(Velocity::default());
            // }

            b.iter(|| {
                // TODO(BENCH): Query all (Position, Velocity) pairs
                // for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
                //     pos.x += vel.x;
                //     pos.y += vel.y;
                //     pos.z += vel.z;
                // }
                black_box(count);
            });
        });
    }

    group.finish();
}

fn bench_filtered_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtered_query");

    for count in [1_000, 10_000, 100_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            // TODO(BENCH): Populate world where only 10% of entities have Health
            // let mut world = World::new();
            // for i in 0..count {
            //     let entity = world.spawn().insert(Position::default());
            //     if i % 10 == 0 {
            //         entity.insert(Health { current: 100.0, max: 100.0 });
            //     }
            // }

            b.iter(|| {
                // TODO(BENCH): Query (Position, Health) - should be ~10% of entities
                // let matched: usize = world.query::<(&Position, &Health)>().count();
                // black_box(matched);
                black_box(count);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Component add/remove benchmarks
// ---------------------------------------------------------------------------

fn bench_add_remove_component(c: &mut Criterion) {
    c.bench_function("add_remove_component", |b| {
        // TODO(BENCH): Pre-spawn entities, benchmark adding/removing a component
        // let mut world = World::new();
        // let entities: Vec<_> = (0..10_000)
        //     .map(|_| world.spawn().insert(Position::default()).id())
        //     .collect();

        b.iter(|| {
            // for &entity in &entities {
            //     world.insert(entity, Health { current: 100.0, max: 100.0 });
            // }
            // for &entity in &entities {
            //     world.remove::<Health>(entity);
            // }
            black_box(0);
        });
    });
}

// ---------------------------------------------------------------------------
// System scheduling benchmarks
// ---------------------------------------------------------------------------

fn bench_system_execution(c: &mut Criterion) {
    c.bench_function("system_execution_10_systems", |b| {
        // TODO(BENCH): Register 10 systems, benchmark a single tick
        // let mut world = World::new();
        // // Spawn entities
        // for _ in 0..10_000 {
        //     world.spawn()
        //         .insert(Position::default())
        //         .insert(Velocity::default());
        // }
        // // Register systems
        // for _ in 0..10 {
        //     world.add_system(MovementSystem);
        // }

        b.iter(|| {
            // world.tick(1.0 / 60.0);
            black_box(0);
        });
    });
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_entity_spawn,
    bench_simple_query,
    bench_filtered_query,
    bench_add_remove_component,
    bench_system_execution,
);

criterion_main!(benches);
