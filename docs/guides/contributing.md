# Contributing to Genovo Engine

Thank you for your interest in contributing to the Genovo game engine. This guide covers the development workflow, coding standards, and submission process.

## Development Setup

1. Follow the [Getting Started](getting_started.md) guide to set up your environment.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-feature main
   ```

## Code Style

### Rust Conventions

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- Use `rustfmt` with the project's configuration (default settings).
- Use `clippy` and address all warnings:
  ```bash
  cargo clippy --workspace -- -D warnings
  ```
- Maximum line length: 100 characters (soft limit, `rustfmt` handles this).
- Use `///` doc comments on all public items.
- Prefer explicit types on public function signatures; use inference internally.

### Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Crates | `genovo-{name}` | `genovo-core`, `genovo-ecs` |
| Modules | `snake_case` | `asset_browser`, `transform_system` |
| Types | `PascalCase` | `RenderContext`, `PhysicsWorld` |
| Traits | `PascalCase` (adjective/verb) | `Renderable`, `Inspectable` |
| Functions | `snake_case` | `create_buffer`, `spawn_entity` |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_ENTITIES`, `DEFAULT_GRAVITY` |
| Type parameters | Single uppercase or descriptive `PascalCase` | `T`, `ComponentType` |

### TODO Markers

Use the standardized TODO format for tracking incomplete work:

```rust
// TODO(CATEGORY): Description - Timeline estimate
```

Categories:
- `CORE` - Core module tasks
- `ECS` - Entity Component System
- `RENDER` - Rendering pipeline
- `PHYSICS` - Physics engine
- `AUDIO` - Audio system
- `EDITOR` - Editor tooling
- `FFI` - C++ interop
- `BUILD` - Build system and tooling
- `TEST` - Test infrastructure
- `BENCH` - Benchmarks
- `EXAMPLE` - Example code
- `PERF` - Performance optimization
- `SECURITY` - Security considerations

### Error Handling

- Use `thiserror` for defining error types in libraries.
- Use `Result<T, E>` for fallible operations; avoid `unwrap()` in library code.
- Use `log` macros (`error!`, `warn!`, `info!`, `debug!`, `trace!`) for diagnostics.
- Panics are acceptable only for programming errors (invariant violations), never for runtime failures.

### Safety

- Minimize `unsafe` code. When `unsafe` is necessary:
  - Write a `// SAFETY:` comment explaining why the usage is sound.
  - Encapsulate unsafe code in safe abstractions.
  - Add tests covering the unsafe code paths.
- FFI boundaries (`genovo-ffi`) are the exception where `unsafe` is expected.

## Testing

### Test Hierarchy

1. **Unit tests** - In-module `#[cfg(test)]` tests for individual functions.
2. **Integration tests** - In `tests/integration/` for cross-module behavior.
3. **Benchmarks** - In `tests/benchmarks/` using Criterion for performance tracking.

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_entity_returns_valid_id() {
        let mut world = World::new();
        let entity = world.spawn();
        assert!(entity.is_valid());
    }
}
```

### Running Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p genovo-ecs

# With output
cargo test --workspace -- --nocapture

# Benchmarks
cargo bench -p genovo-benchmarks
```

## Pull Request Process

1. **One concern per PR** - Keep PRs focused on a single feature, fix, or refactor.
2. **Write tests** - New features must include tests. Bug fixes should include a regression test.
3. **Update documentation** - Update module docs and relevant markdown files.
4. **Pass CI** - All tests, clippy, and formatting checks must pass.
5. **Description** - Write a clear PR description explaining what changes and why.

### PR Title Format

```
module: Brief description of change

Examples:
  ecs: Add archetypal component storage
  render: Fix swapchain recreation on resize
  core: Implement frame allocator
  editor: Add property inspector widget types
```

### Review Checklist

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] `cargo clippy` reports no warnings
- [ ] `cargo fmt` produces no changes
- [ ] Public APIs have doc comments
- [ ] Unsafe code has SAFETY comments
- [ ] No unintended dependency additions

## Architecture Decisions

For significant design changes, open a discussion issue before submitting a PR. Include:

- Problem statement
- Proposed solution
- Alternatives considered
- Performance implications
- Breaking changes (if any)

## Module Ownership

Each engine module has a designated owner responsible for reviewing PRs and maintaining quality. Check the CODEOWNERS file for current assignments.

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).
