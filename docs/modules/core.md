# genovo-core

The foundation crate providing math, memory, reflection, and threading primitives used by all other engine modules.

## Sub-modules

### Math (`core::math`)

Wraps and extends the `glam` crate for linear algebra:

- `Vec2`, `Vec3`, `Vec4` - Vector types
- `Mat3`, `Mat4` - Matrix types
- `Quat` - Quaternion for rotations
- `Transform` - Position + Rotation + Scale bundle
- `Aabb` - Axis-aligned bounding box
- `Ray` - Origin + direction ray
- `Frustum` - View frustum for culling
- Intersection tests: ray-AABB, ray-sphere, frustum-AABB, frustum-sphere

### Memory (`core::memory`)

Custom allocators for performance-critical allocation patterns:

- `FrameAllocator` - Bump allocator reset once per frame. O(1) allocation, zero-cost deallocation.
- `PoolAllocator<T>` - Fixed-size block allocator for homogeneous objects. Reduces fragmentation.
- `ArenaAllocator` - Region-based allocator for data with uniform lifetime (e.g. level-scoped).
- `AlignedVec<T>` - SIMD-friendly aligned vector container.

### Reflection (`core::reflection`)

Minimal runtime type information:

- `TypeInfo` - Type name, size, alignment, field descriptors
- `FieldInfo` - Field name, offset, type, metadata
- `#[derive(Reflect)]` (future) - Procedural macro for automatic reflection

Used by the editor's property inspector and the serialization layer.

### Threading (`core::threading`)

Task-based parallelism primitives:

- `JobSystem` - Work-stealing thread pool
- `TaskGraph` - Directed acyclic graph of tasks with dependency edges
- `TaskHandle` - Handle to a pending/completed task
- `ScopedThreadPool` - Scoped thread pool for structured concurrency
- `AtomicCell<T>` - Lock-free single-value container

## Key Traits

```rust
/// Types that can provide runtime reflection metadata.
pub trait Reflect {
    fn type_info() -> &'static TypeInfo;
}

/// Types that can be serialized to/from the engine's binary format.
pub trait Serialize {
    fn serialize(&self, writer: &mut dyn Write) -> Result<()>;
    fn deserialize(reader: &mut dyn Read) -> Result<Self> where Self: Sized;
}
```

## Usage

```rust
use genovo_core::math::{Vec3, Transform, Aabb};

let transform = Transform {
    position: Vec3::new(1.0, 2.0, 3.0),
    rotation: Quat::IDENTITY,
    scale: Vec3::ONE,
};

let aabb = Aabb::from_center_extents(Vec3::ZERO, Vec3::splat(5.0));
assert!(aabb.contains_point(transform.position));
```

## Status

- Math: Wrapper types defined, implementing intersection tests (Month 1-2)
- Memory: Allocator interfaces defined, FrameAllocator in progress (Month 2)
- Reflection: Trait defined, derive macro planned (Month 3)
- Threading: JobSystem interface defined, implementation planned (Month 2-3)
