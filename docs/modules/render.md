# genovo-render

GPU rendering subsystem providing a hardware abstraction layer, render graph, material system, and multi-pass rendering pipeline.

## Architecture

```
Application / Editor
       |
  Render Graph (frame description)
       |
  Pass Compiler (dependency resolution, resource allocation)
       |
  Command Encoder (GPU command buffer generation)
       |
  GPU Backend (Vulkan via ash, or wgpu for portability)
       |
  GPU Hardware
```

## GPU Backend

The render module abstracts over graphics APIs through a backend trait:

- **Vulkan** (primary) - Direct Vulkan access via `ash` for maximum control. Used on Windows, Linux, and Android.
- **wgpu** (portable) - WebGPU-based backend via the `wgpu` crate. Automatically selects Vulkan, Metal, DX12, or WebGPU depending on platform. Preferred for macOS/iOS and web targets.

### Backend Selection

The backend is chosen at initialization time based on platform and configuration:

```rust
let backend = match platform {
    Platform::Windows | Platform::Linux => Backend::Vulkan,
    Platform::MacOS | Platform::IOS => Backend::Wgpu,  // Metal underneath
    Platform::Web => Backend::Wgpu,                     // WebGPU
    Platform::Android => Backend::Vulkan,
    _ => Backend::Wgpu,  // Fallback
};
```

## Render Graph

The render graph is a frame-level description of all render passes and their dependencies. The graph compiler:

1. Resolves pass ordering from resource dependencies
2. Allocates transient GPU resources (render targets, buffers)
3. Inserts pipeline barriers and layout transitions
4. Merges compatible passes when possible

```rust
let mut graph = RenderGraph::new();

let depth = graph.create_texture("depth", TextureDesc::depth(1920, 1080));
let color = graph.create_texture("color", TextureDesc::rgba16f(1920, 1080));

graph.add_pass("shadow", |pass| {
    pass.writes(depth);
    pass.execute(|ctx| { /* render shadow map */ });
});

graph.add_pass("geometry", |pass| {
    pass.reads(depth);
    pass.writes(color);
    pass.execute(|ctx| { /* render geometry */ });
});

graph.add_pass("post_process", |pass| {
    pass.reads(color);
    pass.writes_to_swapchain();
    pass.execute(|ctx| { /* tone mapping, FXAA */ });
});

graph.compile_and_execute();
```

## Material System

Materials describe how surfaces are rendered:

- **Shader programs** - Vertex, fragment, compute stages
- **Parameters** - Uniforms, textures, samplers bound per-material
- **Render state** - Blend mode, depth test, cull mode, polygon mode

```rust
let material = Material::new()
    .shader("pbr_standard.vert.spv", "pbr_standard.frag.spv")
    .param("albedo_map", albedo_texture)
    .param("normal_map", normal_texture)
    .param("roughness", 0.5f32)
    .param("metallic", 0.0f32)
    .blend(BlendMode::Opaque)
    .cull(CullMode::Back);
```

## Render Pipeline Stages

1. **Culling** - Frustum and occlusion culling to eliminate non-visible objects
2. **Shadow Pass** - Render shadow maps for each light source
3. **G-Buffer Pass** (deferred) - Render geometry attributes to multiple render targets
4. **Lighting Pass** - Evaluate lighting using G-Buffer data
5. **Forward Pass** - Render transparent and special-case objects
6. **Post-Processing** - Tone mapping, bloom, SSAO, TAA, FXAA

## Status

- GPU backend abstraction trait defined (Month 2)
- Vulkan initialization and swapchain (Month 2-3)
- First triangle rendering (Month 3 milestone)
- Render graph framework (Month 3-4)
- Material system (Month 4)
- PBR lighting pipeline (Month 4-5)
- Shadow mapping (Month 5)
- Post-processing stack (Month 5-6)
