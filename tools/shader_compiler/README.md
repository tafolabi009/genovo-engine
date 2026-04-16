# Shader Compiler

Compiles shader source files through a multi-stage pipeline:

```
GLSL / HLSL source
     |
     v
glslangValidator / DXC (front-end compilation)
     |
     v
SPIR-V (intermediate representation)
     |
     v
spirv-cross / spirv-opt (optimization + cross-compilation)
     |
     v
Platform bytecode:
  - Vulkan: SPIR-V (direct)
  - DirectX 12: DXIL (via DXC)
  - Metal: MSL (via spirv-cross)
  - OpenGL ES: ESSL (via spirv-cross)
```

## Usage

```bash
python compile.py --input shaders/ --output build/shaders/ --target vulkan
python compile.py --input shaders/ --output build/shaders/ --target dx12
python compile.py --input shaders/ --output build/shaders/ --target metal
```

## Requirements

- **glslangValidator** - GLSL to SPIR-V compiler (part of Vulkan SDK)
- **spirv-opt** - SPIR-V optimizer (part of SPIRV-Tools)
- **spirv-cross** - SPIR-V cross-compiler to HLSL/MSL/ESSL
- **DXC** (optional) - DirectX Shader Compiler for DXIL output on Windows

Install the Vulkan SDK to get glslangValidator and spirv-opt. Install spirv-cross separately or from the Vulkan SDK.

## Shader Conventions

- Vertex shaders: `*.vert.glsl` or `*.vert.hlsl`
- Fragment shaders: `*.frag.glsl` or `*.frag.hlsl`
- Compute shaders: `*.comp.glsl` or `*.comp.hlsl`
- Include files: `*.glsl.inc` or `*.hlsl.inc` (not compiled directly)

## Shader Metadata

Each compiled shader produces a `.meta.json` sidecar file containing:

- Input/output bindings
- Descriptor set layouts
- Push constant ranges
- Workgroup sizes (compute shaders)

This metadata is consumed by the engine's render pipeline at load time.
