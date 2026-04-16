# Asset Cooker

The asset cooker processes raw source assets into optimized, platform-specific formats suitable for runtime loading.

## Pipeline Overview

```
Source Asset (FBX, PNG, WAV, etc.)
     |
     v
Import (parse format, extract data)
     |
     v
Process (optimize, compress, generate LODs)
     |
     v
Cook (serialize to engine runtime format)
     |
     v
.gasset file (Genovo Asset Binary)
```

## Supported Asset Types

| Source Format       | Asset Type | Processing                                      |
|---------------------|------------|-------------------------------------------------|
| `.fbx`, `.gltf`    | Mesh       | Vertex optimization, LOD generation, tangent calc |
| `.png`, `.jpg`, `.tga`, `.exr` | Texture | Mipmap generation, block compression (BC7/ASTC) |
| `.wav`, `.ogg`, `.mp3` | Audio  | Sample rate conversion, channel mapping          |
| `.glsl`, `.hlsl`    | Shader     | Delegated to shader_compiler                     |
| `.json`, `.ron`     | Data       | Validation and binary serialization              |
| `.ttf`, `.otf`      | Font       | SDF atlas generation                             |

## Usage

```bash
python cook.py --input assets/ --output build/cooked/ --platform windows
python cook.py --input assets/ --output build/cooked/ --platform ios --quality high
python cook.py --input assets/textures/hero.png --output build/cooked/ --platform windows
```

## Platform-Specific Cooking

Different platforms require different asset formats:

- **Windows/Linux**: BC7 textures, SPIR-V shaders, full-res meshes
- **macOS/iOS**: ASTC textures, Metal shaders, mesh LODs
- **Android**: ASTC/ETC2 textures, Vulkan SPIR-V shaders, aggressive LODs
- **Consoles**: Platform-specific texture formats per TRC/TCR requirements

## Caching

The cooker maintains a content-addressed cache in `.cook_cache/` to avoid re-processing unchanged assets. Cache keys are computed from:

- Source file content hash (SHA-256)
- Cooking settings hash
- Tool version
