#!/usr/bin/env python3
"""
Genovo Asset Cooker
===================

Processes raw source assets into optimized, platform-specific formats
for runtime loading by the engine.

Usage:
    python cook.py --input <asset_dir> --output <output_dir> --platform <platform>

Platforms: windows, linux, macos, ios, android, xbox, playstation

TODO(BUILD): Implement asset cooking pipeline
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Asset type registry
# ---------------------------------------------------------------------------

ASSET_TYPES = {
    # Meshes
    ".fbx":  "mesh",
    ".gltf": "mesh",
    ".glb":  "mesh",
    ".obj":  "mesh",
    # Textures
    ".png":  "texture",
    ".jpg":  "texture",
    ".jpeg": "texture",
    ".tga":  "texture",
    ".bmp":  "texture",
    ".exr":  "texture",
    ".hdr":  "texture",
    # Audio
    ".wav":  "audio",
    ".ogg":  "audio",
    ".mp3":  "audio",
    ".flac": "audio",
    # Fonts
    ".ttf":  "font",
    ".otf":  "font",
    # Data
    ".json": "data",
    ".ron":  "data",
    ".toml": "data",
    # Shaders (delegated to shader_compiler)
    ".glsl": "shader",
    ".hlsl": "shader",
}

# Quality presets per platform
QUALITY_PRESETS = {
    "windows":     {"texture_format": "bc7",  "max_texture_size": 4096, "mesh_lods": 3},
    "linux":       {"texture_format": "bc7",  "max_texture_size": 4096, "mesh_lods": 3},
    "macos":       {"texture_format": "astc", "max_texture_size": 4096, "mesh_lods": 3},
    "ios":         {"texture_format": "astc", "max_texture_size": 2048, "mesh_lods": 4},
    "android":     {"texture_format": "astc", "max_texture_size": 2048, "mesh_lods": 4},
    "xbox":        {"texture_format": "bc7",  "max_texture_size": 4096, "mesh_lods": 3},
    "playstation": {"texture_format": "gnf",  "max_texture_size": 4096, "mesh_lods": 3},
}


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file for cache invalidation."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def should_recook(source: Path, output: Path, cache_dir: Path) -> bool:
    """Check if an asset needs to be re-cooked based on content hash cache."""
    cache_file = cache_dir / (source.name + ".cache")
    if not output.exists():
        return True
    if not cache_file.exists():
        return True

    current_hash = compute_file_hash(source)
    try:
        cached_hash = cache_file.read_text().strip()
        return current_hash != cached_hash
    except Exception:
        return True


def update_cache(source: Path, cache_dir: Path):
    """Update the cook cache for a source asset."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (source.name + ".cache")
    cache_file.write_text(compute_file_hash(source))


def cook_texture(source: Path, output: Path, settings: dict) -> bool:
    """Cook a texture asset (resize, generate mipmaps, compress)."""
    # TODO(BUILD): Implement texture cooking
    #   1. Load source image
    #   2. Resize to max_texture_size if larger
    #   3. Generate mipmap chain
    #   4. Compress to target format (BC7, ASTC, etc.)
    #   5. Write .gasset binary
    print(f"  [texture] {source.name} -> {output.name} ({settings.get('texture_format', 'raw')})")
    return True


def cook_mesh(source: Path, output: Path, settings: dict) -> bool:
    """Cook a mesh asset (optimize, generate LODs, compute tangents)."""
    # TODO(BUILD): Implement mesh cooking
    #   1. Load source mesh (FBX/glTF via assimp or custom parser)
    #   2. Optimize vertex order (cache-friendly)
    #   3. Generate LOD chain
    #   4. Compute tangent space if missing
    #   5. Write .gasset binary with vertex/index buffers
    print(f"  [mesh] {source.name} -> {output.name} (LODs: {settings.get('mesh_lods', 1)})")
    return True


def cook_audio(source: Path, output: Path, settings: dict) -> bool:
    """Cook an audio asset (convert sample rate, compress)."""
    # TODO(BUILD): Implement audio cooking
    #   1. Load source audio
    #   2. Resample if needed
    #   3. Convert to platform format
    #   4. Write .gasset binary
    print(f"  [audio] {source.name} -> {output.name}")
    return True


def cook_font(source: Path, output: Path, settings: dict) -> bool:
    """Cook a font asset (generate SDF atlas)."""
    # TODO(BUILD): Implement font cooking (SDF atlas generation)
    print(f"  [font] {source.name} -> {output.name}")
    return True


def cook_data(source: Path, output: Path, settings: dict) -> bool:
    """Cook a data asset (validate and serialize)."""
    # TODO(BUILD): Implement data asset cooking
    print(f"  [data] {source.name} -> {output.name}")
    return True


def cook_shader(source: Path, output: Path, settings: dict) -> bool:
    """Delegate shader cooking to the shader compiler."""
    # TODO(BUILD): Invoke shader_compiler for shader assets
    print(f"  [shader] {source.name} -> (delegated to shader_compiler)")
    return True


COOKERS = {
    "texture": cook_texture,
    "mesh":    cook_mesh,
    "audio":   cook_audio,
    "font":    cook_font,
    "data":    cook_data,
    "shader":  cook_shader,
}


def find_assets(input_dir: Path) -> list[tuple[Path, str]]:
    """Discover all cookable assets in the input directory."""
    assets = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file():
            ext = path.suffix.lower()
            asset_type = ASSET_TYPES.get(ext)
            if asset_type:
                assets.append((path, asset_type))
    return assets


def cook_all(input_dir: Path, output_dir: Path, platform: str, force: bool = False) -> int:
    """Cook all assets in the input directory."""
    assets = find_assets(input_dir)
    if not assets:
        print(f"No assets found in {input_dir}")
        return 0

    settings = QUALITY_PRESETS.get(platform, QUALITY_PRESETS["windows"])
    cache_dir = input_dir / ".cook_cache" / platform

    print(f"Found {len(assets)} asset(s) to cook for '{platform}'")

    errors = 0
    skipped = 0

    for asset_path, asset_type in assets:
        rel_path = asset_path.relative_to(input_dir)
        out_path = output_dir / rel_path.with_suffix(".gasset")

        if not force and not should_recook(asset_path, out_path, cache_dir):
            skipped += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        cooker = COOKERS.get(asset_type)
        if cooker and cooker(asset_path, out_path, settings):
            update_cache(asset_path, cache_dir)
        else:
            errors += 1

    if skipped > 0:
        print(f"  Skipped {skipped} up-to-date asset(s)")

    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Genovo Asset Cooker")
    parser.add_argument("--input", "-i", required=True, help="Input asset directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--platform", "-p",
        choices=["windows", "linux", "macos", "ios", "android", "xbox", "playstation"],
        default="windows",
        help="Target platform (default: windows)",
    )
    parser.add_argument("--force", "-f", action="store_true", help="Force re-cook all assets")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Genovo Asset Cooker")
    print(f"  Input:    {input_dir}")
    print(f"  Output:   {output_dir}")
    print(f"  Platform: {args.platform}")
    print()

    errors = cook_all(input_dir, output_dir, args.platform, args.force)

    if errors > 0:
        print(f"\nCompleted with {errors} error(s)")
        sys.exit(1)
    else:
        print(f"\nAll assets cooked successfully")


if __name__ == "__main__":
    main()
