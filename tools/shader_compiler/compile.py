#!/usr/bin/env python3
"""
Genovo Shader Compiler
======================

Compiles GLSL/HLSL shader sources to SPIR-V and platform-specific bytecode.

Usage:
    python compile.py --input <shader_dir> --output <output_dir> --target <platform>

Targets: vulkan, dx12, metal, opengl_es

TODO(BUILD): Implement shader compilation pipeline
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHADER_EXTENSIONS = {
    ".vert.glsl": "vert",
    ".frag.glsl": "frag",
    ".comp.glsl": "comp",
    ".geom.glsl": "geom",
    ".tesc.glsl": "tesc",
    ".tese.glsl": "tese",
    ".vert.hlsl": "vert",
    ".frag.hlsl": "frag",
    ".comp.hlsl": "comp",
}

INCLUDE_EXTENSIONS = {".glsl.inc", ".hlsl.inc"}


def find_tool(name: str) -> str | None:
    """Locate a tool on PATH."""
    # TODO(BUILD): Implement robust tool discovery (check Vulkan SDK env vars, etc.)
    import shutil
    return shutil.which(name)


def compile_glsl_to_spirv(input_path: Path, output_path: Path, stage: str) -> bool:
    """Compile a GLSL shader to SPIR-V using glslangValidator."""
    glslang = find_tool("glslangValidator")
    if not glslang:
        print(f"ERROR: glslangValidator not found on PATH", file=sys.stderr)
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    spirv_path = output_path.with_suffix(".spv")

    cmd = [
        glslang,
        "-V",                   # Compile for Vulkan
        "-S", stage,            # Shader stage
        "-o", str(spirv_path),  # Output path
        str(input_path),        # Input path
    ]

    print(f"  Compiling: {input_path} -> {spirv_path}")
    # TODO(BUILD): Execute compilation and handle errors
    # result = subprocess.run(cmd, capture_output=True, text=True)
    # if result.returncode != 0:
    #     print(f"  ERROR: {result.stderr}", file=sys.stderr)
    #     return False
    return True


def optimize_spirv(spirv_path: Path) -> bool:
    """Optimize a SPIR-V binary using spirv-opt."""
    spirv_opt = find_tool("spirv-opt")
    if not spirv_opt:
        print(f"WARNING: spirv-opt not found, skipping optimization", file=sys.stderr)
        return True

    cmd = [
        spirv_opt,
        "-O",                   # Optimize for performance
        str(spirv_path),
        "-o", str(spirv_path),  # In-place
    ]

    # TODO(BUILD): Execute optimization pass
    # result = subprocess.run(cmd, capture_output=True, text=True)
    return True


def cross_compile_spirv(spirv_path: Path, target: str, output_path: Path) -> bool:
    """Cross-compile SPIR-V to target language using spirv-cross."""
    spirv_cross = find_tool("spirv-cross")
    if not spirv_cross:
        print(f"ERROR: spirv-cross not found on PATH", file=sys.stderr)
        return False

    target_flags = {
        "metal":     ["--msl"],
        "opengl_es": ["--es", "--version", "310"],
        "dx12":      ["--hlsl", "--shader-model", "60"],
    }

    if target not in target_flags:
        print(f"ERROR: Unknown cross-compile target: {target}", file=sys.stderr)
        return False

    cmd = [
        spirv_cross,
        *target_flags[target],
        "--output", str(output_path),
        str(spirv_path),
    ]

    # TODO(BUILD): Execute cross-compilation
    # result = subprocess.run(cmd, capture_output=True, text=True)
    return True


def generate_metadata(spirv_path: Path, meta_path: Path) -> bool:
    """Generate shader metadata JSON from SPIR-V reflection."""
    # TODO(BUILD): Use spirv-cross --reflect to extract binding info
    metadata = {
        "source": str(spirv_path),
        "bindings": [],
        "push_constants": [],
        "inputs": [],
        "outputs": [],
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return True


def find_shaders(input_dir: Path) -> list[tuple[Path, str]]:
    """Find all shader source files and their stages."""
    shaders = []
    for ext, stage in SHADER_EXTENSIONS.items():
        for path in input_dir.rglob(f"*{ext}"):
            shaders.append((path, stage))
    return shaders


def compile_shaders(input_dir: Path, output_dir: Path, target: str) -> int:
    """Compile all shaders in the input directory."""
    shaders = find_shaders(input_dir)

    if not shaders:
        print(f"No shaders found in {input_dir}")
        return 0

    print(f"Found {len(shaders)} shader(s) to compile for target '{target}'")

    errors = 0
    for shader_path, stage in shaders:
        rel_path = shader_path.relative_to(input_dir)
        out_path = output_dir / rel_path

        # Step 1: GLSL -> SPIR-V
        if not compile_glsl_to_spirv(shader_path, out_path, stage):
            errors += 1
            continue

        spirv_path = out_path.with_suffix(".spv")

        # Step 2: Optimize SPIR-V
        optimize_spirv(spirv_path)

        # Step 3: Generate metadata
        generate_metadata(spirv_path, out_path.with_suffix(".meta.json"))

        # Step 4: Cross-compile if not targeting Vulkan
        if target != "vulkan":
            target_ext = {
                "metal": ".metal",
                "opengl_es": ".essl",
                "dx12": ".dxil",
            }.get(target, ".bin")
            cross_path = out_path.with_suffix(target_ext)
            if not cross_compile_spirv(spirv_path, target, cross_path):
                errors += 1

    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Genovo Shader Compiler")
    parser.add_argument("--input", "-i", required=True, help="Input shader directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--target", "-t",
        choices=["vulkan", "dx12", "metal", "opengl_es"],
        default="vulkan",
        help="Target platform (default: vulkan)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Genovo Shader Compiler")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Target: {args.target}")
    print()

    errors = compile_shaders(input_dir, output_dir, args.target)

    if errors > 0:
        print(f"\nCompleted with {errors} error(s)")
        sys.exit(1)
    else:
        print(f"\nAll shaders compiled successfully")


if __name__ == "__main__":
    main()
