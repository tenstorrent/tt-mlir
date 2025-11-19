# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import pathlib
import shutil
import subprocess
import re
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Configuration Constants
PYTHON_VERSION = "cpython-310"
DEFAULT_ARCH = "x86_64"


def extract_tt_metal_version() -> str | None:
    """Extract TT_METAL_VERSION from third_party/CMakeLists.txt"""
    try:
        project_root = pathlib.Path(__file__).resolve().parent.parent.parent
        cmake_file = project_root / "third_party" / "CMakeLists.txt"
        with open(cmake_file, "r") as f:
            content = f.read()
            match = re.search(r'set\(TT_METAL_VERSION\s+"([^"]+)"\)', content)
            if match:
                return match.group(1)
    except Exception:
        pass
    return None


def get_dynamic_version():
    """Get version from environment variables and always include tt-metal version"""
    version_major = os.getenv("TTMLIR_VERSION_MAJOR", "0")
    version_minor = os.getenv("TTMLIR_VERSION_MINOR", "0")
    version_patch = os.getenv("TTMLIR_VERSION_PATCH", "0")
    base_version = f"{version_major}.{version_minor}.{version_patch}"

    full_hash = extract_tt_metal_version()
    if not full_hash:
        raise RuntimeError(
            "Could not extract TT_METAL_VERSION from third_party/CMakeLists.txt. "
            "This is required for building ttnn-jit package."
        )

    short_hash = full_hash[:8]
    if base_version == "0.0.0":
        return f"0.1.0+ttnn.{short_hash}"
    else:
        return f"{base_version}+ttnn.{short_hash}"


def expand_patterns(patterns):
    """
    Given a list of glob patterns with brace expansion (e.g. `*.{h,hpp}`),
    return a flat list of glob patterns with the braces expanded.
    """
    expanded = []
    for pattern in patterns:
        if "{" in pattern and "}" in pattern:
            pre = pattern[: pattern.find("{")]
            post = pattern[pattern.find("}") + 1 :]
            options = pattern[pattern.find("{") + 1 : pattern.find("}")].split(",")
            for opt in options:
                expanded.append(f"{pre}{opt}{post}")
        else:
            expanded.append(pattern)
    return expanded


def copy_tree_with_patterns(src_dir, dst_dir, patterns, exclude_files=[]):
    """Copy only files matching glob patterns from src_dir into dst_dir, excluding specified files"""
    exclude_files = set(exclude_files) if exclude_files else None

    for pattern in expand_patterns(patterns):
        full_pattern = os.path.join(src_dir, pattern)
        matched_files = glob.glob(full_pattern, recursive=True)
        print(f"Copying matched files for pattern '{pattern}': {len(matched_files)} files")
        
        for src_path in matched_files:
            if os.path.isdir(src_path):
                continue
            rel_path = os.path.relpath(src_path, src_dir)
            
            if exclude_files is not None:
                filename = os.path.basename(rel_path)
                if filename in exclude_files:
                    print(f"Excluding file: {rel_path}")
                    continue
                    
            dst_path = os.path.join(dst_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)


def get_is_srcdir_build():
    """Check if we're in a source directory build"""
    working_dir = Path(__file__).parent.parent.parent
    git_dir = working_dir / ".git"
    return git_dir.exists()


class CMakeBuild(build_ext):
    """
    Custom build_ext command that runs CMake to build tt-mlir,
    then copies the MLIR libraries and Python bindings into the wheel.
    
    Similar to tt-metal's approach but focused on MLIR components only.
    """
    
    @staticmethod
    def get_working_dir():
        """Get the project root directory"""
        working_dir = Path(__file__).resolve().parent.parent.parent
        assert working_dir.is_dir()
        return working_dir
    
    @staticmethod
    def get_build_env():
        """Get build environment variables"""
        return {
            **os.environ.copy(),
        }
    
    @staticmethod
    def get_arch():
        """Get system architecture"""
        return os.environ.get("CMAKE_SYSTEM_PROCESSOR", DEFAULT_ARCH)
    
    def run(self) -> None:
        """Build tt-mlir via CMake and copy MLIR libraries into wheel"""
        
        # Skip for editable installs (dev workflow)
        if self.is_editable_install_():
            raise Exception("Editable install not supported for ttnn-jit")
            # assert get_is_srcdir_build(), "Editable install detected in a non-srcdir environment, aborting"
            # print("Editable install detected, skipping CMake build")
            # return
        
        build_env = CMakeBuild.get_build_env()
        source_dir = CMakeBuild.get_working_dir()
        assert source_dir.is_dir(), f"Source dir {source_dir} seems to not exist"
        
        # Determine build type
        build_type = os.environ.get("CIBW_BUILD_TYPE", "Release")
        build_dir = source_dir / "build"
        
        # Check if we're in a wheel build environment (cibuildwheel)
        if "CIBUILDWHEEL" in os.environ:
            print("=" * 80)
            print("CIBUILDWHEEL DETECTED: Running full CMake configure and build")
            print("=" * 80)
            
            # CMake configuration arguments for wheel build
            cmake_args = [
                "cmake",
                "-B", str(build_dir),
                "-G", "Ninja",
                f"-DCMAKE_BUILD_TYPE={build_type}",
                "-DTTMLIR_ENABLE_RUNTIME=ON",
                "-DTTMLIR_ENABLE_TTNN_JIT=ON",
                "-DTTMLIR_ENABLE_BINDINGS_PYTHON=ON",
                "-DTTMLIR_ENABLE_OPMODEL=OFF",  # MLIR-only, skip OpModel for now
                "-DTTMLIR_ENABLE_TESTS=OFF",    # Skip tests in wheel build
            ]
            
            # Add optional flags from environment
            
            cmake_args.extend(["-S", str(source_dir)])
            
            print("Running CMake configure...")
            print(f"Command: {' '.join(cmake_args)}")
            subprocess.check_call(cmake_args, env=build_env)
            
            print("Running CMake build...")
            build_cmd = ["cmake", "--build", str(build_dir), "--parallel"]
            print(f"Command: {' '.join(build_cmd)}")
            subprocess.check_call(build_cmd, env=build_env)

            # install_cmd = ["cmake", "--install", str(build_dir), "--component", "SharedLib"]
            # print(f"Command: {' '.join(install_cmd)}")
            # subprocess.check_call(install_cmd, env=build_env)

            print("CMake build completed successfully")
        else:
            # Not in cibuildwheel - assume build already exists
            if not build_dir.exists():
                raise RuntimeError(
                    f"Build directory not found: {build_dir}\n"
                    "Please run 'cmake --build build' before building the wheel,\n"
                    "or set CIBUILDWHEEL=1 to trigger automatic CMake build."
                )
            print(f"Using existing build directory: {build_dir}")
        
        # Verbose sanity logging
        print("=" * 80)
        print("Build directory contents:")
        subprocess.check_call(["ls", "-lah", str(build_dir)], env=build_env)
        
        if (build_dir / "runtime" / "lib").exists():
            print("\nRuntime libraries:")
            subprocess.check_call(["ls", "-lah", str(build_dir / "runtime" / "lib")], env=build_env)
        
        if (build_dir / "python_packages").exists():
            print("\nPython packages:")
            subprocess.check_call(["ls", "-lah", str(build_dir / "python_packages")], env=build_env)
            subprocess.check_call(["find", str(build_dir / "python_packages"), "-name", "*.so"], env=build_env)
        print("=" * 80)
        
        # Copy MLIR .so libraries into ttnn_jit/lib/
        arch = CMakeBuild.get_arch()
        self._copy_mlir_so(build_dir, arch)
        self._copy_ttmlir_bindings(build_dir)
        print("Successfully copied all MLIR libraries")
    
    """
    These .so are installed into site-packages/ttnn_jit/runtime/lib
    ttnn wheel .so will be installed into site-packages/ttnn/build/lib
    also need to point to tt-metal build directory if they have an editable install
    where we assume they are using an env created by `create_venv.sh` script.

    Assumed structure starting from $TT_METAL_HOME:
    tt_metal/
    ├── python_env/
    │   └── lib/
    │       └── python3.10/
    │           └── site-packages/
    │               ├── ttnn/  (if installed via wheel)
    │               │   └── build/
    │               │       └── lib/
    │               │           ├── libtt_metal.so
    │               │           ├── _ttnncpp.so
    │               │           ├── libdevice.so
    │               │           └── libtt_stl.so
    │               │
    │               └── ttnn_jit/
    │                   ├── __init__.py
    │                   ├── api.py
    │                   ├── _src/
    │                   └── runtime/
    │                       ├── libTTMLIRRuntime.so
    │                       ├── _ttmlir_runtime.cpython-310-x86_64-linux-gnu.so
    │                       ├── _ttnn_jit.cpython-310-x86_64-linux-gnu.so
    │                       ├── libJITCPP.so
    │                       └── ttmlir/
    │                           ├── dialects/
    │                           ├── _mlir_libs/
    │                           ├── ir.py (and others)
    │                           └── passes.py
    │
    └── build/  (if using editable install via build_metal.sh)
        └── lib/
            ├── libtt_metal.so
            ├── _ttnncpp.so
            ├── libdevice.so
            ├── libtt_stl.so
            └── libtracy.so.0.10.0
    """
    def _copy_mlir_so(self, build_dir: Path, arch: str):
        """Copy all required MLIR .so libraries to lib/"""
        print("\nCopying MLIR libraries to lib/...")
        
        lib_dest_dir = Path(self.build_lib) / "ttnn_jit" / "runtime"
        os.makedirs(lib_dest_dir, exist_ok=True)
        
        libraries_to_copy = [
            # (source_path, library_name)
            (build_dir / "python_packages" / "ttnn_jit", f"_ttnn_jit.{PYTHON_VERSION}-{arch}-linux-gnu.so"),
            (build_dir / "python_packages" / "ttmlir", f"_ttmlir.{PYTHON_VERSION}-{arch}-linux-gnu.so"),
            (build_dir / "runtime" / "lib", "libTTMLIRRuntime.so"),
            (build_dir / "runtime" / "python", f"_ttmlir_runtime.{PYTHON_VERSION}-{arch}-linux-gnu.so"),
            (build_dir / "tools" / "ttnn-jit" / "csrc", "libJITCPP.so"),
        ]
        for src_dir, lib_name in libraries_to_copy:
            src_file = src_dir / lib_name
            if src_file.exists():
                shutil.copy2(src_file, lib_dest_dir / lib_name)
                print(f"  Copied: {lib_name}")
                rpath_patches = [
                    "$ORIGIN",
                    "$ORIGIN/../../ttnn/build/lib",
                    "$ORIGIN/../../../../../../build/lib",
                    "/usr/local/lib"
                ]
                print(f"  Patching rpath: {rpath_patches}")
                subprocess.check_call(["patchelf", "--set-rpath", ":".join(rpath_patches), lib_dest_dir / lib_name])
            else:
                print(f"  Warning: {lib_name} not found at {src_file}")
    
    def _copy_ttmlir_bindings(self, build_dir: Path):
        """Copy all required TTMLIR bindings to lib/"""
        print("\nCopying TTMLIR bindings to lib/...")
        lib_dest_dir = Path(self.build_lib) / "ttnn_jit" / "runtime" / "ttmlir"
        os.makedirs(lib_dest_dir, exist_ok=True)
        
        ttmlir_src_dir = build_dir / "python_packages" / "ttmlir"
        if ttmlir_src_dir.exists():
            for item in ttmlir_src_dir.iterdir():
                src_path = ttmlir_src_dir / item.name
                dest_path = lib_dest_dir / item.name
                
                if item.is_file():
                    print(f"    Copying {item.name}")
                    shutil.copy2(src_path, dest_path)
                elif item.is_dir():
                    print(f"    Copying directory {item.name}/")
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(src_path, dest_path)
        else:
            raise RuntimeError(f"TTMLIR bindings not found at {ttmlir_src_dir}")
        
    

    def is_editable_install_(self):
        """Check if this is an editable install"""
        return self.inplace


def get_readme():
    """Get README content"""
    project_root = pathlib.Path(__file__).parent.parent.parent
    readme_path = project_root / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()

    return """# ttnn-jit
TTNN Just-In-Time Compilation Interface for TT-MLIR

This package provides a Python interface for just-in-time compilation 
of TTNN operations using the TT-MLIR compiler infrastructure.

## Installation
pip install ttnn-jit## Usage
import ttnn_jit
# Use ttnn-jit APIsFor more information, visit: https://docs.tenstorrent.com/tt-mlir/
"""

# Dummy extension to force build_ext to run
ttnn_jit_ext = Extension("ttnn_jit._build_trigger", sources=[])

setup(
    name="ttnn-jit",
    version=get_dynamic_version(),
    author="Tenstorrent",
    author_email="info@tenstorrent.com",
    url="https://github.com/tenstorrent/tt-mlir",
    description="TTNN Just-In-Time Compilation Interface for TT-MLIR",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    # Python source files are in current directory (tools/ttnn-jit/)
    packages=["ttnn_jit", "ttnn_jit._src"],
    package_dir={"ttnn_jit": "."},
    ext_modules=[ttnn_jit_ext],
    cmdclass={"build_ext": CMakeBuild}
)