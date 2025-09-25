# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
import subprocess
import re
from setuptools import setup
from datetime import datetime


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


# Environment and build configuration (similar to ttrt setup.py)
TTMLIR_VERSION_MAJOR = os.getenv("TTMLIR_VERSION_MAJOR", "0")
TTMLIR_VERSION_MINOR = os.getenv("TTMLIR_VERSION_MINOR", "0")
TTMLIR_VERSION_PATCH = os.getenv("TTMLIR_VERSION_PATCH", "0")

src_dir = os.environ.get(
    "SOURCE_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
)
# Use 'src_dir/build' as default location if TTMLIR_BINARY_DIR env variable is not available.
ttmlir_build_dir = os.environ.get(
    "TTMLIR_BINARY_DIR",
    os.path.join(src_dir, "build"),
)

metaldir = f"{src_dir}/third_party/tt-metal/src/tt-metal/build"
ttmetalhome = os.environ.get("TT_METAL_HOME", "")

# Set RPATH for runtime linking (similar to ttrt)
os.environ["LDFLAGS"] = "-Wl,-rpath,'$ORIGIN'"

# Feature flags - assume these are enabled for ttnn-jit
enable_runtime = True  # Always needed for ttnn-jit
enable_ttnn = True  # Always needed for ttnn-jit
enable_ttmetal = True  # Always needed for ttnn-jit
enable_ttnn_jit = True  # Always enabled for this package
enable_ttrt = True  # Always needed for ttnn-jit
arch = os.environ.get("CMAKE_SYSTEM_PROCESSOR", "x86_64")

# Runtime libraries that ttnn-jit needs
runtime_module = f"_ttmlir_runtime.cpython-311-{arch}-linux-gnu.so"
# ttnn-jit specific module
ttnn_jit_module = f"_ttnn_jit.cpython-311-{arch}-linux-gnu.so"
dylibs = []
runlibs = []
metallibs = []

# Core ttnn-jit libraries
dylibs += ["libTTMLIRRuntime.so", runtime_module]

# TTMLIR libraries (required for ttmlir package)
ttmlir_libs = [
    "_mlir.cpython-311-x86_64-linux-gnu.so",
    "_mlirDialectsLinalg.cpython-311-x86_64-linux-gnu.so",
    "_mlirDialectsQuant.cpython-311-x86_64-linux-gnu.so",
    "_mlirLinalgPasses.cpython-311-x86_64-linux-gnu.so",
    "_ttmlir.cpython-311-x86_64-linux-gnu.so",
    "libTTMLIRPythonCAPI.so",
]

# TTNN libraries (required for ttnn-jit)
if enable_ttnn:
    runlibs += ["_ttnncpp.so"]

# TT-Metal libraries (required for ttnn-jit)
if enable_ttmetal:
    runlibs += ["libtt_metal.so"]

# Common libraries for TTNN/TT-Metal
if enable_ttnn or enable_ttmetal:
    runlibs += ["libdevice.so"]
    runlibs += ["libtt_stl.so"]
    runlibs += ["libtracy.so.0.10.0"]

# Copy pre-built binaries to wheel directory
wheel_runtime_dir = f"{ttmlir_build_dir}/python_packages/ttnn_jit/runtime"

if enable_runtime:
    # Ensure runtime directory exists
    os.makedirs(wheel_runtime_dir, exist_ok=True)

    # Copy core runtime library
    shutil.copy(
        f"{ttmlir_build_dir}/runtime/lib/libTTMLIRRuntime.so",
        wheel_runtime_dir,
    )

    # Copy Python runtime module
    shutil.copy(
        f"{ttmlir_build_dir}/runtime/python/{runtime_module}",
        wheel_runtime_dir,
    )

    # Copy ttnn-jit specific module (if it exists)
    ttnn_jit_module_path = f"{ttmlir_build_dir}/tools/ttnn-jit/python/{ttnn_jit_module}"
    if os.path.exists(ttnn_jit_module_path):
        shutil.copy(ttnn_jit_module_path, wheel_runtime_dir)
        dylibs += [ttnn_jit_module]

    # Copy TTMLIR libraries from _mlir_libs directory
    ttmlir_libs_src_dir = f"{ttmlir_build_dir}/python_packages/ttmlir/_mlir_libs"
    # Keep the ttmlir libs in their original structure under ttmlir package
    ttmlir_libs_dst_dir = f"{ttmlir_build_dir}/python_packages/ttmlir/_mlir_libs"
    # No need to copy since they're already in the right location for packaging

    # Copy TT-Metal/TTNN libraries
    for runlib in runlibs:
        src_path = f"{metaldir}/lib/{runlib}"
        if os.path.exists(src_path):
            shutil.copy(src_path, wheel_runtime_dir)

    # Copy essential TT-Metal directories (similar to ttrt but more selective for ttnn-jit)
    essential_dirs = ["tt_metal", "runtime", "ttnn"]

    for dirname in essential_dirs:
        src_dir_path = f"{ttmetalhome}/{dirname}"
        dst_dir_path = f"{wheel_runtime_dir}/{dirname}"

        if os.path.exists(src_dir_path):
            shutil.copytree(src_dir_path, dst_dir_path, dirs_exist_ok=True)

            # Package files in this directory
            def package_files(directory):
                paths = []
                for path, directories, filenames in os.walk(directory):
                    for filename in filenames:
                        paths.append(os.path.join("..", path, filename))
                return paths

            metallibs += package_files(dst_dir_path)

dylibs += runlibs
dylibs += metallibs
# Note: ttmlir_libs are handled separately in package_data


def get_dynamic_version():
    """Get version from environment variables or generate based on tt-metal version"""
    # Use environment variables first (similar to ttrt)
    version = f"{TTMLIR_VERSION_MAJOR}.{TTMLIR_VERSION_MINOR}.{TTMLIR_VERSION_PATCH}"

    # If no version set via environment, generate one based on tt-metal version
    if version == "0.0.0":
        full_hash = extract_tt_metal_version()
        if full_hash:
            short_hash = full_hash[:8]  # First 8 chars
            return f"0.1.0+tt.{short_hash}"
        else:
            # Fallback version
            return "0.1.0.dev0"

    return version


def get_dynamic_dependencies():
    """Get dependencies needed for TTNN JIT functionality"""
    # Core dependencies for ttnn-jit (simplified, similar to ttrt approach)
    install_requires = [
        "nanobind",  # Python binding framework
        "numpy",  # Core numeric computing
        "torch",  # PyTorch for tensor operations
        "pyyaml",  # YAML parsing
        "loguru",  # Logging (used by TTNN)
        "tqdm",  # Progress bars
        "psutil",  # System info
        "graphviz",  # Visualization for JIT graphs
    ]

    return install_requires


def get_readme():
    """Get README content"""
    project_root = pathlib.Path(__file__).parent.parent.parent
    readme_path = project_root / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()

    return """# ttnn-jit

TTNN Just-In-Time Compilation Interface for TT-MLIR

This package provides a Python interface for just-in-time compilation of TTNN operations using the TT-MLIR compiler infrastructure.

## Installation

```bash
pip install ttnn-jit
```

## Usage

```python
import ttnn_jit
# Use ttnn-jit APIs
```

For more information, visit: https://docs.tenstorrent.com/tt-mlir/
"""


# Package configuration (using relative paths from setup.py location)
packages = [
    "ttnn_jit",
    "ttnn_jit._src",
    "ttnn_jit.runtime",
    # Include ttmlir package that ttnn_jit depends on
    "ttmlir",
    "ttmlir.dialects",
    "ttmlir.dialects.linalg",
    "ttmlir.dialects.linalg.opdsl",
    "ttmlir.dialects.linalg.passes",
    "ttmlir.extras",
    "ttmlir._mlir_libs",
    # Include ttnn as top-level package (required by dispatch_op.py)
    "ttnn",
    "ttnn.ttnn",
    "ttnn.ttnn.operations",
    "ttnn.ttnn.distributed",
    "ttnn.ttnn.examples",
    "ttnn.ttnn.examples.bert",
    "ttnn.ttnn.examples.usage",
    "ttnn.ttnn.experimental_loader",
    "ttnn.tt_lib",
    "ttnn.tt_lib._internal",
    "ttnn.tt_lib.fallback_ops",
    "ttnn.tt_lib.fused_ops",
    "ttnn.tracy",
    # Note: _mlir is a shared library, not a Python package directory
]

# Calculate relative paths from setup.py directory
setup_dir = os.path.dirname(os.path.abspath(__file__))
rel_build_dir = os.path.relpath(ttmlir_build_dir, setup_dir)

package_dir = {
    "ttnn_jit": f"{rel_build_dir}/python_packages/ttnn_jit",
    "ttnn_jit._src": f"{rel_build_dir}/python_packages/ttnn_jit/_src",
    "ttnn_jit.runtime": f"{rel_build_dir}/python_packages/ttnn_jit/runtime",
    # Include ttmlir package directories
    "ttmlir": f"{rel_build_dir}/python_packages/ttmlir",
    "ttmlir.dialects": f"{rel_build_dir}/python_packages/ttmlir/dialects",
    "ttmlir.dialects.linalg": f"{rel_build_dir}/python_packages/ttmlir/dialects/linalg",
    "ttmlir.dialects.linalg.opdsl": f"{rel_build_dir}/python_packages/ttmlir/dialects/linalg/opdsl",
    "ttmlir.dialects.linalg.passes": f"{rel_build_dir}/python_packages/ttmlir/dialects/linalg/passes",
    "ttmlir.extras": f"{rel_build_dir}/python_packages/ttmlir/extras",
    "ttmlir._mlir_libs": f"{rel_build_dir}/python_packages/ttmlir/_mlir_libs",
    # Map ttnn packages to the runtime ttnn directory
    "ttnn": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn",
    "ttnn.ttnn": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/ttnn",
    "ttnn.ttnn.operations": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/ttnn/operations",
    "ttnn.ttnn.distributed": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/ttnn/distributed",
    "ttnn.ttnn.examples": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/ttnn/examples",
    "ttnn.ttnn.examples.bert": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/ttnn/examples/bert",
    "ttnn.ttnn.examples.usage": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/ttnn/examples/usage",
    "ttnn.ttnn.experimental_loader": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/ttnn/experimental_loader",
    "ttnn.tt_lib": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/tt_lib",
    "ttnn.tt_lib._internal": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/tt_lib/_internal",
    "ttnn.tt_lib.fallback_ops": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/tt_lib/fallback_ops",
    "ttnn.tt_lib.fused_ops": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/tt_lib/fused_ops",
    "ttnn.tracy": f"{rel_build_dir}/python_packages/ttnn_jit/runtime/ttnn/tracy",
}

setup(
    name="ttnn-jit",
    version=get_dynamic_version(),
    install_requires=get_dynamic_dependencies(),
    author="Saber Gholami",
    author_email="sgholami@tenstorrent.com",
    url="https://github.com/tenstorrent/tt-mlir",
    description="TTNN Just-In-Time Compilation Interface for TT-MLIR",
    packages=packages,
    package_dir=package_dir,
    entry_points={
        "console_scripts": ["ttnn-jit = ttnn_jit:main"],
    },
    package_data={
        "ttnn_jit.runtime": dylibs,
        "ttmlir": ["*.py"],
        "ttmlir._mlir_libs": ["*.so", "*.py", "*.pyi"],
        "ttmlir.dialects": ["*.py"],
        "ttmlir.dialects.linalg": ["*.py"],
        "ttmlir.dialects.linalg.opdsl": ["*.py"],
        "ttmlir.dialects.linalg.passes": ["*.py"],
        "ttmlir.extras": ["*.py"],
        # Include ttnn package data
        "ttnn": ["*.py", "*.so", "*.md", "*.ipynb", "*.txt", "*.sh", "*.svg"],
        "ttnn.ttnn": ["*.py", "*.so"],
        "ttnn.ttnn.operations": ["*.py"],
        "ttnn.ttnn.distributed": ["*.py"],
        "ttnn.ttnn.examples": ["*.py"],
        "ttnn.ttnn.examples.bert": ["*.py", "*.svg"],
        "ttnn.ttnn.examples.usage": ["*.py"],
        "ttnn.ttnn.experimental_loader": ["*.py"],
        "ttnn.tt_lib": ["*.py"],
        "ttnn.tt_lib._internal": ["*.py"],
        "ttnn.tt_lib.fallback_ops": ["*.py"],
        "ttnn.tt_lib.fused_ops": ["*.py"],
        "ttnn.tracy": ["*.py"],
        # Note: _ttmlir_runtime is handled by data_files for top-level installation
    },
    zip_safe=False,
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
)
