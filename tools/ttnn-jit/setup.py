# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
import subprocess
import re
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
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


class TTNNJITExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if "ttnn_jit" in ext.name:
                self.build_(ext)
            else:
                raise Exception("Unknown extension")

    def in_ci(self) -> bool:
        return os.environ.get("IN_CIBW_ENV") == "ON"

    def get_tt_metal_version(self) -> str:
        """Extract TT_METAL_VERSION from third_party/CMakeLists.txt"""
        version = extract_tt_metal_version()
        if version is None:
            raise RuntimeError(
                "Could not find TT_METAL_VERSION in third_party/CMakeLists.txt"
            )
        return version

    def get_project_root(self) -> pathlib.Path:
        """Get the project root (tt-mlir directory)"""
        return pathlib.Path(__file__).resolve().parent.parent.parent

    def build_(self, ext):
        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            return

        cwd = pathlib.Path().absolute()
        build_dir = cwd.parent.parent / "build"

        # Set install directory for wheel
        install_dir = pathlib.Path(self.build_lib)
        if self.in_ci():
            install_dir = cwd / "build" / install_dir.name

        print(f"Building ttnn-jit wheel with:")
        print(f"  Build dir: {build_dir}")
        print(f"  Install dir: {install_dir}")
        print(f"  TT-Metal version: {self.get_tt_metal_version()}")
        print(f"  CWD: {cwd}")

        # Clean build directory to avoid generator conflicts
        cmake_cache = build_dir / "CMakeCache.txt"
        cmake_files = build_dir / "CMakeFiles"
        if cmake_cache.exists():
            print(f"  Removing CMake cache: {cmake_cache}")
            cmake_cache.unlink()
        if cmake_files.exists():
            print(f"  Removing CMake files: {cmake_files}")
            shutil.rmtree(cmake_files)

        # Configure CMake for ttnn-jit (needs TTNN runtime)
        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DTTMLIR_ENABLE_TTNN_JIT=ON",  # Enable ttnn-jit build
            "-DTTMLIR_ENABLE_RUNTIME=ON",  # Need runtime for TTNN
            "-DTT_RUNTIME_ENABLE_TTNN=ON",  # Need TTNN specifically
            "-DTT_RUNTIME_ENABLE_TTMETAL=ON",  # Likely needed for TTNN
            "-DTTMLIR_ENABLE_TTRT=ON",  # Runtime tools
            "-DTTMLIR_ENABLE_RUNTIME_TESTS=OFF",
            "-DTTMLIR_ENABLE_STABLEHLO=OFF",
            "-DTTMLIR_ENABLE_OPMODEL=OFF",
            "-DTTMLIR_ENABLE_EXPLORER=OFF",
        ]

        # Set source like pykernel does
        if not self.in_ci():
            cmake_args.extend(["-S", str(cwd.parent.parent)])  # Project root

        # Configure CMake following pykernel pattern
        if self.in_ci():
            subprocess.run(
                " ".join(
                    [
                        "cd",
                        str(cwd.parent.parent),
                        "&&",
                        "source",
                        "env/activate",
                        "&&",
                        "cmake",
                        *cmake_args,
                    ]
                ),
                shell=True,
                check=True,
            )
        else:
            self.spawn(["cmake", *cmake_args])

        # Build ttnn-jit and dependencies
        self.spawn(["cmake", "--build", str(build_dir)])

        # Install components needed for ttnn-jit
        install_components = [
            "TTMLIRPythonWheel",  # MLIR Python bindings
            "SharedLib",  # Runtime shared libraries
        ]

        for component in install_components:
            print(f"Installing component: {component}")
            self.spawn(["cmake", "--install", str(build_dir), "--component", component])


def get_dynamic_version():
    """Generate PEP 440 compliant version based on date + tt-metal version"""
    # Use simpler date format: YYYYMMDD
    date = datetime.now().strftime("%Y%m%d")

    # Try to get short hash of tt-metal version for local version identifier
    full_hash = extract_tt_metal_version()
    if full_hash:
        short_hash = full_hash[:8]  # First 8 chars
        # Use local version identifier format: major.minor.micro+local
        return f"0.1.{date}+tt.{short_hash}"

    # Fallback version
    return f"0.1.{date}.dev0"


def get_dynamic_dependencies():
    """Get dependencies needed for TTNN functionality"""
    # ttnn-jit needs TTNN runtime dependencies (not dev dependencies)

    # Runtime dependencies needed by TTNN
    runtime_deps = [
        "loguru==0.6.0",  # Required by TTNN (exact version from tt-metal)
        "numpy",  # Core numeric computing
        "torch",  # PyTorch (will use latest compatible)
        "pyyaml",  # YAML parsing (used by TTNN configs)
        "tqdm",  # Progress bars (used by TTNN)
        "psutil",  # System info (used by TTNN)
        "graphviz",  # Visualization (for ttnn-jit)
        "libnuma-dev",
    ]

    # Try to get additional runtime deps from tt-metal requirements
    try:
        project_root = pathlib.Path(__file__).parent.parent.parent
        metal_req_path = (
            project_root
            / "third_party"
            / "tt-metal"
            / "src"
            / "tt-metal"
            / "tt_metal"
            / "python_env"
            / "requirements-dev.txt"
        )

        if metal_req_path.exists():
            # Look for additional runtime deps (not dev tools)
            dev_tools = {
                "pre-commit",
                "black",
                "clang-format",
                "build",
                "twine",
                "yamllint",
                "mypy",
                "pytest",
                "sphinx",
                "flake8",
            }

            with open(metal_req_path, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip git+, -r, comments, empty lines, and dev tools
                    if (
                        line
                        and not line.startswith("git+")
                        and not line.startswith("-r")
                        and not line.startswith("#")
                        and not any(tool in line.lower() for tool in dev_tools)
                    ):

                        # Clean up platform-specific tags (e.g., +cpu, +cu118)
                        clean_line = line
                        if "+" in clean_line and "==" in clean_line:
                            # Remove platform tags like +cpu, +cu118 from version specifiers
                            pkg_name, version_spec = clean_line.split("==", 1)
                            if "+" in version_spec:
                                version_spec = version_spec.split("+")[0]
                                clean_line = f"{pkg_name}=={version_spec}"

                        # Extract package name to check if it's already in runtime_deps
                        pkg_name = (
                            clean_line.split("==")[0]
                            .split(">=")[0]
                            .split("<=")[0]
                            .split("<")[0]
                            .split(">")[0]
                        )
                        if pkg_name not in [
                            dep.split("==")[0]
                            .split(">=")[0]
                            .split("<=")[0]
                            .split("<")[0]
                            .split(">")[0]
                            for dep in runtime_deps
                        ]:
                            runtime_deps.append(clean_line)

    except Exception as e:
        print(f"Warning: Could not extract tt-metal requirements: {e}")

    return runtime_deps


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


# Create extension for CMake build
ttnn_jit_ext = TTNNJITExtension("ttnn_jit")

setup(
    name="ttnn-jit",
    version=get_dynamic_version(),
    install_requires=get_dynamic_dependencies(),
    author="Saber Gholami",
    author_email="sgholami@tenstorrent.com",
    # Include ttnn_jit and its subpackages
    packages=["ttnn_jit", "ttnn_jit._src"],
    package_dir={"ttnn_jit": ""},
    # Include all Python files and subdirectories
    package_data={"ttnn_jit": ["*.py", "_src/*.py"]},
    ext_modules=[ttnn_jit_ext],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    long_description=get_readme(),
    long_description_content_type="text/markdown",
)
