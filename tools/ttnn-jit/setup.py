# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
import subprocess
import re
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

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
        raise RuntimeError(
            "Could not extract TT_METAL_VERSION from third_party/CMakeLists.txt. "
            "Please ensure tt-metal is downloaded."
        )
    return None


def get_dynamic_version():
    """Get version from environment variables and always include tt-metal version"""
    version_major = os.getenv("TTMLIR_VERSION_MAJOR", "0")
    version_minor = os.getenv("TTMLIR_VERSION_MINOR", "0")
    version_patch = os.getenv("TTMLIR_VERSION_PATCH", "0")
    base_version = f"{version_major}.{version_minor}.{version_patch}"
    if base_version == "0.0.0":
        # 0.1.0 to indicate pre-release..
        return "0.1.0"
    return base_version


class CMakeBuild(build_ext):
    """
    Custom build_ext command that runs CMake to build tt-mlir,
    then copies the MLIR libraries and Python bindings into the wheel.
    """

    @staticmethod
    def get_python_version() -> str:
        """
        Parse TTMLIR_PYTHON_VERSION environment variable. This is needed bc tt-metal will use Python3.10 while tt-mlir will use Python3.11.
            eg: TTMLIR_PYTHON_VERSION="python3.10" -> "cpython-310"
                TTMLIR_PYTHON_VERSION="3.10" -> "cpython-310"
        """
        python_version_env = os.getenv("TTMLIR_PYTHON_VERSION", "python3.11")
        match = re.search(r"(\d+)\.(\d+)", python_version_env)

        if match:
            major = match.group(1)
            minor = match.group(2)
            return f"cpython-{major}{minor}"

        return "cpython-311"

    @staticmethod
    def get_working_dir():
        """Get the project root directory"""
        working_dir = pathlib.Path(__file__).resolve().parent.parent.parent
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
        """Build tt-mlir with TTNN-JIT enabled and copy MLIR libraries into wheel"""

        self.tt_metal_home = os.environ.get("TT_METAL_HOME", "")
        assert self.tt_metal_home, "TT_METAL_HOME is not set"
        self.dev_build = os.environ.get("TTMLIR_DEV_BUILD", "OFF") == "ON"
        self.python_version = self.get_python_version()
        # Skip for editable installs (dev workflow)
        if self.is_editable_install_():
            raise Exception("Editable install not supported for ttnn-jit")

        build_env = CMakeBuild.get_build_env()
        source_dir = CMakeBuild.get_working_dir()
        assert source_dir.is_dir(), f"Source dir {source_dir} seems to not exist"

        build_type = "Release"
        build_dir = source_dir / "build"
        self.install_dir = pathlib.Path(self.build_lib)

        if not self.dev_build:
            print("=" * 80)
            print("Running full CMake configure and build")
            print("=" * 80)

            # CMake configuration arguments for wheel build
            cmake_args = [
                "cmake",
                "-B",
                str(build_dir),
                "-G",
                "Ninja",
                f"-DCMAKE_BUILD_TYPE={build_type}",
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DTTMLIR_ENABLE_RUNTIME=ON",
                "-DTTMLIR_ENABLE_TTNN_JIT=ON",
                "-DTTMLIR_ENABLE_BINDINGS_PYTHON=ON",
                "-DTTMLIR_ENABLE_TTRT=OFF",
                "-DTTMLIR_ENABLE_OPMODEL=OFF",
                "-DTTMLIR_ENABLE_TESTS=OFF",
                "-DTTMLIR_ENABLE_ALCHEMIST=OFF",
            ]
            cmake_args.extend(["-S", str(source_dir)])
            print(f"Running CMake configure: {' '.join(cmake_args)}")
            subprocess.check_call(cmake_args, env=build_env)

            build_cmd = ["cmake", "--build", str(build_dir), "--", "ttnn-jit-deps"]
            print(f"Running CMake build: {' '.join(build_cmd)}")
            subprocess.check_call(build_cmd, env=build_env)
            print("CMake build completed successfully")

        else:
            # In dev env.. assume build already exists through cmake flow
            if not build_dir.exists():
                raise RuntimeError(f"Build directory not found: {build_dir}\n")
            print(f"Using existing build directory: {build_dir}")

        subprocess.check_call(
            [
                "cmake",
                "--install",
                str(build_dir),
                "--component",
                "TTNNJITWheel",
                "--prefix",
                str(self.install_dir),
            ],
            env=build_env,
        )

        if not self.dev_build:
            self.write_build_metadata()

    def write_build_metadata(self):
        """Write metal git SHA to _build_metadata.py ONLY in the wheel, this info is picked up
        by __init__.py at runtime to check against TT_METAL_HOME git SHA to ensure compatibility.
        """
        metal_git_sha = extract_tt_metal_version()
        build_metadata_file = self.install_dir / "ttnn_jit" / "_build_metadata.py"
        with open(build_metadata_file, "w") as f:
            f.write(f'METAL_GIT_SHA = "{metal_git_sha}"\n')

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
# Use ttnn-jit APIs
# For more information, visit: https://docs.tenstorrent.com/tt-mlir/
"""


# Dummy extension to force build_ext to run
ttnn_jit_ext = Extension("ttnn_jit._build_trigger", sources=[])

"""
Installed wheel in a metal dev env starting from TT_METAL_HOME:
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
│                           ├── ir.py
│                           └── passes.py (and others)
│
└── build/  (if using editable install via build_metal.sh)
    └── lib/
        ├── libtt_metal.so
        ├── _ttnncpp.so
        ├── libdevice.so
        ├── libtt_stl.so
        └── libtracy.so.0.10.0
"""

setup(
    name="ttnn-jit",
    version=get_dynamic_version(),
    author="Vincent Tang",
    author_email="vtang@tenstorrent.com",
    url="https://github.com/tenstorrent/tt-mlir",
    description="TTNN Just-In-Time Compilation Interface for TT-MLIR",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    packages=["ttnn_jit", "ttnn_jit._src"],
    package_dir={"ttnn_jit": "."},
    ext_modules=[ttnn_jit_ext],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=["ttmlir"],
    python_requires=">=3.10",  # tt-metal uses python3.10
    zip_safe=False,
)
