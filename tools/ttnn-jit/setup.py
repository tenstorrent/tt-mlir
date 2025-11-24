# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
import subprocess
import re
from pathlib import Path
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
        install_dir = pathlib.Path(self.build_lib)

        # Check if we're in a wheel build environment (cibuildwheel)
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
            ]
            cmake_args.extend(["-S", str(source_dir)])
            print(f"Running CMake configure: {' '.join(cmake_args)}")
            self.spawn(cmake_args, env=build_env)

            build_cmd = ["cmake", "--build", str(build_dir), "--", "ttnn-jit"]
            print(f"Running CMake build: {' '.join(build_cmd)}")
            self.spawn(build_cmd, env=build_env)
            print("CMake build completed successfully")

        else:
            # In dev env.. assume build already exists through cmake flow
            if not build_dir.exists():
                raise RuntimeError(f"Build directory not found: {build_dir}\n")
            print(f"Using existing build directory: {build_dir}")

        self.spawn(
            [
                "cmake",
                "--install",
                str(build_dir),
                "--component",
                "TTNNJITWheel",
                "--prefix",
                str(install_dir),
            ]
        )

        if not self.dev_build:
            self._write_build_metadata()

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

    def _copy_mlir_so(self, build_dir: Path, arch: str):
        """Copy all required MLIR .so libraries to lib/"""
        print("\nCopying MLIR libraries to lib/...")

        lib_dest_dir = Path(self.build_lib) / "ttnn_jit" / "runtime"
        os.makedirs(lib_dest_dir, exist_ok=True)

        libraries_to_copy = [
            (
                build_dir / "python_packages" / "ttnn_jit",
                f"_ttnn_jit.{self.python_version}-{arch}-linux-gnu.so",
            ),
            (
                build_dir / "python_packages" / "ttmlir",
                f"_ttmlir.{self.python_version}-{arch}-linux-gnu.so",
            ),
            (build_dir / "runtime" / "lib", "libTTMLIRRuntime.so"),
            (
                build_dir / "runtime" / "python",
                f"_ttmlir_runtime.{self.python_version}-{arch}-linux-gnu.so",
            ),
            (build_dir / "tools" / "ttnn-jit" / "csrc", "libJITCPP.so"),
        ]
        rpath_patches = [
            "$ORIGIN",
            "$ORIGIN/../../ttnn/build/lib",
            "$ORIGIN/../../../../../../build/lib",
            "/usr/local/lib",
        ]
        if self.dev_build:
            rpath_patches.append(f"{self.tt_metal_home}/build/lib")
        for src_dir, lib_name in libraries_to_copy:
            src_file = src_dir / lib_name
            if src_file.exists():
                shutil.copy2(src_file, lib_dest_dir / lib_name)
                print(f"  Copied: {lib_name}")
                print(f"  Patching rpath: {rpath_patches}")
                subprocess.check_call(
                    [
                        "patchelf",
                        "--set-rpath",
                        ":".join(rpath_patches),
                        lib_dest_dir / lib_name,
                    ]
                )
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

    def _write_build_metadata(self):
        """Write metal git SHA to _build_metadata.py ONLY in the wheel, this info is picked up
        by __init__.py at runtime to check against TT_METAL_HOME git SHA to ensure compatibility.
        """
        metal_git_sha = extract_tt_metal_version()
        build_metadata_file = Path(self.build_lib) / "ttnn_jit" / "_build_metadata.py"
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
    python_requires=">=3.10",  # tt-metal uses python3.10
    zip_safe=False,
)
