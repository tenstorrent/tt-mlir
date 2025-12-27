# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

THIS_DIR = Path(os.path.realpath(os.path.dirname(__file__)))
REPO_DIR = (THIS_DIR / ".." / "..").resolve()


# Global config to store build options
class BuildConfig:
    enable_perf = "ON"  # Default to ON for standalone wheel builds
    enable_runtime_debug = "OFF"


def get_cmake_options() -> dict:
    """Get CMake build options from build config."""
    return {
        "CMAKE_BUILD_TYPE": "Release",
        "TTMLIR_ENABLE_RUNTIME": "ON",
        "TT_RUNTIME_ENABLE_TTNN": "ON",
        "TT_RUNTIME_ENABLE_TTMETAL": "ON",
        "TT_RUNTIME_ENABLE_PERF_TRACE": BuildConfig.enable_perf,
        "TTMLIR_ENABLE_RUNTIME_TESTS": "OFF",
        "TT_RUNTIME_DEBUG": BuildConfig.enable_runtime_debug,
    }


def get_version() -> str:
    """Get version string from environment or default."""
    major = os.getenv("TTMLIR_VERSION_MAJOR", "0")
    minor = os.getenv("TTMLIR_VERSION_MINOR", "0")
    patch = os.getenv("TTMLIR_VERSION_PATCH", "0")
    return f"{major}.{minor}.{patch}"


class BdistWheel(bdist_wheel):
    """Custom wheel builder for platform-specific package."""

    user_options = bdist_wheel.user_options + [
        ("enable-perf=", None, "Enable perf trace: ON or OFF (default: ON)"),
        (
            "enable-runtime-debug=",
            None,
            "Enable runtime debug: ON or OFF (default: OFF)",
        ),
    ]

    def initialize_options(self):
        super().initialize_options()
        # Default values
        self.enable_perf = "ON"
        self.enable_runtime_debug = "OFF"

    def finalize_options(self):
        super().finalize_options()
        # Validate and store options in BuildConfig
        self.enable_perf = self.enable_perf.upper()
        self.enable_runtime_debug = self.enable_runtime_debug.upper()

        if self.enable_perf not in ["ON", "OFF"]:
            raise ValueError(
                f"Invalid --enable-perf value: {self.enable_perf}. Must be ON or OFF"
            )
        if self.enable_runtime_debug not in ["ON", "OFF"]:
            raise ValueError(
                f"Invalid --enable-runtime-debug value: {self.enable_runtime_debug}. Must be ON or OFF"
            )

        BuildConfig.enable_perf = self.enable_perf
        BuildConfig.enable_runtime_debug = self.enable_runtime_debug

        # Mark wheel as platform-specific (contains native binaries)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = super().get_tag()
        # Force specific Python version tag
        py_major, py_minor, _ = platform.python_version_tuple()
        python = f"cp{py_major}{py_minor}"
        abi = f"cp{py_major}{py_minor}"
        return python, abi, plat


class CMakeBuildPy(build_py):
    """
    Custom build_py command that runs CMake to build and install ttrt.

    For wheel builds:
    1. Configures CMake with all features enabled (runtime, TTNN, TTMetal, perf)
    2. Builds the project
    3. Installs SharedLib component to build_lib/ttrt/runtime/
    4. Copies _ttmlir_runtime.so separately

    For editable installs:
    - Skips the build (relies on CMake-managed symlinks)
    """

    def run(self):
        # Check if in editable mode
        if hasattr(self, "editable_mode") and self.editable_mode:
            print("Editable mode: skipping CMake build")
            super().run()
            return

        print("Building ttrt wheel with CMake")

        # Determine build directory - use separate dir for wheel builds
        # to avoid conflicts with main build
        build_dir = (REPO_DIR / "build-ttrt-wheel").resolve()

        # Run CMake configure, build, and install
        self.run_cmake_build(build_dir)

        # Continue with Python build
        super().run()

    def run_cmake_build(self, build_dir: Path):
        """Run CMake configure, build, and install."""
        # Install directory for runtime binaries
        install_dir = (Path(self.build_lib) / "ttrt" / "runtime").resolve()
        install_dir.mkdir(parents=True, exist_ok=True)

        # Get CMake options
        cmake_options = get_cmake_options()

        # Configure CMake with install prefix
        configure_command = [
            "cmake",
            "-G",
            "Ninja",
            "-B",
            str(build_dir),
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]

        # Add CMake options
        for key, value in cmake_options.items():
            configure_command.append(f"-D{key}={value}")

        print(f"Configuring CMake: {' '.join(configure_command)}")
        subprocess.check_call(configure_command, cwd=REPO_DIR)

        # Build
        build_command = ["cmake", "--build", str(build_dir)]
        print(f"Building: {' '.join(build_command)}")
        subprocess.check_call(build_command, cwd=REPO_DIR)

        # Install SharedLib component (prefix already set during configure)
        install_command = [
            "cmake",
            "--install",
            str(build_dir),
            "--component",
            "SharedLib",
        ]
        print(f"Installing: {' '.join(install_command)}")
        subprocess.check_call(install_command, cwd=REPO_DIR)

        # Copy _ttmlir_runtime.so separately
        runtime_so = self._find_runtime_module(build_dir)
        if runtime_so and runtime_so.exists():
            dest = install_dir / runtime_so.name
            print(f"Copying {runtime_so.name} to {install_dir}")
            shutil.copy(runtime_so, dest)
        else:
            raise RuntimeError(
                f"_ttmlir_runtime.so not found in {build_dir}/runtime/python/"
            )

    def _find_runtime_module(self, build_dir: Path) -> Path:
        """Find _ttmlir_runtime.cpython-*.so in build directory."""
        runtime_python_dir = build_dir / "runtime" / "python"
        if not runtime_python_dir.exists():
            return None

        for so_file in runtime_python_dir.glob("_ttmlir_runtime.cpython-*.so"):
            return so_file
        return None


# Get version
version = get_version()

# All requirements (perf tools always included)
requirements = [
    "nanobind",
    "loguru",
    "pandas",
    "seaborn",
    "graphviz",
    "pyyaml",
    "click",
]

# Setup package directories (must be relative paths)
packages = ["ttrt", "ttrt.common", "ttrt.binary", "ttrt.runtime"]
package_dir = {
    "ttrt": ".",
    "ttrt.common": "common",
    "ttrt.binary": "binary",
    "ttrt.runtime": "runtime",
}

setup(
    name="ttrt",
    version=version,
    author="Tenstorrent",
    author_email="nsmith@tenstorrent.com",
    url="https://github.com/tenstorrent/tt-mlir",
    description="Python bindings to tt-mlir runtime libraries",
    long_description="",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
    ],
    cmdclass={
        "bdist_wheel": BdistWheel,
        "build_py": CMakeBuildPy,
    },
    entry_points={
        "console_scripts": ["ttrt = ttrt:main"],
    },
    install_requires=requirements,
    packages=packages,
    package_dir=package_dir,
    python_requires=">=3.7",
    zip_safe=False,
)
