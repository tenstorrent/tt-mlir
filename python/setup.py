# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# TTMLIR Wheel setup.py
# Heavily inspired by: https://github.com/tenstorrent/tt-forge-fe/blob/main/setup.py

import os
import pathlib
import shutil
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from datetime import datetime


class TTExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if "ttmlir" in ext.name:
                self.build_(ext)
            else:
                raise Exception("Unknown extension")

    def rmdir(self, _dir: pathlib.Path):
        if _dir.exists():
            shutil.rmtree(_dir)

    def build_(self, ext):
        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            # Might be an editable install or something else
            return

        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        print(f"Running cmake to install ttmlir at {extension_path}")

        cwd = pathlib.Path().absolute()
        build_dir = cwd.parent / "build"

        # Set it to install directly into the wheel, so there's no need to raise the directory for ttmlir python files
        install_dir = extension_path.parent

        # Check for flatbuffers library path in environment or use a default
        flatbuffers_lib_dir = os.environ.get(
            "FLATBUFFERS_LIB_DIR", "/opt/ttmlir-toolchain/lib"
        )

        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            f"-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
            f"-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
        ]

        # CD Into root instead
        subprocess.run(
            " ".join(
                [
                    "cd",
                    str(cwd.parent),
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

        self.spawn(["cmake", "--build", str(build_dir), "--", "-j4"])

        # Install the PythonWheel Component
        self.spawn(
            ["cmake", "--install", str(build_dir), "--component", "TTMLIRPythonWheel"]
        )

        # Remove empty pykernel dir
        self.rmdir(install_dir / "pykernel")


date = datetime.now().strftime("%y.%m.%d")
version = "0.1." + date + ".dev0"

# Only the ttmlir package relies on the CMake build process
ttmlir_c = TTExtension("ttmlir")

setup(
    name="ttmlir",
    version=version,
    install_requires=[],
    # Include both pykernel and ttmlir as top-level packages
    packages=["ttmlir"],
    # Map the package names to their locations.
    # "." will include files in the current source (not needed)
    # We delete all of this during the build_ext step
    package_dir={"ttmlir": "."},
    ext_modules=[ttmlir_c],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)

VALID_AL9_BUILD_SCRIPT = """
#!/bin/bash
set -exo pipefail

dnf check-update || true
dnf install -y epel-release
dnf config-manager --set-enabled crb

dnf install -y \
    gcc-c++ make cmake ninja-build pkgconf-pkg-config ccache \
    clang \
    git wget curl jq sudo patch unzip \
    hwloc-devel tbb-devel capstone-devel \
    yaml-cpp-devel boost-devel libcurl-devel \
    pandoc doxygen graphviz patchelf lcov perf \
    xz


dnf clean all
clang --version

# Update ninja
echo "Attempting to install latest Ninja build tool..."
NINJA_VERSION="1.11.1" # Check https://github.com/ninja-build/ninja/releases for latest
NINJA_URL="https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip"
NINJA_ZIP="ninja-linux.zip"

# Use curl to download (already installed)
curl -L -o "${NINJA_ZIP}" "${NINJA_URL}"
# Need unzip - add 'unzip' to the dnf install list above!
unzip "${NINJA_ZIP}" -d /usr/local/bin/
# Make sure it's executable
chmod +x /usr/local/bin/ninja
rm -f "${NINJA_ZIP}"

echo "Installed Ninja version:"
/usr/local/bin/ninja --version
# Ensure /usr/local/bin is early in the PATH if the system ninja wasn't removed
# The CIBW_ENVIRONMENT PATH setting should already handle this if /usr/local/bin is standard.
# Verify which ninja will be used:
which ninja

# Need to build environment from here

# Clean potential copied artifacts
rm -rf env/build
rm -rf build

mkdir -p /opt/ttmlir-toolchain
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

cd /project
cmake -B env/build env
cmake --build env/build

source env/activate
"""

VALID_AL8_BUILD_SCRIPT = """
#!/bin/bash
set -exo pipefail

dnf check-update || true
dnf install -y epel-release
dnf config-manager --set-enabled powertools

dnf install -y \
    gcc-c++ make cmake ninja-build pkgconf-pkg-config ccache \
    clang \
    git wget curl jq sudo patch \
    hwloc-devel tbb-devel capstone-devel \
    yaml-cpp-devel boost-devel libcurl-devel \
    pandoc doxygen graphviz patchelf lcov perf

dnf clean all
# Verify clang version from the new source
clang --version

# Update ninja
echo "Attempting to install latest Ninja build tool..."
NINJA_VERSION="1.11.1" # Check https://github.com/ninja-build/ninja/releases for latest
NINJA_URL="https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip"
NINJA_ZIP="ninja-linux.zip"

# Use curl to download (already installed)
curl -L -o "${NINJA_ZIP}" "${NINJA_URL}"
# Need unzip - add 'unzip' to the dnf install list above!
unzip "${NINJA_ZIP}" -d /usr/local/bin/
# Make sure it's executable
chmod +x /usr/local/bin/ninja
rm -f "${NINJA_ZIP}"

echo "Installed Ninja version:"
/usr/local/bin/ninja --version
# Ensure /usr/local/bin is early in the PATH if the system ninja wasn't removed
# The CIBW_ENVIRONMENT PATH setting should already handle this if /usr/local/bin is standard.
# Verify which ninja will be used:
which ninja

# Need to build environment from here

# Clean potential copied artifacts
rm -rf env/build
rm -rf build

mkdir -p /opt/ttmlir-toolchain
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

cd /project
cmake -B env/build env
cmake --build env/build

source env/activate
"""
