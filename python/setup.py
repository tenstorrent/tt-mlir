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

        print("Working Directories: ", cwd, build_dir, install_dir)

        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            # Add the Source Directory (root)
            "-S",
            str(cwd.parent),
        ]

        self.spawn(["cmake", *cmake_args])
        self.spawn(["cmake", "--build", str(build_dir)])

        # Install the PythonWheel Component
        self.spawn(
            ["cmake", "--install", str(build_dir), "--component", "TTMLIRPythonWheel"]
        )

        # Remove empty pykernel dir
        self.rmdir(install_dir / "pykernel")


# Compute a dynamic version from git, taken from tt-forge-fe
short_hash = (
    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    .decode("ascii")
    .strip()
)
date = (
    subprocess.check_output(
        ["git", "show", "-s", "--format=%cd", "--date=format:%y%m%d", "HEAD"]
    )
    .decode("ascii")
    .strip()
)
version = "0.1." + date + "+dev." + short_hash

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
