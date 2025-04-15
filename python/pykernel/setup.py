# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# PyKernel Wheel setup.py
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
        print(f"Running cmake to install pykernel at {extension_path}")

        cwd = pathlib.Path().absolute()
        build_dir = cwd.parent.parent / "build"
        install_dir = extension_path.parent / "ttmlir"

        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DTTMLIR_ENABLE_PYKERNEL=ON",
            # Add the Source Directory (root)
            "-S",
            str(cwd.parent.parent),
        ]

        self.spawn(["cmake", *cmake_args])
        self.spawn(["cmake", "--build", str(build_dir)])

        # Install the PyKernel Component
        self.spawn(["cmake", "--install", str(build_dir), "--component", "PyKernel"])

        # Don't need the pykernel directory
        self.rmdir(install_dir / "pykernel")

        # Remove top-level .py files (pykernel files)
        for py_file in install_dir.glob("*.py"):
            py_file.unlink()

        # Move all ttmlir content to parent directory
        if (install_dir / "ttmlir").exists():
            # List all items in the ttmlir directory
            for item in (install_dir / "ttmlir").iterdir():
                # Get destination path in parent directory
                dest = install_dir / item.name
                # Move the item
                if dest.exists():
                    if dest.is_dir():
                        self.rmdir(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(install_dir))

            # Remove the now empty ttmlir directory
            self.rmdir(install_dir / "ttmlir")

        # Delete the python dir
        self.rmdir(install_dir / "python")


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
    name="pykernel",
    version=version,
    install_requires=[],
    # Include both pykernel and ttmlir as top-level packages
    packages=["pykernel", "ttmlir"],
    # Map the package names to their locations.
    # "." will redundantly include the pykernel files in both of the packages by default
    # We delete all of this during the build_ext step
    package_dir={"pykernel": ".", "ttmlir": "."},
    ext_modules=[ttmlir_c],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
