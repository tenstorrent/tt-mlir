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


readme = None


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

    def in_ci(self) -> bool:
        return os.environ.get("IN_CIBW_ENV") == "ON"

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
        install_dir = pathlib.Path(self.build_lib)

        # Fix install dir if using cibuildwheel
        if self.in_ci():
            install_dir = cwd / "build" / install_dir.name

        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
        ]

        if not self.in_ci():
            cmake_args.extend(["-S", str(cwd.parent)])

        # Run source env/activate if in ci, otherwise onus is on dev
        if self.in_ci():
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
        else:
            self.spawn(["cmake", *cmake_args])

        self.spawn(["cmake", "--build", str(build_dir), "--", "TTMLIRPythonModules"])

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

# Read README.md file from project root
readme_path = pathlib.Path(__file__).absolute().parent.parent / "README.md"
with open(str(readme_path), "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="ttmlir",
    version=version,
    install_requires=[],
    # Include ttmlir as top-level packages
    packages=["ttmlir"],
    package_dir={"ttmlir": ""},
    ext_modules=[ttmlir_c],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    # Write to readme
    long_description=readme,
    long_description_content_type="text/markdown",
)
