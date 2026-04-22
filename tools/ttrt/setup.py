# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import platform
from setuptools import setup

TTMLIR_VERSION_MAJOR = os.getenv("TTMLIR_VERSION_MAJOR", "0")
TTMLIR_VERSION_MINOR = os.getenv("TTMLIR_VERSION_MINOR", "0")
TTMLIR_VERSION_PATCH = os.getenv("TTMLIR_VERSION_PATCH", "0")

__version__ = f"{TTMLIR_VERSION_MAJOR}.{TTMLIR_VERSION_MINOR}.{TTMLIR_VERSION_PATCH}"


def load_requirements(filename):
    requirements = []
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(filepath, encoding="utf-8") as requirements_file:
        for raw_line in requirements_file:
            requirement = raw_line.strip()
            if (
                not requirement
                or requirement.startswith("#")
                or requirement.startswith("--")
            ):
                continue

            # Skip torch requirements - we'll handle them separately based on platform
            if requirement.startswith("torch"):
                continue

            requirements.append(requirement)

    return requirements


src_dir = os.environ.get(
    "SOURCE_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
)
ttmlir_build_dir = os.environ.get(
    "TTMLIR_BINARY_DIR",
    os.path.join(src_dir, "build"),
)

enable_perf = os.environ.get("TT_RUNTIME_ENABLE_PERF_TRACE", "OFF") == "ON"
ttmetalhome = os.environ.get("TT_METAL_RUNTIME_ROOT", "")

install_requires = load_requirements("requirements.txt")
install_requires.append("ttmlir_runtime")

# Add platform-specific torch requirement
if platform.system() == "Linux":
    install_requires.append(
        "torch @ https://download.pytorch.org/whl/cpu/torch-2.9.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"
    )
elif platform.system() == "Darwin":
    install_requires.append("torch==2.9.1")

packages = ["ttrt", "ttrt.common", "ttrt.binary", "ttrt.runtime"]
package_dir = {
    "ttrt": f"{ttmlir_build_dir}/python_packages/ttrt",
    "ttrt.common": f"{ttmlir_build_dir}/python_packages/ttrt/common",
    "ttrt.binary": f"{ttmlir_build_dir}/python_packages/ttrt/binary",
    "ttrt.runtime": f"{ttmlir_build_dir}/python_packages/ttrt/runtime",
}
if enable_perf:
    install_requires += load_requirements("requirements-perf.txt")
    packages += ["tracy"]
    packages += ["tt_metal"]
    package_dir["tracy"] = f"{ttmetalhome}/tools/tracy"
    package_dir["tt_metal"] = f"{ttmetalhome}/tt_metal"

setup(
    name="ttrt",
    version=__version__,
    author="Nicholas Smith",
    author_email="nsmith@tenstorrent.com",
    url="https://github.com/tenstorrent/tt-mlir",
    description="TTRT CLI tool for running compiled TTMLIR flatbuffers",
    long_description="",
    packages=packages,
    package_dir=package_dir,
    install_requires=install_requires,
    entry_points={
        "console_scripts": ["ttrt = ttrt:main"],
    },
    zip_safe=False,
    python_requires=">=3.7",
)
