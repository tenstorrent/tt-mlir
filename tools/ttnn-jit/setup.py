# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
import re
from setuptools import setup

# Configuration Constants
import sysconfig

# Use Python 3.11 to match the build environment
PYTHON_VERSION = "cpython-311"
DEFAULT_ARCH = "x86_64"


# Package structure definitions
TTNN_SUBPACKAGES = [
    "ttnn",
    "ttnn.operations",
    "ttnn.distributed",
]

TTMLIR_SUBPACKAGES = [
    "dialects",
    "dialects.linalg",
    "dialects.linalg.opdsl",
    "dialects.linalg.passes",
    "extras",
    "_mlir_libs",
]

# Library definitions
CORE_TTNN_LIBS = ["_ttnncpp.so"]
CORE_TTMETAL_LIBS = [
    "libtt_metal.so",
    "libdevice.so",
    "libtt_stl.so",
    "libtracy.so.0.10.0",
]
ESSENTIAL_TTMETAL_DIRS = ["tt_metal", "runtime", "ttnn"]


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


def get_build_configuration():
    """Get build paths and environment configuration"""
    src_dir = os.environ.get(
        "SOURCE_ROOT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
    )

    ttmlir_build_dir = os.environ.get(
        "TTMLIR_BINARY_DIR", os.path.join(src_dir, "build")
    )

    metaldir = f"{src_dir}/third_party/tt-metal/src/tt-metal/build"
    ttmetalhome = os.environ.get("TT_METAL_HOME", "")
    arch = os.environ.get("CMAKE_SYSTEM_PROCESSOR", DEFAULT_ARCH)

    # Set RPATH for runtime linking
    os.environ["LDFLAGS"] = "-Wl,-rpath,'$ORIGIN'"

    return {
        "src_dir": src_dir,
        "ttmlir_build_dir": ttmlir_build_dir,
        "metaldir": metaldir,
        "ttmetalhome": ttmetalhome,
        "arch": arch,
    }


def copy_library_files(src_path, dst_path, libraries):
    """Copy library files from source to destination if they exist"""
    copied_libs = []
    for lib in libraries:
        lib_src = f"{src_path}/{lib}"
        if os.path.exists(lib_src):
            shutil.copy(lib_src, dst_path)
            copied_libs.append(lib)
    return copied_libs


def copy_directories_and_get_files(src_base, dst_base, directories):
    """Copy directories and return list of all files for packaging"""
    all_files = []
    for dirname in directories:
        src_dir_path = f"{src_base}/{dirname}"
        dst_dir_path = f"{dst_base}/{dirname}"

        if os.path.exists(src_dir_path):
            shutil.copytree(src_dir_path, dst_dir_path, dirs_exist_ok=True)

            # Collect all files in the copied directory
            for path, _, filenames in os.walk(dst_dir_path):
                for filename in filenames:
                    all_files.append(os.path.join("..", path, filename))

    return all_files


def setup_runtime_libraries(config):
    """Setup and copy all required runtime libraries"""
    wheel_runtime_dir = f"{config['ttmlir_build_dir']}/python_packages/ttnn_jit/runtime"
    ttmlir_libs_dst_dir = (
        f"{config['ttmlir_build_dir']}/python_packages/ttmlir/_mlir_libs"
    )

    os.makedirs(wheel_runtime_dir, exist_ok=True)

    # Core libraries
    runtime_module = f"_ttmlir_runtime.{PYTHON_VERSION}-{config['arch']}-linux-gnu.so"
    ttnn_jit_module = f"_ttnn_jit.{PYTHON_VERSION}-{config['arch']}-linux-gnu.so"

    dylibs = ["libTTMLIRRuntime.so", runtime_module]

    # Copy core runtime files
    shutil.copy(
        f"{config['ttmlir_build_dir']}/runtime/lib/libTTMLIRRuntime.so",
        wheel_runtime_dir,
    )
    shutil.copy(
        f"{config['ttmlir_build_dir']}/runtime/python/{runtime_module}",
        wheel_runtime_dir,
    )

    # Copy ttnn-jit module if it exists
    ttnn_jit_module_path = (
        f"{config['ttmlir_build_dir']}/tools/ttnn-jit/python/{ttnn_jit_module}"
    )
    if os.path.exists(ttnn_jit_module_path):
        shutil.copy(ttnn_jit_module_path, wheel_runtime_dir)
        dylibs.append(ttnn_jit_module)

    # Copy ALL MPI and system libraries to make the wheel completely standalone
    # MPI libraries from OpenMPI
    mpi_lib_dir = "/opt/openmpi-v5.0.7-ulfm/lib"
    mpi_libs = ["libmpi.so.40", "libopen-pal.so.80", "libpmix.so.2", "libprrte.so.3"]
    copied_mpi_libs = copy_library_files(mpi_lib_dir, wheel_runtime_dir, mpi_libs)
    dylibs.extend(copied_mpi_libs)

    # System libraries that MPI depends on
    system_lib_dir = "/usr/lib/x86_64-linux-gnu"
    system_libs = [
        "libhwloc.so.15",  # Hardware locality library
        "libnsl.so.2",  # Network service library
        "libevent_core-2.1.so.7",  # Event library core
        "libevent_pthreads-2.1.so.7",  # Event library pthreads
        "libudev.so.1",  # Device management
        "libz.so.1",  # Compression library
        # Additional dependencies of libnsl.so.2
        "libtirpc.so.3",  # Transport Independent RPC
        "libgssapi_krb5.so.2",  # GSSAPI Kerberos
        "libkrb5.so.3",  # Kerberos library
        "libk5crypto.so.3",  # Kerberos crypto
        "libcom_err.so.2",  # Common error handling
        "libkrb5support.so.0",  # Kerberos support
        "libkeyutils.so.1",  # Key utilities
        "libresolv.so.2",  # DNS resolver
    ]
    copied_system_libs = copy_library_files(
        system_lib_dir, wheel_runtime_dir, system_libs
    )
    dylibs.extend(copied_system_libs)

    # Copy TTMLIRCompiler library to fix missing symbols
    compiler_lib_path = f"{config['ttmlir_build_dir']}/lib/libTTMLIRCompiler.so"
    if os.path.exists(compiler_lib_path):
        shutil.copy(compiler_lib_path, wheel_runtime_dir)
        dylibs.append("libTTMLIRCompiler.so")

    # Copy runtime libraries
    metal_lib_dir = f"{config['metaldir']}/lib"
    all_runtime_libs = CORE_TTNN_LIBS + CORE_TTMETAL_LIBS

    # Copy to both ttmlir libs and runtime dirs
    copy_library_files(metal_lib_dir, ttmlir_libs_dst_dir, all_runtime_libs)
    copy_library_files(metal_lib_dir, wheel_runtime_dir, all_runtime_libs)

    # Copy essential TT-Metal directories
    metallibs = copy_directories_and_get_files(
        config["ttmetalhome"], wheel_runtime_dir, ESSENTIAL_TTMETAL_DIRS
    )

    return dylibs + all_runtime_libs + metallibs


def generate_package_configuration(config):
    """Generate packages and package_dir configurations"""
    rel_build_dir = os.path.relpath(
        config["ttmlir_build_dir"], os.path.dirname(os.path.abspath(__file__))
    )

    packages = ["ttnn_jit", "ttnn_jit._src", "ttnn_jit.runtime"]
    package_dir = {
        "ttnn_jit": f"{rel_build_dir}/python_packages/ttnn_jit",
        "ttnn_jit._src": f"{rel_build_dir}/python_packages/ttnn_jit/_src",
        "ttnn_jit.runtime": f"{rel_build_dir}/python_packages/ttnn_jit/runtime",
    }

    # Add ttmlir packages
    packages.extend(
        [
            f"ttmlir.{sub}" if sub != "_mlir_libs" else "ttmlir._mlir_libs"
            for sub in TTMLIR_SUBPACKAGES
        ]
    )
    packages.append("ttmlir")

    for sub in TTMLIR_SUBPACKAGES:
        # Convert dot notation to filesystem path (e.g., "dialects.linalg" -> "dialects/linalg")
        sub_path = sub.replace(".", "/")
        package_dir[
            f"ttmlir.{sub}"
        ] = f"{rel_build_dir}/python_packages/ttmlir/{sub_path}"
    package_dir["ttmlir"] = f"{rel_build_dir}/python_packages/ttmlir"

    # Add ttnn packages
    runtime_ttnn_base = f"{rel_build_dir}/python_packages/ttnn_jit/runtime"
    for sub in TTNN_SUBPACKAGES:
        full_package = f"ttnn.{sub}"
        packages.append(full_package)
        # Convert dot notation to filesystem path (e.g., "ttnn.operations" -> "ttnn/operations")
        sub_path = sub.replace(".", "/")
        package_dir[full_package] = f"{runtime_ttnn_base}/ttnn/{sub_path}"

    return packages, package_dir


def get_dynamic_version():
    """Get version from environment variables and always include tt-metal version"""
    version_major = os.getenv("TTMLIR_VERSION_MAJOR", "0")
    version_minor = os.getenv("TTMLIR_VERSION_MINOR", "0")
    version_patch = os.getenv("TTMLIR_VERSION_PATCH", "0")
    base_version = f"{version_major}.{version_minor}.{version_patch}"

    # Always try to get TT Metal version - this is required
    full_hash = extract_tt_metal_version()

    if not full_hash:
        raise RuntimeError(
            "Could not extract TT_METAL_VERSION from third_party/CMakeLists.txt. "
            "This is required for building ttnn-jit package."
        )

    short_hash = full_hash[:8]  # First 8 chars
    # Always include TT Metal version as build metadata
    if base_version == "0.0.0":
        # Use default base version when no env vars set
        return f"0.1.0+ttnn.{short_hash}"
    else:
        # Include TT Metal version with custom base version
        return f"{base_version}+ttnn.{short_hash}"


def get_dynamic_dependencies():
    """Get dependencies needed for TTNN JIT functionality"""
    return [
        "nanobind",  # Python binding framework
        "torch==2.7.0",  # PyTorch for tensor operations - specific version required
        "numpy",  # Required for tensor operations
    ]


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


def generate_package_data(all_runtime_libs):
    """Generate package_data configuration with file patterns"""
    package_data = {
        "ttnn_jit.runtime": all_runtime_libs,
        "ttmlir": ["*.py"],
        "ttmlir._mlir_libs": ["*.so", "*.py", "*.pyi"] + all_runtime_libs,
        "ttmlir.extras": ["*.py"],
    }

    # Add ttmlir dialect package data
    for sub in [
        sub for sub in TTMLIR_SUBPACKAGES if sub != "_mlir_libs" and sub != "extras"
    ]:
        package_data[f"ttmlir.{sub}"] = ["*.py"]

    # Add ttnn package data with essential file patterns only
    base_ttnn_patterns = ["*.py", "*.so"]

    package_data["ttnn"] = base_ttnn_patterns
    for sub in TTNN_SUBPACKAGES:
        package_data[f"ttnn.{sub}"] = base_ttnn_patterns

    return package_data


def main():
    """Main setup function"""
    # Get build configuration
    config = get_build_configuration()

    # Setup runtime libraries and get list of all libraries
    all_runtime_libs = setup_runtime_libraries(config)

    # Generate package configuration
    packages, package_dir = generate_package_configuration(config)

    # Generate package data
    package_data = generate_package_data(all_runtime_libs)

    # Call setup with clean configuration
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
        package_data=package_data,
        zip_safe=False,
        long_description=get_readme(),
        long_description_content_type="text/markdown",
        python_requires=">=3.11",
    )


if __name__ == "__main__":
    main()
