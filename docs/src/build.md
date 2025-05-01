# Build

This page walks you through the steps required to set up tt-mlir. After the project is running, you can do development work, run models, or set up additional projects that build on tt-mlir.

> **NOTE:** If you have a build issue, you can file a bug [here](https://github.com/tenstorrent/tt-mlir/issues).

## Prerequisites

### Hardware Setup

Use this guide to set up your hardware - [Hardware Setup](https://docs.tenstorrent.com/getting-started/README.html).

### System Dependencies

The tt-mlir project has the following system dependencies:
* Ubuntu 22.04 OS
* Clang 14
* Ninja
* CMake 3.20 or higher
* Python 3.10
* python3.10-venv

Install Clang 14, Ninja, CMake, and python3.10-venv:

```bash
sudo apt install git clang cmake ninja-build pip python3.10-venv
```

You should now have the required dependencies installed.

### Clone the tt-mlir Repo

1. Clone the tt-mlir repo:

```bash
git clone https://github.com/tenstorrent/tt-mlir.git
```

2. Navigate into the **tt-mlir** folder.

## Environment Setup

There are two ways to set up the environment, either using a docker image or building the environment manually. The docker image is recommended since it is easier to set up and use.

### Using a Docker Image

Please see [Docker Notes](docker-notes.md#using-the-docker-image) for details on how to set up and use the docker image.

Once you have the docker image running and you are logged into the container, you should be ready to build.

### Setting up the Environment Manually

This section explains how to manually build the environment so you can use tt-mlir. You only need to build this once, it builds llvm, flatbuffers, and a Python virtual environment. You can specify the build type by using `-DLLVM_BUILD_TYPE=*`. The default is `MinSizeRel`, and available options are listed [here](https://llvm.org/docs/CMake.html#frequently-used-cmake-variables).

1. Navigate into the **tt-mlir** folder.

2. Create a directory for the tt-mlir toolchain and set ownership over the directory:

```bash
sudo mkdir -p /opt/ttmlir-toolchain
sudo chown -R ubuntu /opt/ttmlir-toolchain
```

3. Set the path to the toolchain:

```bash
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
```

4. Please ensure that you do not already have an environment (venv) activated before running the following commands:

```bash
source env/activate
cmake -B env/build env
cmake --build env/build
```

> **NOTE:** The last command takes time to run, so give it time to complete.

#### Building the tt-mlir Project

In this step, you build the tt-mlir project:

```bash
source env/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-14 -DCMAKE_CXX_COMPILER=clang++-14
cmake --build build
```
You have now configured tt-mlir.

You can add different flags to your build. Here are some options to consider:

* To enable the ttnn/metal runtime add `-DTTMLIR_ENABLE_RUNTIME=ON`. Clang 17 is the minimum required version when enabling the runtime.
* To enable the ttnn/metal perf runtime add `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`.
* To accelerate the builds with ccache use `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`.
* If Python bindings aren't required for your project, you can accelerate builds further with the command `-DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF`.
* The TTNN build is automatically integrated / handled by the tt-mlir cmake build system.  For debugging and further information regarding the TTNN backend build step, please refer to [TTNN Documentation](https://tenstorrent.github.io/tt-metal/latest/ttnn/ttnn/installing.html).
* If you want to build the runtime for a different architecture, please set `ARCH_NAME` to the desired value:
  * `grayskull` - this product is at End of Life
  * `wormhole_b0`
  * `blackhole`
* The runtime build step depends on the `ARCH_NAME` environment variable, which is set in the `env/activate` script. Please note that the runtime is built only if `TTMLIR_ENABLE_RUNTIME=ON`.
* In addition to `ARCH_NAME`, the runtime build depends on `TT_METAL_HOME` variable, which is also set in `env/activate` script. For more information, please refer to [TT-NN and TT-Metailium installation documentation](https://tenstorrent.github.io/tt-metal/latest/ttnn/ttnn/installing.html#step-4-install-and-start-using-tt-nn-and-tt-metalium).

| OS | Offline Compiler Only | Runtime Enabled Build | Runtime + Perf Enabled Build |
|----|-----------------------|-----------------------| -----------------------------|
| Ubuntu 22.04  | ✅ | ✅ | ✅ |
| Ubuntu 20.04  | ✅ | ❌ | ❌ |
| MacOS         | ✅ | ❌ | ❌ |

#### Test the Build

Use this step to check your build. Do the following:

```bash
source env/activate
cmake --build build -- check-ttmlir
```

## Lint
Set up lint so you can spot errors and stylistic issues before runtime.

In order for this to build correctly, the runtime must be enabled (if it is not enabled, you get an error message asking for tt-metal-download). Make sure your environment is active, and then do the following to build clang-tidy:

```bash
source env/activate
cmake --build build -- clang-tidy
```

### Pre-Commit
Pre-Commit applies a git hook to the local repository such that linting is checked and applied on every `git commit` action. Install from the root of the repository using:

```bash
source env/activate
pre-commit install
```

If you have already committed before installing the pre-commit hooks, you can run on all files to "catch up":

```bash
pre-commit run --all-files
```

For more information visit [pre-commit](https://pre-commit.com/)

## Docs

Build the documentation by doing the following:

1. Make sure you have `mdbook` and `doxygen` installed.

2. Build the docs:

```bash
source env/activate
cmake --build build -- docs
mdbook serve build/docs
```
> **NOTE:** `mdbook serve` will by default create a local server at `http://localhost:3000`.

## Common Build Errors

### `TTMLIRPythonCAPI target requires changing an RPATH`

```
CMake Error at /opt/ttmlir-toolchain/lib/cmake/llvm/AddLLVM.cmake:594 (add_library):
  The install of the TTMLIRPythonCAPI target requires changing an RPATH from
  the build tree, but this is not supported with the Ninja generator unless
  on an ELF-based or XCOFF-based platform.  The
  CMAKE_BUILD_WITH_INSTALL_RPATH variable may be set to avoid this relinking
  step.
```

If you get the above error, it means you tried to build with an old version of cmake or ninja and there is a stale file. To fix this, `rm -rf` your build directory, install a newer version of cmake/ninja, and then rebuild. If you installed ninja via `sudo apt install ninja-build`, it might still be not up-to-date (v1.10.0). You may use ninja in the python virtual environment, or install it via `pip3 install -U ninja`, either way the version `1.11.1.git.kitware.jobserver-1` should work.

### `clang++ is not a full path and was not found in the PATH`

```
CMake Error at CMakeLists.txt:2 (project):
  The CMAKE_CXX_COMPILER:
    clang++
  is not a full path and was not found in the PATH.
  Tell CMake where to find the compiler by setting either the environment
  variable "CXX" or the CMake cache entry CMAKE_CXX_COMPILER to the full path
  to the compiler, or to the compiler name if it is in the PATH.
CMake Error at CMakeLists.txt:2 (project):
  The CMAKE_C_COMPILER:
    clang
  is not a full path and was not found in the PATH.
  Tell CMake where to find the compiler by setting either the environment
  variable "CC" or the CMake cache entry CMAKE_C_COMPILER to the full path to
  the compiler, or to the compiler name if it is in the PATH.
```

If you get the following error, it means you need to install clang which you can do with `sudo apt install clang` on Ubuntu.

### tt-metal Update Failures

```
Failed to unstash changes in: '/path/to/tt-metal/src/tt-metal'
You will have to resolve the conflicts manually
```

This error occurs during CMake's ExternalProject update of tt-metal. The build system tries to apply changes using Git's stash mechanism, but fails due to conflicts. This can happen even if you haven't manually modified any files, as the build process itself may leave behind artifacts or partial changes from previous builds.

To resolve, run the following command:

```bash
rm -rf third_party/tt-metal
```

Then retry your build command. If the error persists, you may need to do the following:

1. Remove the build directory: `rm -rf build`

2. Run CMake commands again.

3. Run the above.
