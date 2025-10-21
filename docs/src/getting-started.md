# Getting Started

This page walks you through the steps required to set up tt-mlir.

> **NOTE:** If you have a build issue, you can file a bug [here](https://github.com/tenstorrent/tt-mlir/issues).

## Prerequisites

### Hardware Setup

Use this guide to set up your hardware - [Hardware Setup](https://docs.tenstorrent.com/getting-started/README.html).

### System Dependencies

You can use tt-mlir with Ubuntu or Mac OS, however the runtime does not work on Mac OS. tt-mlir project has the following system dependencies:

- Ubuntu 22.04 OS or Mac OS
- Clang >= 14 & <= 18
- Ninja
- CMake 3.24 or higher
- Python 3.11
- python3.11-venv

#### Ubuntu

Install Clang, Ninja, CMake, and python3.11-venv:

```bash
sudo apt install git clang cmake ninja-build pip python3.11-venv
```

You should now have the required dependencies installed.

> **NOTE:** If you intend to build with runtime enabled
> (`-DTTMLIR_ENABLE_RUNTIME=ON`), you also need to install tt-metal
> dependencies which can be found
> [here](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/installing.html#install-system-level-dependencies).

Full developer dependencies as packaged in our docker image:
```bash
{{#include ../../../.github/Dockerfile.base:developer_dependencies}}
```

#### Mac OS

On MacOS we need to install the latest version of [cmake](https://cmake.org/), and [ninja](https://ninja-build.org/) which can be done using Homebrew with (Docs for installing Homebrew: https://brew.sh).

```bash
brew install cmake ninja
```

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

This section explains how to manually build the environment so you can use tt-mlir. You only need to build this once, it builds llvm, flatbuffers, and a Python virtual environment. You can specify the LLVM build type by using `-DLLVM_BUILD_TYPE=*`. The default is `MinSizeRel`, and available options are listed [here](https://llvm.org/docs/CMake.html#frequently-used-cmake-variables).

1. Navigate into the **tt-mlir** folder.

2. The environment gets installed into a toolchain directory, which is by default set to `/opt/ttmlir-toolchain`, but can be overrideen by setting (and persisting in your environment) the environment variable `TTMLIR_TOOLCHAIN_DIR`. You need to manually create the toolchain directory as follows:

```bash
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
sudo mkdir -p "${TTMLIR_TOOLCHAIN_DIR}"
sudo chown -R "${USER}" "${TTMLIR_TOOLCHAIN_DIR}"
```

3. Please ensure that you do not already have an environment (venv) activated before running the following commands:

```bash
{{#include ../../../.github/Dockerfile.ci:environment_build}}
source env/activate
```

> **NOTE:** The last command takes time to run, so give it time to complete.

#### Building the tt-mlir Project

In this step, you build the tt-mlir project:

```bash
source env/activate
cmake -G Ninja -B build
cmake --build build
```

You have now configured tt-mlir.

You can add different flags to your build. Here are some options to consider:

- To enable the ttnn/metal runtime add `-DTTMLIR_ENABLE_RUNTIME=ON`. Clang 17 is the minimum required version when enabling the runtime.
- To enable the ttnn/metal perf runtime add `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`.
- To accelerate the builds with ccache use `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`.
- To workaround OOM issues it can be useful to decrease the number of parallel jobs with `-DCMAKE_BUILD_PARALLEL_LEVEL=4`.
- If Python bindings aren't required for your project, you can accelerate builds further with the command `-DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF`.
- To enable `tt-explorer` add the `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`, `-DTTMLIR_ENABLE_RUNTIME=ON`, and `-DTT_RUNTIME_DEBUG=ON`.
- To enable optimizer pass that uses the op model library, add `-DTTMLIR_ENABLE_OPMODEL=ON`.
- The TTNN build is automatically integrated / handled by the tt-mlir cmake build system. For debugging and further information regarding the TTNN backend build step, please refer to [TTNN Documentation](https://tenstorrent.github.io/tt-metal/latest/ttnn/ttnn/installing.html).
- The runtime build depends on the `TT_METAL_HOME` variable, which is also set in `env/activate` script. For more information, please refer to [TT-NN and TT-Metailium installation documentation](https://tenstorrent.github.io/tt-metal/latest/ttnn/ttnn/installing.html#step-4-install-and-start-using-tt-nn-and-tt-metalium).

| OS           | Offline Compiler Only | Runtime Enabled Build | Runtime + Perf Enabled Build |
| ------------ | --------------------- | --------------------- | ---------------------------- |
| Ubuntu 22.04 | ✅                    | ✅                    | ✅                           |
| Ubuntu 20.04 | ✅                    | ❌                    | ❌                           |
| MacOS        | ✅                    | ❌                    | ❌                           |

#### Test the Build

Use this step to check your build. Do the following:

```bash
source env/activate
cmake --build build -- check-ttmlir
```

## Lint

Set up lint so you can spot errors and stylistic issues before runtime:

```bash
source env/activate
cmake --build build -- clang-tidy
```

> **Note for developers:** You can run:
>
> ```bash
> source env/activate
> cmake --build build -- clang-tidy-ci
> ```
>
> This reproduces the `Lint (clang-tidy)` CI job. It runs `clang-tidy` only on committed files that have been modified relative to the `origin/main` branch.

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

1. Make sure you have `mdbook`, `doxygen`, `sphinx`, and `sphinx-markdown-builder` installed.

2. Build the docs:

```bash
source env/activate
cmake --build build -- docs
mdbook serve build/docs
```

> **NOTE:** `mdbook serve` will by default create a local server at `http://localhost:3000`.

For more information about building the docs please read [the full guide on building the docs](./docs.md).

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

## Common Runtime Errors

### Debugging Python on Mac OS

When debugging python on macOS via lldb you may see an error like:

```
(lldb) r
error: process exited with status -1 (attach failed (Not allowed to attach to process.  Look in the console messages (Console.app), near the debugserver entries, when the attach failed.  The subsystem that denied t
he attach permission will likely have logged an informative message about why it was denied.))
```

For preinstalled macOS binaries you must manually codesign with debug entitlements.

Create file `debuggee-entitlement.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
        <key>com.apple.security.cs.disable-library-validation</key>
        <true/>
        <key>com.apple.security.get-task-allow</key>
        <true/>
</dict>
</plist>
```

Sign the binary:

```bash
sudo codesign -f -s - --entitlements debuggee-entitlement.xml /opt/ttmlir-toolchain/venv/bin/python
```
