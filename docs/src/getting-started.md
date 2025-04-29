# Getting Started

This page walks you through the steps required to set up tt-mlir for development and running models. When you complete the information here you will have:

* Configured your hardware - [Hardware Setup](#hardware-setup)
* Installed the required system dependencies - [System Dependencies](#system-dependencies)
* Clone the tt-mlir repo - [Clone the tt-mlir Repo](#clone-the-tt-mlir-repo)
* Created an environment using one of two methods - [Environment Setup](#environment-setup)
  * Docker (*recommended*) - [Using a Docker Image](#using-a-docker-image)
  * Manual - [Setting up the Environment Manually](#setting-up-the-environment-manually)
* Built the runtime environment - [Building the tt-mlir Project](#building-the-tt-mlir-project)
* Added pre-commit
* Built the docs (*optional*)

After the project is running, you can do development work, run models, or set up additional projects that build on tt-mlir. Documentation links for the various options are presented at the end of this document.

## Prerequisites

### Hardware Setup

Use this guide to set up your hardware - [Hardware Setup](https://docs.tenstorrent.com/getting-started/README.html).

### System Dependencies

The tt-mlir project has the following system dependencies:
* Ubuntu 22.04 OS
* Clang 17
* Ninja
* CMake 3.20 or higher
* Python 3.10

> [!NOTE] The installation instructions use Ubuntu 22.04.

1. On Ubuntu 22.04, start by refreshing the list of available software packages and their versions from the system's package repositories:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

When you receive a screen telling you there is a pending kernel upgrade, press **return** for each screen.

2. Next, install Clang 17. In this step you retrieve what you need for installation, install, and then set the paths for clang and clang++:

```bash
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh 17
sudo apt install -y libc++-17-dev libc++abi-17-dev
sudo ln -s /usr/bin/clang-17 /usr/bin/clang
sudo ln -s /usr/bin/clang++-17 /usr/bin/clang++
```

When you receive a screen telling you there is a pending kernel upgrade, press **return** for each screen.

3. Install or update the version of CMake to 3.20, which is the minimum required for this project:

> [!NOTE] If you do not have CMake installed on your system, you do not need to run the command ```sudo apt remove cmake -y```.

> [!NOTE] If you do not have pip3 installed, you can install it with ```sudo apt install python3-pip```.

```bash
sudo apt remove cmake -y
pip3 install cmake --upgrade
hash -r
```

4. You may get a message that says CMake is not added to PATH. If so, to correct it, run the following command:

```bash
export PATH="$PATH:/home/ubuntu/.local/bin"
```

Next, you need to save and reload, or you can source your configuration file:

```bash
source ~/.bashrc  # Or .zshrc
```

5. Run the following command to check that the CMake version is 3.2.0 or later:

```bash
cmake --version
```

6. Install Ninja with the following command:

```bash
sudo apt install ninja-build
```

You should now have the required dependencies installed, and can move on to setting up your environment.

## Clone the tt-mlir Repo

1. Clone the tt-mlir repo using your preferred method. In this walkthrough, the following command is used:

```bash
git clone https://github.com/tenstorrent/tt-mlir.git
```

2. Navigate into the **tt-mlir** folder.

## Environment Setup

There are two ways to set up the environment, either using a docker image or building the environment manually. The docker image is recommended since it is easier to set up and use.

### Using a Docker Image

Please see [Docker Notes](docker-notes.md#using-the-docker-image) for more information on how to use the docker image.

Once you have the docker image running and you are logged into the container, you should be ready to build.

### Setting up the Environment Manually

This section explains how to manually build the environment so you can use tt-mlir. Make sure you are in the **tt-mlir** folder. Do the following:

1. Create a directory for the tt-mlir toolchain and set ownership over the directory:

```bash
sudo mkdir -p /opt/ttmlir-toolchain
sudo chown -R ubuntu /opt/ttmlir-toolchain
```

2. Set the path to the toolchain:

```bash
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
```

3. To successfully create a virtual environment, you need python3.10-venv for this project. It is recommended to use the **system installation of python3** for the virtual environment. Install it if you do not already have it with the following command:

```bash
sudo apt install python3-venv
```

When you receive a screen telling you there is a pending kernel upgrade, press **return** for each screen.

4. You only need to build this once, it builds llvm, flatbuffers, and a Python virtual environment. You can specify the build type by using `-DLLVM_BUILD_TYPE=*`. The default is `MinSizeRel`, and available options are listed [here](https://llvm.org/docs/CMake.html#frequently-used-cmake-variables).

Please ensure that you do not already have an environment (venv) activated before running the following commands:

```bash
source env/activate
cmake -B env/build env  *** made it to here *** broken
cmake --build env/build
```

> [!NOTE] When running ```source env/activate``` this time, it will not show that the environment is
> on.

>[!NOTE] The last command takes time to run, so give it time to complete.

#### Building the tt-mlir Project

In this step, you build the tt-mlir project.

```bash
source env/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17
cmake --build build
```
You have now configured the basics for working with tt-mlir. For best results, please see the next two sections:
* Installing Additional Dependencies
* Flags You May Want to Add When You Build the tt-mlir Project

#### Installing Additional Dependencies

Deactivate your environment if it is running. The command in this section adds any packages that were missed, and includes doxygen if you want to build them for reference. To add the additional packages, run the following command:

```bash
sudo apt-get install -y \
    software-properties-common \
    build-essential \
    python3-dev \
    python3-venv \
    python3-pip \
    git \
    libhwloc-dev \
    pandoc \
    libtbb-dev \
    libcapstone-dev \
    pkg-config \
    linux-tools-generic \
    ninja-build \
    wget \
    libgtest-dev \
    cmake \
    ccache \
    doxygen \
    graphviz \
    patchelf \
    libyaml-cpp-dev \
    libboost-all-dev \
    curl \
    jq \
    sudo \
    gh \
    lcov
```

#### Flags You May Want to Add When You Build the tt-mlir Project

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

> [!NOTE] If your environment is already active, you do not need to run ```source env/activate``` again.

## Lint
Set up lint so you can spot errors and stylistic issues before runtime.

> [!NOTE] If your environment is already active, you do not need to run ```source env/activate``` again.

In order for this to build correctly, the runtime must be enabled (if it is not enabled, you get an error message asking for tt-metal-download). Make sure your environment is active, and then do the following:

1. Run the command to enable runtime:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 -DTTMLIR_ENABLE_RUNTIME=ON
```

2. Build clang-tidy:

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

Docs built by doing the following:

1. Make sure you have `mdbook` installed:

```bash
sudo snap install mdbook
```

2. Make sure Python can be found on the right path:

```bash
echo $TTMLIR_TOOLCHAIN_DIR
```

You should get a response that the path is ```/opt/ttmlir-toolchain```. If not, then run:

```bash
ln -s /opt/ttmlir-toolchain/venv/bin/python3 /opt/ttmlir-toolchain/venv/bin/python
```

3. Build the docs:

```bash
source env/activate
cmake --build build -- docs
mdbook serve build/docs
```
> [!NOTE] `mdbook serve` will by default create a local server at `http://localhost:3000`.

## Common Build Errors

### tt-metal Update Failures

```
Failed to unstash changes in: '/path/to/tt-metal/src/tt-metal'
You will have to resolve the conflicts manually
```

This error occurs during CMake's ExternalProject update of tt-metal. The build system tries to apply changes using Git's stash mechanism, but fails due to conflicts. This can happen even if you haven't manually modified any files, as the build process itself may leave behind artifacts or partial changes from previous builds.

To resolve, clean up the tt-metal directory:

```bash
cd third_party/tt-metal/src/tt-metal
git reset --hard HEAD  # Reset any tracked file modifications
git clean -fd         # Remove all untracked files and directories
```

Then retry your build command. If the error persists, you may also need to:
1. Remove the build directory: `rm -rf build`
2. Run CMake commands again.
3. Run the above.
