#!/bin/bash
set -e  # Stop script if any command fails

# -------------------------------
# Update system and install base tools
# -------------------------------
apt update
apt install -y git cmake ninja-build python3.11-venv wget pip

# -------------------------------
# Install LLVM 17
# -------------------------------
wget https://apt.llvm.org/llvm.sh
chmod +x ./llvm.sh
./llvm.sh 17
clang-17 --version

# Set LLVM 17 as default
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100
clang --version

# -------------------------------
# Install required dependencies
# -------------------------------
apt-get install -y \
    software-properties-common \
    build-essential \
    python3.11-dev \
    python3.11-venv \
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
    libyaml-cpp-dev \
    libboost-all-dev \
    curl \
    jq \
    sudo \
    gh \
    lcov \
    unzip

# -------------------------------
# Install tt-metal dependencies
# -------------------------------
wget https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/{install_dependencies.sh,tt_metal/sfpi-version.sh}
bash install_dependencies.sh --docker

# -------------------------------
# Setup TTMLIR Toolchain directory
# -------------------------------
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
mkdir -p /opt/ttmlir-toolchain
chown -R $USER:$USER /opt/ttmlir-toolchain

# -------------------------------
# Build and activate environment
# -------------------------------
cmake -B env/build env
cmake --build env/build
source env/activate

# -------------------------------
# Build project with TTMLIR options
# -------------------------------
cmake -G Ninja \
    -DTTMLIR_ENABLE_RUNTIME=ON \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DTTMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DTT_RUNTIME_DEBUG=ON \
    -DCMAKE_DEBUG=ON \
    -DTTMLIR_ENABLE_STABLEHLO=ON \
    -DTTMLIR_ENABLE_RUNTIME_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -B build

cmake --build build
cmake --build build -- ttrt

#
python -m pytest test/python/golden/test_ttir_ops.py --path=output --sys-desc=ttrt-artifacts/system_desc.ttsys -vv
