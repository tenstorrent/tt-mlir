FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    python3-dev \
    python3-venv \
    python3-pip \
    git \
    git-lfs \
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
    cargo \
    doxygen \
    graphviz \
    patchelf \
    libyaml-cpp-dev \
    libboost-all-dev

# Install clang 17
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod u+x llvm.sh && \
    ./llvm.sh 17 && \
    apt install -y libc++-17-dev libc++abi-17-dev && \
    ln -s /usr/bin/clang-17 /usr/bin/clang && \
    ln -s /usr/bin/clang++-17 /usr/bin/clang++

# Install python packages
RUN pip install \
    cmake \
    pytest \
    torch

# Install Googletest
RUN git clone https://github.com/google/googletest.git -b release-1.12.1 && \
    cd googletest && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_GMOCK=OFF && \
    make && \
    make install && \
    cd ../.. && \
    rm -rf googletest

# Create a directory for the build and toolchain
ARG BUILD_DIR=/home/build
RUN mkdir -p $BUILD_DIR && \
    mkdir -p $TTMLIR_TOOLCHAIN_DIR

# Clone the project and update submodules
RUN git clone https://github.com/tenstorrent/tt-mlir.git $BUILD_DIR/tt-mlir && \
    cd $BUILD_DIR/tt-mlir && \
    git submodule update --init --recursive -f

# Build the toolchain
WORKDIR $BUILD_DIR/tt-mlir
RUN cmake -B env/build env && \
    cmake --build env/build

# Build project to test the container
RUN source env/activate && \
    cmake -G Ninja \
    -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DTTMLIR_ENABLE_RUNTIME=ON \
    -DTTMLIR_ENABLE_RUNTIME_TESTS=ON && \
    cmake --build build --config Release && \
    cmake --build build -- check-ttmlir && \
    cmake --build build -- ttrt

# Run clang-tidy
RUN cmake --build build -- clang-tidy || true

# Clean up the build directory
RUN rm -rf $BUILD_DIR/tt-mlir
