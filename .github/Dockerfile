# Use the official Ubuntu 20.04 as a parent image
FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential=12.8ubuntu1.1 \
    python3-dev=3.8.2-0ubuntu2 \
    python3.8-venv \
    python3-pip \
    python3-dev \
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
RUN pip install cmake

# Googletest latest version (which isn't in apt for 20.04)
RUN git clone https://github.com/google/googletest.git -b release-1.12.1 && cd googletest && mkdir build && cd build && cmake .. -DBUILD_GMOCK=OFF && make && make install && cd ../.. && rm -rf googletest

# Set the user for RUN commands
ARG USER=builder
RUN useradd -m -s /bin/bash $USER

# Create and change ownership of toolchain dir
USER root
RUN mkdir -p $TTMLIR_TOOLCHAIN_DIR && \
    chown -R $USER:$USER $TTMLIR_TOOLCHAIN_DIR

# Clone the project and update submodules
USER $USER
RUN git clone https://github.com/tenstorrent/tt-mlir.git /home/$USER/tt-mlir && \
    cd /home/$USER/tt-mlir && \
    git submodule update --init --recursive -f

# Build the toolchain
WORKDIR /home/$USER/tt-mlir
RUN cmake -B env/build env && \
    cmake --build env/build

# Build project to test the container
USER $USER
RUN source env/activate && \
    cmake -G Ninja \
    -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DTTMLIR_ENABLE_RUNTIME=ON \
    -DTTMLIR_ENABLE_RUNTIME_TESTS=ON && \
    cmake --build build --config Release

# Run clang-tidy
RUN cmake --build build -- clang-tidy || true

# Run the tests
RUN cmake --build build -- check-ttmlir || true

# Clean up the user directory
RUN rm -rf /home/$USER/tt-mlir

# Define the default workdir
WORKDIR /home/$USER
CMD ["bash"]
