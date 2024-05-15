#!/bin/bash

set -e

# -DMLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS=ON set if debug
# -DLLVM_ENABLE_RUNTIMES="compiler-rt;libc;libcxx;libcxxabi;libunwind"

if [[ ! -z $CONDA_DEFAULT_ENV ]]; then
  echo "Error: cannot build llvm inside of active conda environment ($CONDA_DEFAULT_ENV), please run:"
  echo "  conda deactivate"
  exit 1
fi

if [[ -z $INSTALL_PREFIX ]]; then
  echo "Error: INSTALL_PREFIX is not set"
  exit 1
fi

cmake \
  -S third_party/llvm-project/llvm \
  -B $INSTALL_PREFIX/llvm_build \
  -G Ninja \
  -DCMAKE_PREFIX_PATH=$INSTALL_PREFIX/llvm_build/lib/cmake \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra;lld" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DCMAKE_C_FLAGS="-D_LIBCPP_HAS_NO_LIBRARY_ALIGNED_ALLOCATION"

cmake --build $INSTALL_PREFIX/llvm_build
cmake --install $INSTALL_PREFIX/llvm_build

# Manually install llvm-lit, for some reason doesn't get auto installed
cp $INSTALL_PREFIX/llvm_build/bin/llvm-lit $INSTALL_PREFIX/bin/llvm-lit
