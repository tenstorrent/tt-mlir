#!/bin/bash

# Environment
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CC=clang-17
export CXX=clang++-17
export CMAKE_BUILD_TYPE=Debug
export TTMLIR_ENABLE_RUNTIME=ON

source env/activate

set -e

# Hook[cmake]
#[ ! -f build/build.ninja ] && cmake -G Ninja -B build \
#  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
#  -DCMAKE_C_COMPILER=${CC} \
#  -DCMAKE_CXX_COMPILER=${CXX} \
#  -DTTMLIR_ENABLE_RUNTIME=${TTMLIR_ENABLE_RUNTIME}


set -o pipefail
set -x

# Script[standalone]
#cmake --build build
#ttmlir-opt --ttir-to-emitc-pipeline test/ttmlir/EmitC/TTNN/sanity_add.mlir | ttmlir-translate --mlir-to-cpp > tools/ttnn-standalone/ttnn-standalone.cpp
cd tools/ttnn-standalone
./run 
