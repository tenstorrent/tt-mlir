#!/bin/bash

source env/activate

cmake \
  -G $BUILDER \
  -B $OUT \
  -DCMAKE_BUILD_TYPE=$CONFIG \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_FLAGS="--system-header-prefix=$ENV"
