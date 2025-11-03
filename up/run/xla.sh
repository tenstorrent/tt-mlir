#!/bin/bash

{ # Repo setup
    # Use docker image ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-22-04:latest
    # sudo apt update && sudo apt install -y libprotobuf-dev protobuf-compiler python3.11 python3.11-venv python3.11-dev
    # export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
    # source venv/activate
    # git submodule update --init --recursive
}

{ # Script setup
    set -o pipefail # to make sure exit status is propagated even when using tee to capture output logs
    mkdir -p logs/build
    source "$(dirname "$0")/common.sh" # Source common functions
    echo -e "\n\nbisecting commit: $(git log -1 --oneline)\n" >> logs/bisect_log.log
    # echo -e "Currently bisecting commit:\n$(git log -1 --oneline)\n\nWindow remaining:\n$(git bisect visualize --oneline)" > logs/bisect_status.log
}

# build 
rm -rf build/ install/ third_party/tt-mlir/

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release |& tee logs/build/cmake.log; log_result $? "cmake-config" $UP_SKIP
cmake --build build |& tee logs/build/build.log; log_result $? "cmake-build" $UP_SKIP

pytest -svv tests/jax/single_chip/ops/test_convert.py |& tee logs/convert.log; log_result $? "convert-test"

echo -e "Test passed\n" >> logs/bisect_log.log
exit 0
