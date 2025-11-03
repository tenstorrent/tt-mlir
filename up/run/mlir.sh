#!/bin/bash

# Repo setup
# Use docker image ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-22-04:latest
# source env/activate

{ # Script setup
    set -o pipefail # to make sure exit status is propagated even when using tee to capture output logs
    mkdir -p logs/build
    source "$(dirname "$0")/common.sh" # Source common functions
    echo -e "\n\nbisecting commit: $(git log -1 --oneline)\n" >> logs/bisect_log.log
    # echo -e "Currently bisecting commit:\n$(git log -1 --oneline)\n\nWindow remaining:\n$(git bisect visualize --oneline)" > logs/bisect_status.log
}

{ # build steps
    # rm -rf build 
    # rm -rf third_party/tt-metal
    # rm -rf tools/explorer/model-explorer
    ## speedy
    # cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 -DTTMLIR_ENABLE_RUNTIME=ON -DTTMLIR_ENABLE_RUNTIME_TESTS=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DTTMLIR_ENABLE_STABLEHLO=ON -DTTMLIR_ENABLE_OPMODEL=ON |& tee logs/build/cmake_cfg.log
    ## tracy
    cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 -DTTMLIR_ENABLE_RUNTIME=ON -DTT_RUNTIME_ENABLE_PERF_TRACE=ON -DTTMLIR_ENABLE_RUNTIME_TESTS=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DTT_RUNTIME_DEBUG=ON -DTTMLIR_ENABLE_STABLEHLO=ON -DTTMLIR_ENABLE_OPMODEL=ON |& tee logs/build/cmake_cfg.log
    log_result $? "cmake-config" $UP_SKIP

    cmake --build build |& tee logs/build/build.log; log_result $? "cmake-build" $UP_SKIP

    rm -rf ttrt-artifacts
    ttrt query --save-artifacts |& tee logs/artifacts.log; log_result $? "ttrt-query" $UP_SKIP
    export SYSTEM_DESC_PATH=`pwd`/ttrt-artifacts/system_desc.ttsys
}


# regtest
cmake --build build -- check-ttmlir |& tee logs/check_ttmlir.log
log_result $? "check-ttmlir" $UP_BAD

{ # explorer commands
    # explorer build
    cmake --build build -- explorer |& tee logs/build/explorer.log
    log_result $? "explorer-build" $UP_SKIP

    # explorer run
    # use this export each time after vscode reloaded if tmux is used
    # export VSCODE_IPC_HOOK_CLI=$(ls -t /tmp/vscode-ipc-*.sock 2>/dev/null | head -1)
    tt-explorer |& tee logs/explorer.log
    log_result $? "explorer-run"
}

Exit early
echo -e "Build passed\n" >> logs/bisect_log.log
exit 0

# { # alchemist commands
#     # alchemist build
#     cmake --build build -- tt-alchemist |& tee logs/build/alchemist.log
#     log_result $? "alchemist-build" $UP_SKIP

#     # alchemist test
#     tt-alchemist model-to-cpp tools/tt-alchemist/test/models/mnist.mlir |& tee logs/alchemist_test_cpp.log
#     log_result $? "alchemist-test-cpp"

#     tt-alchemist model-to-python tools/tt-alchemist/test/models/mnist.mlir |& tee logs/alchemist_test_python.log
#     log_result $? "alchemist-test-python"

#     # generate cpp and run
#     rm -rf /tmp/test-generate-cpp-mnist && tt-alchemist generate-cpp tools/tt-alchemist/test/models/mnist.mlir --output /tmp/test-generate-cpp-mnist --standalone && cd /tmp/test-generate-cpp-mnist && ./run && cd - |& tee logs/alchemist_generate_cpp.log
#     log_result $? "alchemist-generate-cpp"

#     # generate python and run
#     rm -rf /tmp/test-generate-python && tt-alchemist generate-python tools/tt-alchemist/test/models/mnist.mlir --output /tmp/test-generate-python --standalone && cd /tmp/test-generate-python && ./run && cd - |& tee logs/alchemist_generate_python.log
#     log_result $? "alchemist-generate-python"
# }

# { # ttrt specific tests
#     # example ttrt
#     ./build/bin/ttmlir-opt -mlir-print-ir-after-all --ttir-to-ttnn-backend-pipeline="system-desc-path=/localdev/achoudhury/up/tt-mlir/ttrt-artifacts/system_desc.ttsys" test/ttmlir/Silicon/TTNN/n150/data_movement/reshape/reshape.mlir | ./build/bin/ttmlir-translate --ttnn-to-flatbuffer -o out.ttnn && ttrt run out.ttnn |& tee logs/jackson_ttrt.log
#     log_result $? "example-ttrt"

#     # specific ttrt test examples
#     # concat op test
#     llvm-lit -sv build/test/ttmlir/Silicon/StableHLO/n150/Binary/concat_op.mlir |& tee logs/silicon_lit_concat.log
#     log_result $? "silicon-lit-concat"

#     ttrt run build/test/ttmlir/Silicon/StableHLO/n150/Binary/concat_op.mlir |& tee logs/silicon_tc_concat.log
#     log_result $? "silicon-tc-concat"

#     ttrt perf build/test/ttmlir/Silicon/StableHLO/n150/Binary/concat_op.mlir |& tee logs/silicon_perf_concat.log
#     log_result $? "silicon-perf-concat"

#     # all reduce test
#     rm -rf ttrt-artifacts && ttrt query --save-artifacts |& tee logs/artifacts.log && llvm-lit -sv build/test/ttmlir/Silicon/TTNN/n300/perf/Output/all_reduce.mlir |& tee logs/silicon_lit_reduce.log && ttrt run build/test/ttmlir/Silicon/TTNN/n300/perf/Output/all_reduce.mlir |& tee logs/silicon_tc_reduce.log && ttrt perf build/test/ttmlir/Silicon/TTNN/n300/perf/Output/all_reduce.mlir |& tee logs/silicon_perf_reduce.log
#     log_result $? "all-reduce-test"

#     # simple add test
#     rm -rf ttrt-artifacts && ttrt query --save-artifacts |& tee logs/artifacts.log && llvm-lit -sv build/test/ttmlir/Silicon/TTMetal/n150/Output/simple_add.mlir |& tee logs/silicon_lit_add.log && ttrt run build/test/ttmlir/Silicon/TTMetal/n150/Output/simple_add.mlir |& tee logs/silicon_tc_add.log && ttrt perf build/test/ttmlir/Silicon/TTMetal/n150/Output/simple_add.mlir |& tee logs/silicon_perf_add.log
#     log_result $? "simple-add-test"
# }

# { # emitc commands
#     # emitc full test suite
#     rm -rf ttrt-artifacts/ && ttrt query --save-artifacts |& tee logs/artifacts.log && llvm-lit -sv test/ttmlir/EmitC/TTNN |& tee logs/emitc_1.log && tools/ttnn-standalone/ci_compile_dylib.py |& tee logs/emitc_2.log && TTRT_LOGGER_LEVEL=DEBUG ttrt run --emitc build/test/ttmlir/EmitC/TTNN |& tee logs/emitc_3.log
#     log_result $? "emitc-full-suite"

#     # emitc pooling tests
#     rm -rf ttrt-artifacts/ && ttrt query --save-artifacts |& tee logs/artifacts.log && llvm-lit -sv test/ttmlir/EmitC/TTNN/pooling |& tee logs/emitc_pooling_1.log && tools/ttnn-standalone/ci_compile_dylib.py |& tee logs/emitc_pooling_2.log && TTRT_LOGGER_LEVEL=DEBUG ttrt run --emitc build/test/ttmlir/EmitC/TTNN/pooling |& tee logs/emitc_pooling_3.log
#     log_result $? "emitc-pooling"

#     # emitc conv tests
#     rm -rf ttrt-artifacts/ && ttrt query --save-artifacts |& tee logs/artifacts.log && llvm-lit -sv test/ttmlir/EmitC/TTNN/conv |& tee logs/emitc_conv_1.log && tools/ttnn-standalone/ci_compile_dylib.py |& tee logs/emitc_conv_2.log && TTRT_LOGGER_LEVEL=DEBUG ttrt run --emitc build/test/ttmlir/EmitC/TTNN/conv |& tee logs/emitc_conv_3.log
#     log_result $? "emitc-conv"

#     # emitc convert ttir to c++ and run
#     ttmlir-opt --ttir-to-emitc-pipeline test/ttmlir/Silicon/TTNN/n150/optimizer/shard_transpose.mlir | ttmlir-translate --mlir-to-cpp > tools/ttnn-standalone/ttnn-standalone.cpp && cd tools/ttnn-standalone && ./run && cd - |& tee logs/emitc_convert_run.log
#     log_result $? "emitc-convert-run"
# }

# { # builder tests
#     # builder tests
#     pytest test/python/golden -m "not run_error and not fails_golden" -v |& tee logs/builder.log
#     log_result $? "builder-golden-tests"

#     ttrt run ttnn/ |& tee logs/builder_run_ttnn.log
#     log_result $? "builder-run-ttnn"

#     ttrt run ttmetal/ |& tee logs/builder_run_ttmetal.log
#     log_result $? "builder-run-ttmetal"

#     # clean and run after changing builder pytest
#     # rm -rf ttnn ttmetal && pytest test/python/golden -m "not run_error and not fails_golden" -v |& tee logs/builder.log && ttrt run ttnn/ |& tee logs/builder_run_ttnn.log && ttrt run ttmetal/ |& tee logs/builder_run_ttmetal.log
# }

# { # functional tests
#     # functional tests n150
#     rm -rf ttrt-artifacts/ && ttrt query --save-artifacts |& tee logs/artifacts.log && llvm-lit -sv test/ttmlir/Silicon/TTNN/n150 |& tee logs/func_1.log && ttrt run --non-zero build/test/ttmlir/Silicon/TTNN/n150 |& tee logs/func_2.log
#     log_result $? "functional-tests-n150"

#     # functional tests optimizer
#     rm -rf ttrt-artifacts/ && ttrt query --save-artifacts |& tee logs/artifacts.log && llvm-lit -sv test/ttmlir/Silicon/TTNN/n150/optimizer |& tee logs/opmodel_1.log && ttrt run --non-zero build/test/ttmlir/Silicon/TTNN/n150/optimizer |& tee logs/opmodel_2.log
#     log_result $? "functional-tests-optimizer"
# }

# { # debug commands
#     # Debug flags
#     # export LLVM_DEBUG=1
#     # -DLLVM_ENABLE_ASSERTIONS=ON

#     # Debug an mlir conversion test (or any pass in tt-mlir)
#     # llvm-lit # to run an mlir test that has `RUN` defined with tt-mlir opt
#     # -a -v # add flags to llvm-lit to see verbose output
#     # --mlir-print-ir-after-all # add flag to tt-mlir opt to see the IR after each pass
#     # --debug # add flag to tt-mlir opt to see each conversion pattern match
# }

echo -e "Test passed\n" >> logs/bisect_log.log
exit 0
