# Runtime libs from tt-mlir
set(TTNN_RUNTIME_LIBS TTRuntime TTRuntimeTTNN TTBinary)

# Dependency libs from tt-metal/ttnn project for ttnn runtime
set(TTNN_LIBS TTMETAL_LIBRARY TTNN_LIBRARY)

# Libs from tt-mlir project
set(TTMLIR_LIBS
    TTNNTargetFlatbuffer
    MLIRTTDialect
    MLIRTTIRDialect
    MLIRTTNNDialect
    MLIRTTKernelDialect
    TTMLIRTTIRToTTNN
    MLIRTTMetalDialect
    MLIRTTIRTransforms
    MLIRTTNNTransforms
    MLIRTTIRAnalysis
    MLIRTTNNPipelines
    TTMLIRTTNNToEmitC
)

# We supply empty.cpp because CMake does not allow creating a library without sources.
add_library(TTMLIR SHARED empty.cpp)

add_dependencies(TTMLIR
    ${TTMLIR_LIBS}
    ${TTNN_RUNTIME_LIBS}
    ${TTNN_LIBS}
)

target_link_libraries(TTMLIR PRIVATE
    LLVM
    MLIR

    # Forces the inclusion of all symbols in the shared object
    # This is necessary because the JIT will not be able to find the symbols otherwise
    -Wl,--whole-archive
    ${TTMLIR_LIBS}
    ${TTNN_RUNTIME_LIBS}
    -Wl,--no-whole-archive

    ${TTNN_LIBS}

    flatbuffers
)

target_link_directories(TTMLIR PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib)
