# Runtime libs from tt-mlir
set(TTNN_RUNTIME_LIBS TTRuntime TTRuntimeTTNN TTBinary)

# Dependency libs from tt-metal/ttnn project for ttnn runtime
set(TTNN_LIBS TTMETAL_LIBRARY TTNN_LIBRARY)

option(TTMLIR_ENABLE_STABLEHLO "Enable StableHLO support" OFF)
if (TTMLIR_ENABLE_STABLEHLO)
include(../../third_party/stablehlo_libs.cmake)
else()
set(TTMLIR_STABLEHLO_LIBRARY_NAMES)
endif()

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

if (TTMLIR_ENABLE_STABLEHLO)
list(APPEND TTMLIR_LIBS TTMLIRStableHLOToTTIR)
endif()

# We supply empty.cpp because CMake does not allow creating a library without sources.
add_library(TTMLIR SHARED empty.cpp)

add_dependencies(TTMLIR
    ${TTMLIR_LIBS}
    ${TTNN_RUNTIME_LIBS}
    ${TTNN_LIBS}
    ${TTMLIR_STABLEHLO_LIBRARY_NAMES}
)

target_link_libraries(TTMLIR PRIVATE
    LLVM
    MLIR

    # Forces the inclusion of all symbols in the shared object
    # This is necessary because the JIT will not be able to find the symbols otherwise
    -Wl,--whole-archive
    ${TTMLIR_LIBS}
    ${TTNN_RUNTIME_LIBS}
    ${TTMLIR_STABLEHLO_LIBRARY_NAMES}
    -Wl,--no-whole-archive

    ${TTNN_LIBS}
    ${STABLEHLO_LIBS}
    flatbuffers
)

target_link_directories(TTMLIR PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib)
