# We supply empty.cpp because CMake does not allow creating a library without sources.
add_library(TTMLIR SHARED empty.cpp)

add_library(ttnn SHARED IMPORTED)
set_property(TARGET ttnn PROPERTY IMPORTED_LOCATION "${TTMETAL_LIBRARY_DIR}/_ttnn.so")

add_dependencies(TTMLIR
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
    TTRuntime
    TTRuntimeTTNN)

target_link_libraries(TTMLIR PRIVATE
    # LLVM
    LLVM
    MLIR

    # Forces the inclusion of all symbols in the shared object
    # This is necessary because the JIT will not be able to find the symbols otherwise
    -Wl,--whole-archive
    # MLIR libraries
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
    TTRuntime
    TTRuntimeTTNN
    # MLIR libraries
    -Wl,--no-whole-archive

    # Metal
    tt_metal
    device
    tt_eager
    ttnn

    # Other
    flatbuffers)

target_link_directories(TTMLIR PRIVATE
    ${TTMLIR_TOOLCHAIN_DIR}/lib
    ${TTMETAL_LIBRARY_DIR})
