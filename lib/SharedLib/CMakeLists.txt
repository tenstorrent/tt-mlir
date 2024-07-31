# We supply empty.cpp because CMake does not allow creating a library without sources.
add_library(TTMLIRSO SHARED empty.cpp)

set(METAL_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-metal/src/tt-metal-build/lib")

add_library(ttnn SHARED IMPORTED)
set_property(TARGET ttnn PROPERTY IMPORTED_LOCATION "${METAL_LIB_DIR}/_ttnn.so")

add_dependencies(TTMLIRSO
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

target_link_libraries(TTMLIRSO PRIVATE
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
    tt_metal
    device
    tt_eager

    # Other
    flatbuffers)

target_link_directories(TTMLIRSO PRIVATE
    ${TTMLIR_TOOLCHAIN_DIR}/lib
    ${METAL_LIB_DIR})
