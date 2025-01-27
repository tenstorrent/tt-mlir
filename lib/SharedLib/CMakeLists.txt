# Runtime libs from tt-mlir
set(TTNN_RUNTIME_LIBS TTRuntime TTRuntimeTTNN TTBinary)

# Dependency libs from tt-metal/ttnn project for ttnn runtime
set(TTNN_LIBS TTMETAL_LIBRARY DEVICE_LIBRARY TTNN_LIBRARY)
if (TT_RUNTIME_ENABLE_PERF_TRACE)
  list(APPEND TTNN_LIBS TRACY_LIBRARY)
endif()

# Libs from tt-mlir project
set(TTMLIR_LIBS
    TTNNTargetFlatbuffer
    TTMetalTargetFlatbuffer
    TTKernelTargetCpp
    MLIRTTDialect
    MLIRTTIRDialect
    MLIRTTNNDialect
    MLIRTTKernelDialect
    TTMLIRTTIRToTTIRDecomposition
    TTMLIRTTIRToTTNN
    TTMLIRTTIRToTTMetal
    MLIRTTMetalDialect
    MLIRTTIRTransforms
    MLIRTTNNTransforms
    MLIRTTNNAnalysis
    MLIRTTNNPipelines
    MLIRTTMetalPipelines
    TTMLIRConversions
    TTMLIRTTNNToEmitC
    TTMLIRStatic
    TTMLIRTTNNUtils
)

# We supply empty.cpp because CMake does not allow creating a library without sources.

add_library(TTMLIR SHARED empty.cpp)
add_library(TTMLIRCompiler SHARED empty.cpp)

add_dependencies(TTMLIR
    ${TTMLIR_LIBS}
    ${TTNN_RUNTIME_LIBS}
    ${TTNN_LIBS}
)

add_dependencies(TTMLIRCompiler
    ${TTMLIR_LIBS}
)

target_link_libraries(TTMLIRCompiler PRIVATE
    LLVM
    MLIR

    # Forces the inclusion of all symbols in the shared object
    # This is necessary because the JIT will not be able to find the symbols otherwise
    -Wl,--whole-archive
    ${TTMLIR_LIBS}
    -Wl,--no-whole-archive

    flatbuffers
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
target_link_directories(TTMLIRCompiler PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib)

set_target_properties(TTMLIR PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(TTMLIRCompiler PROPERTIES INSTALL_RPATH "$ORIGIN")

install(TARGETS TTMLIR DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT SharedLib)
install(TARGETS TTMLIRCompiler DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Test)
