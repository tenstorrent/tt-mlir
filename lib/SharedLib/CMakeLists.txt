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

set_target_properties(TTMLIR PROPERTIES INSTALL_RPATH "$ORIGIN")

install(TARGETS TTMLIR DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT SharedLib)

set(TTMLIR_RUNTIME_LIB_DEPS ${TTMLIR_TOOLCHAIN_DIR}/lib/libLLVM.so ${TTMLIR_TOOLCHAIN_DIR}/lib/libMLIR.so)
install(CODE
    "
    MESSAGE(STATUS \"Installing tt-mlir runtime shared lib dependencies (${TTMLIR_RUNTIME_LIB_DEPS}) to ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
    file(
        INSTALL
        ${TTMLIR_RUNTIME_LIB_DEPS}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
    )"
    COMPONENT SharedLib
    EXCLUDE_FROM_ALL
)
