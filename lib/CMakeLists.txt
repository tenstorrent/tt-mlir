add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)

# Shared library will include runtime code
# so we only build it if runtime is enabled
if (TTMLIR_ENABLE_RUNTIME)
    add_subdirectory(SharedLib)
endif()

add_mlir_library(TTMLIRStatic STATIC RegisterAll.cpp
  LINK_LIBS PUBLIC
    MLIR
    MLIRTTDialect
    MLIRTTIRDialect
    MLIRTTIRTransforms
    TTMLIRConversions
    MLIRTTIRAnalysis
    MLIRTTNNDialect
    MLIRTTNNTransforms
    MLIRTTKernelDialect
    MLIRTTMetalDialect
    MLIRTTMetalTransforms
    MLIRTTNNPipelines
)
