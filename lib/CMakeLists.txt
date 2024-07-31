add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(SharedLib)

add_mlir_library(TTMLIR STATIC RegisterAll.cpp
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
