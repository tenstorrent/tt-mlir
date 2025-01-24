include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)

add_subdirectory(OpModel)
add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(Scheduler)
add_subdirectory(SharedLib)

set(link_libs
MLIRTTDialect
MLIRTTIRDialect
MLIRTTIRTransforms
TTMLIRConversions
MLIRTTNNAnalysis
MLIRTTNNDialect
MLIRTTNNTransforms
MLIRTTKernelDialect
MLIRTTMetalDialect
MLIRTTIRPipelines
MLIRTTNNPipelines
MLIRTTMetalPipelines
)

if (TTMLIR_ENABLE_STABLEHLO)
list(APPEND link_libs StablehloRegister SdyRegister)
endif()

add_mlir_library(TTMLIRStatic STATIC RegisterAll.cpp
  LINK_LIBS PUBLIC
  ${link_libs}
)
