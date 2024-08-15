include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)

add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)

# Shared library will include runtime code
# so we only build it if runtime is enabled

set(ttmlir_dependencies "")
set(link_libs 
MLIR;
MLIRTTDialect;
MLIRTTIRDialect;
MLIRTTIRTransforms;
TTMLIRConversions;
MLIRTTIRAnalysis;
MLIRTTNNDialect;
MLIRTTNNTransforms;
MLIRTTKernelDialect;
MLIRTTMetalDialect;
MLIRTTMetalTransforms;
MLIRTTNNPipelines
)

if (TTMLIR_ENABLE_STABLEHLO)
    list(APPEND link_libs TTMLIRStableHLOToTTIR)
endif()

add_mlir_library(TTMLIRStatic STATIC RegisterAll.cpp

  DEPENDS
  ${ttmlir_dependencies}
 
  LINK_LIBS PUBLIC
  ${link_libs}
)

