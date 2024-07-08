add_mlir_public_c_api_library(TTMLIRCAPI
  Dialects.cpp
  TTKernelTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir-c/

  ENABLE_AGGREGATION

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTTDialect
  MLIRTTIRDialect
  MLIRTTKernelDialect
  MLIRTTIRTransforms
  MLIRTTIRAnalysis
)
