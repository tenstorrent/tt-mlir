add_mlir_dialect_library(MLIRTTMetalPipelines
  TTMetalPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  LINK_LIBS PUBLIC
  MLIRTTIRDialect
  MLIRTTMetalDialect
  MLIRTTIRTransforms
  MLIRTTNNAnalysis
  MLIRPass
  MLIRTransforms
)
