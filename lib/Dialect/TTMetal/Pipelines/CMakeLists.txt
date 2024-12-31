add_mlir_dialect_library(TTMLIRTTMetalPipelines
  TTMetalPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  LINK_LIBS PUBLIC
  TTMLIRTTIRDialect
  TTMLIRTTMetalDialect
  TTMLIRTTIRTransforms
  TTMLIRTTNNAnalysis
  MLIRPass
  MLIRTransforms
)
