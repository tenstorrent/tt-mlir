add_mlir_dialect_library(TTMLIRTTNNPipelines
  TTNNPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  LINK_LIBS PUBLIC
  TTMLIRTTIRDialect
  TTMLIRTTNNDialect
  TTMLIRTTIRTransforms
  TTMLIRTTNNTransforms
  TTMLIRTTNNAnalysis
  MLIRPass
  MLIRTransforms
)
