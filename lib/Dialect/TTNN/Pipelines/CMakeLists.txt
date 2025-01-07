add_mlir_dialect_library(MLIRTTNNPipelines
  TTNNPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  LINK_LIBS PUBLIC
  TTMLIRConversions
  MLIRTTIRDialect
  MLIRTTNNDialect
  MLIRTTIRTransforms
  MLIRTTNNTransforms
  MLIRTTNNAnalysis
  MLIRPass
  MLIRTransforms
)
