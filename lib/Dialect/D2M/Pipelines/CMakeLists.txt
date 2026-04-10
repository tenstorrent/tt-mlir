add_mlir_dialect_library(MLIRD2MPipelines
  D2MPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  LINK_LIBS PUBLIC
  MLIRTTIRDialect
  MLIRTTMetalDialect
  MLIRTTIRTransforms
  MLIRTTCoreDialect
  MLIRTTTransforms
  MLIRPass
  MLIRTransforms
)
