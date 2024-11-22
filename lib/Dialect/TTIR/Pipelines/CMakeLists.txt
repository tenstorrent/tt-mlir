add_mlir_dialect_library(MLIRTTIRPipelines
  TTIRPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion

  LINK_LIBS PUBLIC
  MLIRTTIRDialect
  MLIRPass
  MLIRTransforms
)
