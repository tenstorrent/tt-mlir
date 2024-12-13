add_mlir_dialect_library(MLIRTTIRPipelines
  TTIRPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToLLVM

  LINK_LIBS PUBLIC
  MLIRLLVMTransforms
  MLIRTTIRDialect
  MLIRPass
  MLIRTransforms
)
