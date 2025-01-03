add_mlir_dialect_library(TTMLIRTTIRPipelines
  TTIRPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToLLVM

  LINK_LIBS PUBLIC
  MLIRLinalgTransforms
  MLIRArithToLLVM
  MLIRBufferizationPipelines
  MLIRBufferizationToMemRef
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRTensorToLinalg
  TTMLIRTTIRDialect
  MLIRPass
  MLIRTransforms
)
