add_mlir_dialect_library(MLIRTTIRPipelines
  TTIRPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToLLVM

  LINK_LIBS PUBLIC
  MLIRLLVMTransforms
  MLIRLinalgTransforms
  MLIRArithToLLVM
  MLIRBufferizationPipelines
  MLIRBufferizationToMemRef
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRTensorToLinalg
  MLIRTTIRDialect
  MLIRTTCoreDialect
  MLIRTTTransforms
  MLIRPass
  MLIRTransforms
)
