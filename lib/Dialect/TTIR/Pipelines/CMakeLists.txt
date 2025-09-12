set(TTMLIR_PIPELINES_LIBS
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
  TTMLIRTTIRToLinalg
  MLIRTosaToLinalg
  MLIRTosaToTensor
  MLIRTosaToArith
  TTMLIRTransforms
  MLIRSCFToGPU
  MLIRGPUToNVVMTransforms
  MLIRNVVMToLLVM
  TTMLIRTTIRToTTIRDecomposition
)

if(TTMLIR_ENABLE_STABLEHLO)
  list(APPEND TTMLIR_PIPELINES_LIBS
    TTMLIRStableHLOToTTIR
    TTMLIRSTABLEHLOUtils
    StablehloLinalgTransforms
    StablehloPasses
  )
endif()

add_mlir_dialect_library(MLIRTTIRPipelines
  TTIRPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToLLVM

  LINK_LIBS
  PRIVATE
  ${TTMLIR_PIPELINES_LIBS}
)
