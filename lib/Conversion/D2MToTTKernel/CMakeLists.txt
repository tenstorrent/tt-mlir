add_mlir_conversion_library(TTMLIRD2MToTTKernel
  D2MToTTKernel.cpp
  D2MToTTKernelPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/D2MToTTKernel

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRD2MDialect
  MLIRD2MAnalysis
  MLIRTTKernelDialect
  MLIRTTMetalDialect
  MLIRTTCoreDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIREmitCDialect
  MLIRMathDialect
  MLIRTransforms
  MLIRPass
  MLIRSupport
)
