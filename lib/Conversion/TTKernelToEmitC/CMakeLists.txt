add_mlir_conversion_library(TTMLIRTTKernelToEmitC
  TTKernelToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTKernelToEmitC

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithToEmitC
  MLIRMemRefToEmitC
  MLIRSCFToEmitC
  MLIREmitCDialect
  MLIRTargetCpp
  MLIRTransformUtils
)
