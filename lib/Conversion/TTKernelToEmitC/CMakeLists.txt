add_mlir_library(TTMLIRTTKernelToEmitC
  TTKernelToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTKernelToEmitC

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithToEmitC
  MLIREmitCDialect
  MLIRTransformUtils
)
