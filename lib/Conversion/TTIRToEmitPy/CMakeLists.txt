add_mlir_conversion_library(TTMLIRTTIRToEmitPy
  TTIRCPUToEmitPyPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTIRToEmitPy

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIREmitPyDialect
  MLIRTTIRDialect
)
