add_mlir_conversion_library(TTMLIRTTIRToEmitPy
  CaptureMeshShape.cpp
  TTIRCPUToEmitPyPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTIRToEmitPy

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIREmitPyDialect
  MLIRTTCoreDialect
  MLIRTTIRDialect
)
