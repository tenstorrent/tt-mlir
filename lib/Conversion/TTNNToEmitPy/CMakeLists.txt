add_mlir_conversion_library(TTMLIRTTNNToEmitPy
  TTNNToEmitPy.cpp
  TTNNToEmitPyPass.cpp
  EmitPyPostProcessing.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTNNToEmitPy

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIREmitPyDialect
)
