add_mlir_conversion_library(TTMLIRTTNNToEmitPy
  TTNNToEmitPy.cpp
  TTNNToEmitPyPass.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTNNToEmitPy

  DEPENDS
  TTMLIRConversionPassIncGen
  MLIREMITPYDialect

  LINK_LIBS PUBLIC
  MLIREMITPYDialect
  MLIRTTTransforms
  MLIRIR
  MLIRPass
  MLIRFuncTransforms
  MLIRTargetCpp
  MLIREMITPYDialect
)
