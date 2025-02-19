add_mlir_conversion_library(TTMLIRTTNNToEmitC
  TTNNToEmitC.cpp
  TTNNToEmitCPass.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTNNToEmitC

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCFToEmitC
  MLIREmitCDialect
  MLIRFuncTransforms
  MLIRTargetCpp
)
