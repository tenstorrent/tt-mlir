add_mlir_conversion_library(TTMLIRTTNNToTTIR
  TTNNToTTIR.cpp
  TTNNToTTIRPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTNNToTTIR

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  # TTMLIRTTNNUtils
)
