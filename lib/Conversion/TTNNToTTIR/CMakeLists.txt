include_directories(${TTMLIR_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_mlir_conversion_library(TTMLIRTTNNToTTIR
  TTNNToTTIRPass.cpp
  TTNNToTTIRPatterns.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTNNToTTIR
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Dialect/TTNN

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TTMLIRTTNNUtils
  TTMLIRTTIRUtils
)
