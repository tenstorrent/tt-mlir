add_mlir_conversion_library(TTMLIRTTIRToSMT
  TTIRToSMT.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTIRToSMT

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSMT
  MLIRTransforms
)
