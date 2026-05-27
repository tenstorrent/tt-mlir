add_mlir_conversion_library(TTMLIRConstructTTIRLEC
  ConstructTTIRLEC.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/ConstructTTIRLEC

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSMT
  MLIRTransforms
  MLIRFuncDialect
)
