add_mlir_conversion_library(TTMLIRD2MToTTMetal
  D2MToTTMetal.cpp
  D2MToTTMetalPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/D2MToTTMetal

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
