add_mlir_conversion_library(TTMLIRTTIRToTTMetal
  TTIRToTTMetal.cpp
  TTIRToTTMetalPass.cpp
  AttachMetalLayout.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTIRToTTMetal

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
