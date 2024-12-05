add_mlir_library(TTMLIRTTIRToTTIRFusion
  TTIRToTTIRFusion.cpp
  TTIRToTTIRFusionPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTIRToTTIR

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
