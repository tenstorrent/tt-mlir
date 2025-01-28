add_mlir_library(TTMLIRTTIRToTTIRDecomposition
  TTIRToTTIRDecomposition.cpp
  TTIRToTTIRDecompositionPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTIRToTTIR

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIR
)
