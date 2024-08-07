add_mlir_library(TTMLIRStableHLOToTTIR
  StableHLOToTTIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/StableHLOToTTIR

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIR
)
