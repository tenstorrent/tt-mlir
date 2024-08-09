get_property(STABLEHLO_LIBS GLOBAL PROPERTY STABLEHLO_LIBS)

add_mlir_library(TTMLIRStableHLOToTTIR
  StableHLOToTTIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/StableHLOToTTIR
  # StableHLO includes
  ${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo
  ${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo-build

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIR
  ${STABLEHLO_LIBS}
)
