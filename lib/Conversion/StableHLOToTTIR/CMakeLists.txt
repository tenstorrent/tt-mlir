include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)

get_property(STABLEHLO_LIBS GLOBAL PROPERTY STABLEHLO_LIBS)
add_mlir_library(TTMLIRStableHLOToTTIR
  StableHLOToTTIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/StableHLOToTTIR

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  ${STABLEHLO_LIBS}
)
