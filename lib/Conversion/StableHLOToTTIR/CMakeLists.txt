include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/shardy)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/shardy-build)
include_directories(${TTMLIR_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_mlir_library(TTMLIRStableHLOToTTIR
  StableHLOToTTIRPatterns.cpp
  StableHLOToTTIRPass.cpp
  ArithToStableHLOPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/StableHLOToTTIR

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIR
  StablehloPasses
)
