include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/shardy)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/shardy-build)
include_directories(${TTMLIR_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_mlir_library(TTMLIRStableHLOToTTIR
  ArithToStableHLOPass.cpp
  EmptyOpTypeConversion.cpp
  ShardingUtils.cpp
  StableHLOToTTIRPatterns.cpp
  StableHLOToTTIRPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/StableHLOToTTIR

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIR
  StablehloPasses
)
