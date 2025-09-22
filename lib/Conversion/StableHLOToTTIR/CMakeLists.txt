include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/shardy)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/shardy-build)
include_directories(${TTMLIR_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_mlir_conversion_library(TTMLIRStableHLOToTTIR
  ArithToStableHLOPass.cpp
  EmptyOpTypeConversion.cpp
  StableHLOToTTIRPass.cpp
  StableHLOToTTIRPatterns.cpp
  StableHLOLegalizeCompositePass.cpp
  ShardyToTTIRPatterns.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/StableHLOToTTIR
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Dialect/StableHLO

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  StablehloPasses
  TTMLIRStableHLOUtils
  TTMLIRTTIRUtils
)
