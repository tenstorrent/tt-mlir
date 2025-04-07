include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo-build)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/shardy)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/shardy-build)
include_directories(${TTMLIR_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_mlir_conversion_library(TTMLIRStableHLOToTTIR
  ArithToStableHLOPass.cpp
  EmptyOpTypeConversion.cpp
  ShardingUtils.cpp
  StableHLOToTTIRPass.cpp
  StableHLOToTTIRPatterns.cpp
  ShardyToTTIRPatterns.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/StableHLOToTTIR

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  StablehloPasses
  StablehloReferenceApi
  InterpreterPasses
  StablehloReferenceErrors
  StablehloReferenceConfiguration
)

# Force whole-archive linking for the last four static libs
target_link_options(TTMLIRStableHLOToTTIR PUBLIC
  -Wl,--whole-archive
  $<TARGET_FILE:StablehloReferenceApi>
  $<TARGET_FILE:InterpreterPasses>
  $<TARGET_FILE:StablehloReferenceErrors>
  $<TARGET_FILE:StablehloReferenceConfiguration>
  -Wl,--no-whole-archive
)