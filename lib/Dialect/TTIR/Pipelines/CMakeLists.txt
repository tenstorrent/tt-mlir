set(link_libs
MLIR
MLIRTTIRDialect
)

if (TTMLIR_ENABLE_STABLEHLO)
list(APPEND link_libs StablehloRegister StablehloPasses SdyRegister)
endif()

add_mlir_dialect_library(MLIRTTIRPipelines
  TTIRPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToLLVM

  LINK_LIBS PUBLIC
  ${link_libs}
)
