add_mlir_library(MLIRD2MAnalysis
  GenericInterchangeAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  LINK_LIBS PUBLIC
  MLIRTTCoreDialect
  MLIRIR
)
