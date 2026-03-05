add_mlir_library(MLIRD2MUtils
  DMAUtils.cpp
  Utils.cpp
  AffineMapAnalysis.cpp
  VirtualGrid.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  DEPENDS
  MLIRD2MOpsIncGen
  D2MGenericRegionOpsIncGen

  LINK_LIBS PUBLIC
  MLIRTTCoreDialect
  MLIRIR
  MLIRAffineDialect
  MLIRAffineUtils
)
