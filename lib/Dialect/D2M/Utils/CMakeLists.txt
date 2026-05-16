add_mlir_library(MLIRD2MUtils
  CBUtils.cpp
  DMAUtils.cpp
  DstRegisterAnalysis.cpp
  GridSelectionUtils.cpp
  SpatialOpNormalizeUtil.cpp
  Utils.cpp
  AffineMapAnalysis.cpp
  VirtualGrid.cpp
  SynchronizableOpInterfaceUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  LINK_LIBS PUBLIC
  MLIRTTCoreDialect
  MLIRIR
  MLIRAffineDialect
  MLIRAffineUtils
)
