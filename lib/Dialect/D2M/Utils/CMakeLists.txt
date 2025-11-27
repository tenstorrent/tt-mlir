add_mlir_library(MLIRD2MUtils
  DstCopyLoopBuilder.cpp
  LoopSemantics.cpp
  SelectiveLoopBuilder.cpp
  TileMatmulUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgTransforms
)
