add_mlir_library(MLIRD2MUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  LINK_LIBS PUBLIC
  MLIRIR
)
