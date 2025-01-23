add_mlir_public_c_api_library(TTMLIRCAPI
  Dialects.cpp
  TTKernelTypes.cpp
  TTAttrs.cpp
  TTTypes.cpp
  TTNNAttrs.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir-c/

  ENABLE_AGGREGATION

  LINK_LIBS PUBLIC
  MLIR
  TTMLIRStatic
)
