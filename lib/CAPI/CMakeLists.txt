add_mlir_public_c_api_library(TTMLIRCAPI
  Dialects.cpp
  TTCoreAttrs.cpp
  TTKernelTypes.cpp
  TTNNAttrs.cpp
  TTTypes.cpp
  D2MTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir-c/

  ENABLE_AGGREGATION

  LINK_LIBS PUBLIC
  TTMLIRCompilerStatic
  MLIRIR
  MLIRCAPITransforms
  MLIRSupport
  MLIRTargetCpp
)
