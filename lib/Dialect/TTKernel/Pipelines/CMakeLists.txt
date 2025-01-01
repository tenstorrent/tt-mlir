add_mlir_dialect_library(MLIRTTKernelPipelines
  TTKernelPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  LINK_LIBS PUBLIC
  MLIRTTKernelDialect
)
