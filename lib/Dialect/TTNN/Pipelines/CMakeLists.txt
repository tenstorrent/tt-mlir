if (TTMLIR_ENABLE_OPMODEL)
  add_definitions(-DTTMLIR_ENABLE_OPMODEL)
endif()

add_mlir_dialect_library(MLIRTTNNPipelines
  TTNNPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  LINK_LIBS PUBLIC
  MLIRLLVMTransforms
  MLIRTTIRDialect
  MLIRTTUtils
  MLIRTTNNDialect
  MLIRTTIRPipelines
  MLIRTTIRTransforms
  MLIRTTNNTransforms
  MLIRTTNNAnalysis
  MLIRTTCoreDialect
  MLIRTTTransforms
  MLIRPass
  MLIRTransforms
)
