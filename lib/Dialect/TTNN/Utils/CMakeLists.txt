add_mlir_dialect_library(TTMLIRTTNNUtils
  Conv2dConfigParams.cpp
  OptimizerOverrides.cpp
  OptimizerUtils.cpp
  PassOverrides.cpp
  TransformUtils.cpp
  Utils.cpp


  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TTNN

  LINK_LIBS
  PRIVATE
  MLIRTTCoreDialect
)
