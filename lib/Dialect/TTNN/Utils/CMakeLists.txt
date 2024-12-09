add_mlir_dialect_library(TTMLIRTTNNUtils
  OptimizerOverrides.cpp
  PassOverrides.cpp
  TransformUtils.cpp
  Utils.cpp


  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TTNN
)
