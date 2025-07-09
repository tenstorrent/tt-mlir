add_mlir_dialect_library(TTMLIRSTABLEHLOUtils
  MeshShardingUtils.cpp
  ShardyUtils.cpp


  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/StableHLO
)
