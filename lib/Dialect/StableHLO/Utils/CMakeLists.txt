add_mlir_dialect_library(TTMLIRSTABLEHLOUtils
  ShardyUtils.cpp
  GSPMDUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/StableHLO

  LINK_LIBS
  PUBLIC
  SdyDialect
  PRIVATE
  SdyCommonFileUtils
)
