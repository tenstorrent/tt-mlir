add_mlir_dialect_library(MLIRD2MAllocation
  Planner.cpp
  Tools.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ttmlir
)
