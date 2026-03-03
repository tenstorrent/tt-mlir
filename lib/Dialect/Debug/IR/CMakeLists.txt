add_mlir_dialect_library(MLIRDebugDialect
  DebugDialect.cpp
  DebugOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Dialect/Debug

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
