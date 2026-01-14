add_mlir_dialect_library(MLIREmitPyTransforms
  SplitFiles.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Dialect/EmitPy/Transforms

  DEPENDS
  MLIREmitPyPassesIncGen

  LINK_LIBS PUBLIC
  MLIREmitPyDialect
  MLIRFuncDialect
  MLIRPass
  MLIRIR
  MLIRSupport
)
