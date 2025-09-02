add_mlir_dialect_library(MLIRTTNNValidation
  OpConstraintValidation.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  DEPENDS
  MLIRTTNNOpsIncGen
  MLIRTTCoreOpsIncGen
  TTNNOpModelLib

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTTCoreDialect
  MLIRTTNNDialect
  TTMLIRTTNNUtils
  TTNNOpModelLib
)
