add_mlir_dialect_library(MLIRSFPUDialect
  SFPUDialect.cpp
  SFPUOps.cpp
  SFPUOpsTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Dialect/SFPU

  DEPENDS
  MLIRSFPUOpsIncGen
  MLIRSFPUOpsTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRArithDialect
  MLIRVectorDialect
  MLIRTTCoreDialect
)