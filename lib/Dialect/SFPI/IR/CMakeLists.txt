add_mlir_dialect_library(MLIRSFPIDialect
  SFPIDialect.cpp
  SFPIOps.cpp
  SFPIOpsTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Dialect/SFPI

  DEPENDS
  MLIRSFPIOpsIncGen
  MLIRSFPIOpsTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRArithDialect
  MLIRVectorDialect
  MLIRTTCoreDialect
)
