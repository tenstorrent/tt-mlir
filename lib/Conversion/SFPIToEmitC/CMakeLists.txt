add_mlir_conversion_library(TTMLIRSFPIToEmitC
  SFPIToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/SFPIToEmitC

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRSFPIDialect
)
