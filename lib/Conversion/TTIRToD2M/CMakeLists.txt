add_mlir_conversion_library(TTMLIRTTIRToD2M
  TTIRToD2M.cpp

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRTTCoreDialect
  MLIRTTIRDialect
  MLIRD2MDialect
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRTransformUtils
  MLIRTransforms
)
