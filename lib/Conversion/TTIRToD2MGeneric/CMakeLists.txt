add_mlir_conversion_library(TTMLIRTTIRToD2MGeneric
  TTIRToD2MGeneric.cpp

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
