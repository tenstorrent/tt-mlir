add_mlir_conversion_library(TTMLIRTTIRToTTIRGeneric
  TTIRToTTIRGeneric.cpp
  TTIRToTTIRGenericPass.cpp

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRTTNNDialect
)
