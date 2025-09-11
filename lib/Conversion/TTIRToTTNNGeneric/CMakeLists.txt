add_mlir_conversion_library(TTMLIRTTIRToTTNNGeneric
  TTIRToTTNNGeneric.cpp
  TTIRToTTNNGenericPass.cpp

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTTIRDialect
  MLIRTTNNDialect
)
