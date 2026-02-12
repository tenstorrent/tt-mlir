add_mlir_conversion_library(TTMLIRTTIRToTTMetal
  TTIRToTTMetal.cpp
  ElementwiseBinaryOps.cpp  # NEW FILE

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRTTIRDialect
  MLIRTTMetalDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)