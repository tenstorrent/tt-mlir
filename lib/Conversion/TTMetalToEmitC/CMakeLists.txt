add_mlir_conversion_library(TTMLIRTTMetalToEmitC
  TTMetalToEmitC.cpp
  TTMetalToEmitCPass.cpp
  Utils.cpp

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRTTCoreDialect
  MLIRTTMetalDialect
)