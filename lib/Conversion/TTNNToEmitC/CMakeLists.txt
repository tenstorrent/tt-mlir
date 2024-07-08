add_mlir_library(MLIRTTNNToEmitC
  TTNNToEmitC.cpp

  DEPENDS
  MLIRTTNNDialect
  MLIRTTNNConversionPassIncGen
#   MLIREmitCDialect
#   MLIREmitCConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
