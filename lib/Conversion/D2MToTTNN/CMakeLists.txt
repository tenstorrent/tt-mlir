add_mlir_conversion_library(TTMLIRD2MToTTNN
  D2MToTTNN.cpp
  D2MToTTNNPass.cpp

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTTIRDialect
  MLIRTTNNDialect
)
