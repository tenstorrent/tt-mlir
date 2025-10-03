add_mlir_conversion_library(TTMLIRD2MToTTKernel
  D2MToTTKernel.cpp
  D2MToTTKernelPass.cpp

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRD2MAnalysis
)
