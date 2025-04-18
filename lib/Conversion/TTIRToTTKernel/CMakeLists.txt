add_mlir_conversion_library(TTMLIRTTIRToTTKernel
  TTIRToTTKernel.cpp
  TTIRToTTKernelPass.cpp

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTTIRAnalysis
)
