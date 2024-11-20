add_mlir_library(TTMLIRTTKernelToEmitC
  TTIRToTTNN.cpp
  TTIRToTTNNPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTKernelToEmitC

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
