add_mlir_conversion_library(TTMLIRTTIRToTTKernel
  TTIRToTTKernel.cpp
  TTIRToTTKernelPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTIRToTTKernel

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
