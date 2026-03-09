add_mlir_library(MLIRTTNNInterfaces
  TTNNDeviceOperandInterface.cpp
  TTNNOpModelInterface.cpp
  TTNNTensorSpecInterface.cpp
  TTNNWorkaroundInterface.cpp
  TTNNDecompositionWorkaroundInterface.cpp
  TTNNKernelInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  DEPENDS
  MLIRTTNNDeviceOperandInterfaceIncGen
  MLIRTTNNOpModelInterfaceIncGen
  MLIRTTNNTensorSpecInterfaceIncGen
  MLIRTTNNWorkaroundInterfaceIncGen
  MLIRTTNNDecompositionWorkaroundInterfaceIncGen
  MLIRTTNNKernelInterfaceIncGen

  LINK_LIBS
  PUBLIC
  MLIRIR

  PRIVATE
  MLIRTTCoreDialect
  TTNNOpModelLib
)
