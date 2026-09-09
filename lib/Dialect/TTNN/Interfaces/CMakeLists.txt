add_mlir_library(MLIRTTNNInterfaces
  TTNNDistributedOpInterface.cpp
  TTNNDeviceOperandInterface.cpp
  TTNNOpModelInterface.cpp
  TTNNTensorSpecInterface.cpp
  TTNNWorkaroundInterface.cpp
  TTNNKernelInterface.cpp
  D2MOpCostModel.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  DEPENDS
  MLIRTTNNDistributedOpInterfaceIncGen
  MLIRTTNNDeviceOperandInterfaceIncGen
  MLIRTTNNOpModelInterfaceIncGen
  MLIRTTNNTensorSpecInterfaceIncGen
  MLIRTTNNWorkaroundInterfaceIncGen
  MLIRTTNNKernelInterfaceIncGen
  TTNNOpsInterfacesIncGen
  MLIRTTCoreOpsIncGen

  LINK_LIBS
  PUBLIC
  MLIRIR

  PRIVATE
  MLIRTTCoreDialect
  TTNNOpModelLib
)
