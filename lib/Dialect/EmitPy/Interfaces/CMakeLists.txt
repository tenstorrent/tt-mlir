add_mlir_library(MLIREmitPyInterfaces
  EmitPyInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  DEPENDS
  MLIREmitPyInterfacesIncGen

  LINK_LIBS
  PUBLIC
  MLIRIR
)
