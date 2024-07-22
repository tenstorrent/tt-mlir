add_llvm_library(TTMLIRPlugin
  MODULE
  ttmlir-plugin.cpp

  PLUGIN_TOOL
  ttmlir-opt

  LINK_LIBS
  MLIR
)

target_include_directories(
  TTMLIRPlugin
  PRIVATE
  "${TTMLIR_BINARY_DIR}/include"
)
