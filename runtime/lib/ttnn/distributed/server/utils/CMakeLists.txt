if (NOT TT_RUNTIME_ENABLE_TTNN)
  add_library(TTRuntimeTTNNServerUtils INTERFACE)
  return()
endif()

add_library(TTRuntimeTTNNServerUtils
  STATIC
  utils.cpp
)

set_target_properties(TTRuntimeTTNNServerUtils PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeTTNNServerUtils PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
