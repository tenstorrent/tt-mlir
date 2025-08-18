if (NOT TT_RUNTIME_ENABLE_TTNN)
  add_library(TTRuntimeTTNNClientUtils INTERFACE)
  return()
endif()

add_library(TTRuntimeTTNNClientUtils
  STATIC
  utils.cpp
)

set_target_properties(TTRuntimeTTNNClientUtils PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeTTNNClientUtils PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
