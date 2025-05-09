if (NOT TTNN_RUNTIME_ENABLED OR NOT TT_RUNTIME_DEBUG)
  add_library(TTRuntimeTTNNDebug INTERFACE)
  return()
endif()

add_library(TTRuntimeTTNNDebug
  STATIC
  debug_apis.cpp
)
set_property(TARGET TTRuntimeTTNNDebug PROPERTY CXX_STANDARD 20)
target_compile_options(TTRuntimeTTNNDebug PUBLIC -mavx -mavx2)
target_include_directories(TTRuntimeTTNNDebug PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNDebug SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeTTNNDebug TTRuntimeTTNNUtils)
target_link_libraries(TTRuntimeTTNNDebug PUBLIC TTRuntimeTTNNUtils)
target_link_libraries(TTRuntimeTTNNDebug PUBLIC coverage_config)
