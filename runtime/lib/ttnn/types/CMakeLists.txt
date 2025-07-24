if (NOT TTNN_RUNTIME_ENABLED)
  add_library(TTRuntimeTTNNTypes INTERFACE)
  return()
endif()

add_library(TTRuntimeTTNNTypes
  STATIC
  types.cpp
  layout_converter.cpp
  trace_cache.cpp
)
set_property(TARGET TTRuntimeTTNNTypes PROPERTY CXX_STANDARD 20)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  target_compile_options(TTRuntimeTTNNTypes PUBLIC -march=x86-64-v3)
endif()
target_include_directories(TTRuntimeTTNNTypes PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
add_dependencies(TTRuntimeTTNNTypes FBS_GENERATION TTRuntimeTTNNDebug)
target_link_libraries(TTRuntimeTTNNTypes PUBLIC TTNN::CPP TTRuntimeTTNNDebug)
target_link_libraries(TTRuntimeTTNNTypes PUBLIC coverage_config)
