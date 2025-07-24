if (NOT TTNN_RUNTIME_ENABLED)
  add_library(TTRuntimeTTNNUtils INTERFACE)
  return()
endif()

add_library(TTRuntimeTTNNUtils
  STATIC
  utils.cpp
)
set_property(TARGET TTRuntimeTTNNUtils PROPERTY CXX_STANDARD 20)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  target_compile_options(TTRuntimeTTNNUtils PUBLIC -march=x86-64-v3)
endif()
target_include_directories(TTRuntimeTTNNUtils PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
add_dependencies(TTRuntimeTTNNUtils FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNUtils PUBLIC TTNN::CPP TT::Metalium coverage_config)
