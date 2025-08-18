if (NOT TT_RUNTIME_ENABLE_TTNN)
  add_library(TTRuntimeTTNNClientTypes INTERFACE)
  return()
endif()

add_library(TTRuntimeTTNNClientTypes
  STATIC
  response_factory.cpp
)

set_target_properties(TTRuntimeTTNNClientTypes PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeTTNNClientTypes PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNClientTypes SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(TTRuntimeTTNNClientTypes TTRuntimeTypes TTRuntimeTTNNClientUtils TTRuntimeTTNN FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNClientTypes PRIVATE TTRuntimeTypes TTRuntimeTTNNClientUtils TTRuntimeTTNN flatbuffers)
