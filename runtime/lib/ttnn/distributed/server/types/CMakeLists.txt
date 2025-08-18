if (NOT TT_RUNTIME_ENABLE_TTNN)
  add_library(TTRuntimeTTNNServerTypes INTERFACE)
  return()
endif()

add_library(TTRuntimeTTNNServerTypes
  STATIC
  command_factory.cpp
)

set_target_properties(TTRuntimeTTNNServerTypes PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeTTNNServerTypes PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNServerTypes SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(TTRuntimeTTNNServerTypes TTRuntimeTypes TTRuntimeTTNNServerUtils TTRuntimeTTNN FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNServerTypes PRIVATE TTRuntimeTypes TTRuntimeTTNNServerUtils TTRuntimeTTNN flatbuffers)
