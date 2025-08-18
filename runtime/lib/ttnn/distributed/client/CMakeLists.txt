add_subdirectory(utils)

if (NOT TT_RUNTIME_ENABLE_TTNN)
  add_library(TTRuntimeTTNNClient INTERFACE)
  return()
endif()

add_library(TTRuntimeTTNNClient
  STATIC
  response_factory.cpp
  runtime_client.cpp
)

set_target_properties(TTRuntimeTTNNClient PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeTTNNClient PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNClient SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(TTRuntimeTTNNClient FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNClient PRIVATE TTRuntimeTypes TTRuntimeTTNNClientUtils TTRuntimeTTNN flatbuffers)
