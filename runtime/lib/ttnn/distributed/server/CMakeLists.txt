add_subdirectory(utils)

if (NOT TT_RUNTIME_ENABLE_TTNN)
  add_library(TTRuntimeTTNNServer INTERFACE)
  return()
endif()

add_library(TTRuntimeTTNNServer
  STATIC
  command_factory.cpp
)

set_target_properties(TTRuntimeTTNNServer PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeTTNNServer PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNServer SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(TTRuntimeTTNNServer FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNServer PRIVATE TTBinary TTRuntimeTypes TTRuntimeTTNNServerUtils TTRuntimeTTNN flatbuffers)
