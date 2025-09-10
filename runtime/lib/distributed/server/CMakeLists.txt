if (NOT TT_RUNTIME_ENABLED)
  add_library(TTRuntimeDistributedServer INTERFACE)
  return()
endif()

add_library(TTRuntimeDistributedServer
  STATIC
  command_factory.cpp
)

set_target_properties(TTRuntimeDistributedServer PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeDistributedServer PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeDistributedServer SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(TTRuntimeDistributedServer RUNTIME_FBS_GENERATION)
target_link_libraries(TTRuntimeDistributedServer PRIVATE TTBinary TTRuntimeTypes TTRuntimeTTNN)
