if (NOT TT_RUNTIME_ENABLED)
  add_library(TTRuntimeDistributedClient INTERFACE)
  return()
endif()

add_library(TTRuntimeDistributedClient
  STATIC
  response_factory.cpp
  command_executor.cpp
)

set_target_properties(TTRuntimeDistributedClient PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeDistributedClient PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeDistributedClient SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(TTRuntimeDistributedClient RUNTIME_FBS_GENERATION)
target_link_libraries(TTRuntimeDistributedClient PRIVATE TTMLIRRuntime TTRuntimeSocket)
