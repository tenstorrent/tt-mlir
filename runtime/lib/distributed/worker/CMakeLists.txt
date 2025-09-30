if (NOT DISTRIBUTED_RUNTIME_ENABLED)
  add_library(TTRuntimeDistributedWorker INTERFACE)
  return()
endif()

add_library(TTRuntimeDistributedWorker
  STATIC
  response_factory.cpp
  command_executor.cpp
)

set_target_properties(TTRuntimeDistributedWorker PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeDistributedWorker PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeDistributedWorker SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(TTRuntimeDistributedWorker RUNTIME_FBS_GENERATION)
target_link_libraries(TTRuntimeDistributedWorker PRIVATE TTMLIRRuntime TTRuntimeSocket TTRuntimeDistributedUtils)
