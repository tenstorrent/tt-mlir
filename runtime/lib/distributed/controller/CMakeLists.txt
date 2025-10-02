if (NOT DISTRIBUTED_RUNTIME_ENABLED)
  add_library(TTRuntimeDistributedController INTERFACE)
  return()
endif()

add_library(TTRuntimeDistributedController
  STATIC
  command_factory.cpp
  controller.cpp
)

set_target_properties(TTRuntimeDistributedController PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeDistributedController PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeDistributedController SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(TTRuntimeDistributedController RUNTIME_FBS_GENERATION)
target_link_libraries(TTRuntimeDistributedController PRIVATE TTBinary TTRuntimeTypes TTRuntimeSocket TTRuntimeDistributedUtils)
