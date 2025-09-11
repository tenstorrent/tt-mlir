if (NOT DISTRIBUTED_RUNTIME_ENABLED)
  add_library(TTRuntimeDistributedUtils INTERFACE)
  return()
endif()

add_library(TTRuntimeDistributedUtils
  STATIC
  utils.cpp
)

set_target_properties(TTRuntimeDistributedUtils PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeDistributedUtils PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeDistributedUtils SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(TTRuntimeDistributedUtils RUNTIME_FBS_GENERATION)
