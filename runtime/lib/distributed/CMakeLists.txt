add_subdirectory(utils)
add_subdirectory(worker)
add_subdirectory(controller)

if (NOT DISTRIBUTED_RUNTIME_ENABLED)
  add_library(TTRuntimeDistributed INTERFACE)
  return()
endif()

add_library(TTRuntimeDistributed STATIC runtime.cpp)

set_target_properties(TTRuntimeDistributed PROPERTIES CXX_STANDARD 20)

target_include_directories(TTRuntimeDistributed PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)

target_include_directories(TTRuntimeDistributed SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

target_link_libraries(TTRuntimeDistributed PRIVATE TTRuntimeDistributedController TTRuntimeDistributedUtils)

add_dependencies(TTRuntimeDistributed RUNTIME_FBS_GENERATION)
