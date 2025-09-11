if (NOT DISTRIBUTED_RUNTIME_ENABLED)
  return()
endif()

#
# Runtime Client Executable
#

add_executable(client client.cpp)

set_target_properties(client PROPERTIES CXX_STANDARD 20)

target_include_directories(client PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(client SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(client RUNTIME_FBS_GENERATION)

target_link_libraries(client PRIVATE
  TTMLIRRuntime
  TTRuntimeDistributedClient
  TTNN_LIBRARY
  TTMETAL_LIBRARY
  DEVICE_LIBRARY
  TT_STL_LIBRARY
)
