if (NOT TT_RUNTIME_ENABLED)
  return()
endif()

#
# Runtime Client Executable
#

add_executable(runtime-client runtime-client.cpp)

set_target_properties(runtime-client PROPERTIES CXX_STANDARD 20)

target_include_directories(runtime-client PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(runtime-client SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(runtime-client RUNTIME_FBS_GENERATION)

target_link_libraries(runtime-client PRIVATE
  TTMLIRRuntime
  TTRuntimeDistributedClient
  TTNN_LIBRARY
  TTMETAL_LIBRARY
  DEVICE_LIBRARY
  TT_STL_LIBRARY
)
