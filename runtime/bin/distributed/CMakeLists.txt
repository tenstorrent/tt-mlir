if (NOT DISTRIBUTED_RUNTIME_ENABLED)
  return()
endif()

#
# Runtime Worker Executable
#

add_executable(worker worker.cpp)

set_target_properties(worker PROPERTIES CXX_STANDARD 20)

target_include_directories(worker PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(worker SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(worker RUNTIME_FBS_GENERATION)

target_link_libraries(worker PRIVATE
  TTMLIRRuntime
  TTRuntimeDistributedWorker
  ${METAL_LIBS}
)
