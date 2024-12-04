add_library(TTRuntimeTTMetal
  STATIC
  runtime.cpp
  command_queue.cpp
)
# We have to set the C++ standard to 20 because tt-metal requires it
set_property(TARGET TTRuntimeTTMetal PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeTTMetal PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTMetal SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTMetal PUBLIC TTMETAL_LIBRARY DEVICE_LIBRARY)
add_dependencies(TTRuntimeTTMetal TTMETAL_LIBRARY DEVICE_LIBRARY tt-metal FBS_GENERATION)

# Optionally compile profiling code and link tracy client for perf profiling.
