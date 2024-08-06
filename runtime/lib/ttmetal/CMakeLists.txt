add_library(TTRuntimeTTMetal
  STATIC
  runtime.cpp
  command_queue.cpp
)
target_include_directories(TTRuntimeTTMetal PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTMetal PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeTTMetal tt-metal FBS_GENERATION)
