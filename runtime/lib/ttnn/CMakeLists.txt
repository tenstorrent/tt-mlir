add_subdirectory(types)
add_subdirectory(utils)
add_subdirectory(operations)

add_library(TTRuntimeTTNN
  STATIC
  runtime.cpp
  program.cpp
)
# We have to set the C++ standard to 20 because tt-metal requires it
set_property(TARGET TTRuntimeTTNN PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeTTNN PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn
)
target_include_directories(TTRuntimeTTNN PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNN SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTNN PUBLIC TTRuntimeTTNNOps TTRuntimeTTNNTypes TTRuntimeTTNNUtils)
add_dependencies(TTRuntimeTTNN TTRuntimeTTNNOps TTRuntimeTTNNTypes TTRuntimeTTNNUtils)
