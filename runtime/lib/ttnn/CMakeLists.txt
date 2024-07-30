add_library(TTRuntimeTTNN
  STATIC
  runtime.cpp
  program.cpp
)
target_compile_options(TTRuntimeTTNN PRIVATE -mavx -mavx2)
target_include_directories(TTRuntimeTTNN PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNN PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTNN PUBLIC TTNN_LIBRARY)
add_dependencies(TTRuntimeTTNN TTNN_LIBRARY tt-metal FBS_GENERATION)
