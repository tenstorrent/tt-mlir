add_library(TTRuntimeTTNNUtils
  STATIC
  include/tt/runtime/ttnn/utils.cpp
)
set_property(TARGET TTRuntimeTTNNUtils PROPERTY CXX_STANDARD 20)
target_compile_options(TTRuntimeTTNNUtils PUBLIC -mavx -mavx2 -fsized-deallocation)
target_include_directories(TTRuntimeTTNNUtils PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/operations/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNUtils PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeTTNNUtils TTNN_LIBRARY tt-metal FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNUtils PUBLIC TTNN_LIBRARY)

add_subdirectory(operations)

add_library(TTRuntimeTTNN
  STATIC
  runtime.cpp
  program.cpp
)
# We have to set the C++ standard to 20 because tt-metal requires it
set_property(TARGET TTRuntimeTTNN PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeTTNN PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNN PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTNN PUBLIC TTRuntimeTTNNOps TTRuntimeTTNNUtils)
add_dependencies(TTRuntimeTTNN TTRuntimeTTNNOps)
