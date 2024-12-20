add_library(TTRuntimeTTNNHelpers
  STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tt/runtime/ttnn/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tt/runtime/ttnn/types.cpp
)
set_property(TARGET TTRuntimeTTNNHelpers PROPERTY CXX_STANDARD 20)
target_compile_options(TTRuntimeTTNNHelpers PUBLIC -mavx -mavx2 -fsized-deallocation)
target_include_directories(TTRuntimeTTNNHelpers PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/operations/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNHelpers SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeTTNNHelpers TTNN_LIBRARY tt-metal FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNHelpers PUBLIC TTNN_LIBRARY)

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
target_link_libraries(TTRuntimeTTNN PUBLIC TTRuntimeTTNNOps TTRuntimeTTNNHelpers)
add_dependencies(TTRuntimeTTNN TTRuntimeTTNNOps)
