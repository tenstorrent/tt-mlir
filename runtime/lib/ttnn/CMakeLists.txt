add_library(TTRuntimeTTNNTypes
  STATIC
  types.cpp
)
set_property(TARGET TTRuntimeTTNNTypes PROPERTY CXX_STANDARD 20)
target_compile_options(TTRuntimeTTNNTypes PUBLIC -mavx -mavx2 -fsized-deallocation)
target_include_directories(TTRuntimeTTNNTypes PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNTypes SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeTTNNTypes TTNN_LIBRARY tt-metal FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNTypes PUBLIC TTNN_LIBRARY)


add_library(TTRuntimeTTNNUtils
  STATIC
  utils.cpp
)
set_property(TARGET TTRuntimeTTNNUtils PROPERTY CXX_STANDARD 20)
target_compile_options(TTRuntimeTTNNUtils PUBLIC -mavx -mavx2 -fsized-deallocation)
target_include_directories(TTRuntimeTTNNUtils PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNUtils SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
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
target_include_directories(TTRuntimeTTNN PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn
)
target_include_directories(TTRuntimeTTNN PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNN SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTNN PUBLIC TTRuntimeTTNNOps TTRuntimeTTNNTypes TTRuntimeTTNNUtils)
add_dependencies(TTRuntimeTTNN TTRuntimeTTNNOps)
