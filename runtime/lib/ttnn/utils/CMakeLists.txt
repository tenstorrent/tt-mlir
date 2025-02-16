add_library(TTRuntimeTTNNUtils
  STATIC
  utils.cpp
)
set_property(TARGET TTRuntimeTTNNUtils PROPERTY CXX_STANDARD 20)
target_compile_options(TTRuntimeTTNNUtils PUBLIC -mavx -mavx2)
target_include_directories(TTRuntimeTTNNUtils PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNUtils SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeTTNNUtils TTNN_LIBRARY tt-metal FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNUtils PUBLIC TTNN_LIBRARY)
