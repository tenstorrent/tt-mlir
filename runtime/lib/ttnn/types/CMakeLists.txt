add_library(TTRuntimeTTNNTypes
  STATIC
  types.cpp
)
set_property(TARGET TTRuntimeTTNNTypes PROPERTY CXX_STANDARD 20)
target_compile_options(TTRuntimeTTNNTypes PUBLIC -mavx -mavx2)
target_include_directories(TTRuntimeTTNNTypes PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNTypes SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeTTNNTypes TTNN_LIBRARY tt-metal FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNTypes PUBLIC TTNN_LIBRARY)
