add_library(TTRuntimeTTNNTestHelpers
  STATIC
  utils.cpp
  dylib.cpp
)
set_property(TARGET TTRuntimeTTNNTestHelpers PROPERTY CXX_STANDARD 20)
target_compile_options(TTRuntimeTTNNTestHelpers PUBLIC -mavx -mavx2 -fsized-deallocation)
target_include_directories(TTRuntimeTTNNTestHelpers PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNTestHelpers SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeTTNNTestHelpers TTMLIRRuntime tt-metal FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNTestHelpers PUBLIC TTMLIRRuntime TTNN_LIBRARY)
