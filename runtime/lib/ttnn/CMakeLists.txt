if (TTMLIR_ENABLE_RUNTIME AND TT_RUNTIME_ENABLE_TTNN)
    set(TTNN_RUNTIME_ENABLED ON)
else()
    set(TTNN_RUNTIME_ENABLED OFF)
endif()

add_subdirectory(utils)
add_subdirectory(debug)
add_subdirectory(types)
add_subdirectory(operations)

if (NOT TTNN_RUNTIME_ENABLED)
  add_library(TTRuntimeTTNN INTERFACE)
  return()
endif()

# Find PyTorch
# Try to find torch via Python first
execute_process(
  COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
  OUTPUT_VARIABLE TORCH_CMAKE_PATH
  ERROR_QUIET
)
execute_process(
  COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib', end='')"
  OUTPUT_VARIABLE TORCH_LIB_PATH
  ERROR_QUIET
)
if(TORCH_CMAKE_PATH)
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
endif()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(TTRuntimeTTNN
  STATIC
  runtime.cpp
  program_executor.cpp
)
# We have to set the C++ standard to 20 because tt-metal requires it
set_property(TARGET TTRuntimeTTNN PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeTTNN PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNN SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTNN PUBLIC TTRuntimeTTNNOps TTRuntimeTTNNTypes TTRuntimeTTNNUtils)
target_link_libraries(TTRuntimeTTNN PUBLIC coverage_config)
target_link_libraries(TTRuntimeTTNN PUBLIC ${TORCH_LIBRARIES})
# Add torch library directories to RPATH for this library and any executables linking to it
if(TORCH_LIB_PATH)
  target_link_options(TTRuntimeTTNN PUBLIC "-Wl,-rpath,${TORCH_LIB_PATH}")
endif()
add_dependencies(TTRuntimeTTNN TTRuntimeTTNNOps TTRuntimeTTNNTypes TTRuntimeTTNNUtils)
