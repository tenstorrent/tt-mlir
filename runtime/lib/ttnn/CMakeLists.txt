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

add_library(TTRuntimeTTNN
  STATIC
  runtime.cpp
  program_executor.cpp
)
# We have to set the C++ standard to 20 because tt-metal requires it
set_property(TARGET TTRuntimeTTNN PROPERTY CXX_STANDARD 20)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Configure path to Python scripts
set(TTMLIR_PYTHON_SCRIPTS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../python/scripts")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/python_scripts_path.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/python_scripts_path.h
  @ONLY
)

target_include_directories(TTRuntimeTTNN PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_include_directories(TTRuntimeTTNN SYSTEM PRIVATE ${pybind11_INCLUDE_DIRS})
target_link_libraries(TTRuntimeTTNN PRIVATE Python::Python)
target_include_directories(TTRuntimeTTNN SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTNN PUBLIC TTRuntimeTTNNOps TTRuntimeTTNNTypes TTRuntimeTTNNUtils)
target_link_libraries(TTRuntimeTTNN PUBLIC coverage_config)
add_dependencies(TTRuntimeTTNN TTRuntimeTTNNOps TTRuntimeTTNNTypes TTRuntimeTTNNUtils)
