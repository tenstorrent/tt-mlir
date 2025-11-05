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

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_library(TTRuntimeTTNN
  STATIC
  runtime.cpp
  program_executor.cpp
)
# We have to set the C++ standard to 20 because tt-metal requires it
set_property(TARGET TTRuntimeTTNN PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeTTNN PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
  ${Python3_INCLUDE_DIRS}
)
target_include_directories(TTRuntimeTTNN SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTNN PUBLIC TTRuntimeTTNNOps TTRuntimeTTNNTypes TTRuntimeTTNNUtils)
target_link_libraries(TTRuntimeTTNN PUBLIC coverage_config)
target_link_libraries(TTRuntimeTTNN PUBLIC ${Python3_LIBRARIES})
add_dependencies(TTRuntimeTTNN TTRuntimeTTNNOps TTRuntimeTTNNTypes TTRuntimeTTNNUtils RUNTIME_FBS_GENERATION)
