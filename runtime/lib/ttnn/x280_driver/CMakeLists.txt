if (NOT TTNN_RUNTIME_ENABLED)
  add_library(TTRuntimeX280Driver INTERFACE)
  return()
endif()

add_library(TTRuntimeX280Driver
  STATIC
  driver.cpp
)
set_property(TARGET TTRuntimeX280Driver PROPERTY CXX_STANDARD 20)
target_include_directories(TTRuntimeX280Driver PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/x280_driver
)
target_include_directories(TTRuntimeX280Driver SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeX280Driver PUBLIC DEVICE_LIBRARY)
target_link_libraries(TTRuntimeX280Driver PUBLIC coverage_config)
add_dependencies(TTRuntimeX280Driver tt-metal)
