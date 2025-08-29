if (TTMLIR_ENABLE_RUNTIME AND TT_RUNTIME_ENABLE_TTMETAL)
  set(TTMETAL_RUNTIME_ENABLED ON)
else()
  set(TTMETAL_RUNTIME_ENABLED OFF)
endif()

if (NOT TTMETAL_RUNTIME_ENABLED)
  add_library(TTRuntimeTTMetal INTERFACE)
  return()
endif()

add_library(TTRuntimeTTMetal
  STATIC
  runtime.cpp
  executor.cpp
  meshshard_utils.cpp
)
# We have to set the C++ standard to 20 because tt-metal requires it
set_property(TARGET TTRuntimeTTMetal PROPERTY CXX_STANDARD 20)
target_compile_definitions(TTRuntimeTTMetal PUBLIC TT_RUNTIME_ENABLE_TTMETAL)
target_include_directories(TTRuntimeTTMetal PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTMetal SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTMetal PUBLIC TTMETAL_LIBRARY DEVICE_LIBRARY)
target_link_libraries(TTRuntimeTTMetal PUBLIC coverage_config)
add_dependencies(TTRuntimeTTMetal TTMETAL_LIBRARY DEVICE_LIBRARY tt-metal FBS_GENERATION)
