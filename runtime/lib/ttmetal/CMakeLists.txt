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
)
# We have to set the C++ standard to 20 because tt-metal requires it
set_property(TARGET TTRuntimeTTMetal PROPERTY CXX_STANDARD 20)
target_compile_definitions(TTRuntimeTTMetal PUBLIC TT_RUNTIME_ENABLE_TTMETAL)
target_include_directories(TTRuntimeTTMetal PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
# The following 2 lines are to get profiler header tools/profiler/op_profiler.hpp
# We don't want to link against all of TTNN for the metal runtime.
get_target_property(TTNN_TOOLS_INCLUDES TTNN::CPP INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(TTRuntimeTTMetal SYSTEM PRIVATE ${TTNN_TOOLS_INCLUDES})
target_link_libraries(TTRuntimeTTMetal PUBLIC TT::Metalium coverage_config)
add_dependencies(TTRuntimeTTMetal FBS_GENERATION)
