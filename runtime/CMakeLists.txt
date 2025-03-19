# Options
option(TTMLIR_ENABLE_RUNTIME_TESTS "Enable runtime tests" OFF)
option(TT_RUNTIME_ENABLE_TTNN "Enable TTNN Runtime" ON)
option(TT_RUNTIME_ENABLE_TTMETAL "Enable TTMetal Runtime" ON)
option(TT_RUNTIME_DEBUG "Enable debug tools in runtime" OFF)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(TT_RUNTIME_DEBUG ON)
endif()

set(TT_RUNTIME_OPTIONS
  TT_RUNTIME_DEBUG
  TT_RUNTIME_ENABLE_PERF_TRACE
  TTMLIR_ENABLE_RUNTIME_TESTS
)

foreach(OPTION ${TT_RUNTIME_OPTIONS})
  if (${OPTION})
    add_definitions(-D${OPTION})
  endif()
endforeach()

# Apply coverage_config to the main target
target_link_libraries(TTMLIR INTERFACE coverage_config)

add_subdirectory(lib)
add_subdirectory(tools)

# Ensure coverage_config is applied to runtime tests if enabled
if (TTMLIR_ENABLE_RUNTIME_TESTS)
  add_subdirectory(test)
else()
  add_library(TTRuntimeTTNNTestLib INTERFACE)
  target_link_libraries(TTRuntimeTTNNTestLib INTERFACE coverage_config)
endif()
