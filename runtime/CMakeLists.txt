# Options
option(TTMLIR_ENABLE_RUNTIME_TESTS "Enable runtime tests" OFF)
option(TT_RUNTIME_ENABLE_TTNN "Enable TTNN Runtime" ON)
option(TT_RUNTIME_ENABLE_TTMETAL "Enable TTMetal Runtime" ON)
option(TT_RUNTIME_DEBUG "Enable debug tools in runtime" OFF)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Asan")
  set(TT_RUNTIME_DEBUG ON)
endif()

if (CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
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

add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(tools)

# Ensure coverage_config is applied to runtime tests if enabled
if (TTMLIR_ENABLE_RUNTIME_TESTS)
  add_subdirectory(test)
else()
  add_library(TTRuntimeTTNNTestLib INTERFACE)
endif()

# Add coverage_config to all targets in runtime
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target IN ITEMS TTRuntimeTTNNTestLib)
    if (TARGET ${target})
      target_link_libraries(${target} INTERFACE coverage_config)
    endif()
  endforeach()
endif()
