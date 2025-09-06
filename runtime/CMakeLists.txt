# Options
if (TTMLIR_ENABLE_RUNTIME)
  option(TT_RUNTIME_ENABLE_TTNN "Enable TTNN Runtime (requires TTMLIR_ENABLE_RUNTIME)" ON)
  option(TT_RUNTIME_ENABLE_TTMETAL "Enable TTMetal Runtime (requires TTMLIR_ENABLE_RUNTIME)" ON)
else()
  set(TT_RUNTIME_ENABLE_TTNN OFF CACHE BOOL "Enable TTNN Runtime (requires TTMLIR_ENABLE_RUNTIME)" FORCE)
  set(TT_RUNTIME_ENABLE_TTMETAL OFF CACHE BOOL "Enable TTMetal Runtime (requires TTMLIR_ENABLE_RUNTIME)" FORCE)
endif()
option(TTMLIR_ENABLE_RUNTIME_TESTS "Enable runtime tests" OFF)
option(TT_RUNTIME_DEBUG "Enable debug tools in runtime" OFF)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Asan")
  set(TT_RUNTIME_DEBUG ON)
endif()

if (CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
endif()


set(TT_RUNTIME_OPTIONS
  TT_RUNTIME_ENABLE_TTNN
  TT_RUNTIME_ENABLE_TTMETAL
  TT_RUNTIME_DEBUG
  TT_RUNTIME_ENABLE_PERF_TRACE
  TTMLIR_ENABLE_RUNTIME_TESTS
)

if (TTMLIR_ENABLE_RUNTIME AND (TT_RUNTIME_ENABLE_TTNN OR TT_RUNTIME_ENABLE_TTMETAL))
  set(TT_RUNTIME_ENABLED ON)
else()
  set(TT_RUNTIME_ENABLED OFF)
endif()

foreach(OPTION ${TT_RUNTIME_OPTIONS})
  if (${OPTION})
    add_definitions(-D${OPTION})
  endif()
endforeach()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(tools)

# Ensure coverage_config is applied to runtime tests if enabled
if (TTMLIR_ENABLE_RUNTIME_TESTS)
  add_subdirectory(test)
else()
  add_library(TTRuntimeTTNNTestLib INTERFACE)
endif()

add_subdirectory(bin)

# Add coverage_config to all targets in runtime
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target IN ITEMS TTRuntimeTTNNTestLib)
    if (TARGET ${target})
      target_link_libraries(${target} INTERFACE coverage_config)
    endif()
  endforeach()
endif()
