# Options
option(TTMLIR_ENABLE_RUNTIME_TESTS "Enable runtime tests" OFF)
option(TT_RUNTIME_ENABLE_TTNN "Enable TTNN Runtime" ON)
option(TT_RUNTIME_ENABLE_TTMETAL "Enable TTMetal Runtime" ON)
option(TT_RUNTIME_DEBUG "Enable debug tools in runtime" OFF)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(TT_RUNTIME_DEBUG ON)
endif()

if (TT_RUNTIME_DEBUG)
  add_definitions(-DTT_RUNTIME_DEBUG)
endif()

if (TT_RUNTIME_ENABLE_PERF_TRACE)
  add_definitions(-DTT_RUNTIME_ENABLE_PERF_TRACE)
endif()

add_subdirectory(lib)
add_subdirectory(tools)
if (TTMLIR_ENABLE_RUNTIME_TESTS)
    add_subdirectory(test)
endif()
