find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Options
option(TTMLIR_ENABLE_RUNTIME_TESTS "Enable runtime tests" OFF)
option(TT_RUNTIME_ENABLE_TTNN "Enable TTNN Runtime" ON)
option(TT_RUNTIME_ENABLE_TTMETAL "Enable TTMetal Runtime" ON)
option(TT_RUNTIME_DEBUG "Enable debug tools in runtime" OFF)
option(TT_RUNTIME_WORKAROUNDS "Enable toggling workarounds in runtime" OFF)

include_directories(${PYBIND11_INCLUDE_DIR})
link_libraries(Python3::Python pybind11::pybind11 pybind11::embed)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(TT_RUNTIME_DEBUG ON)
  set(TT_RUNTIME_WORKAROUNDS ON)
endif()

set(TT_RUNTIME_OPTIONS
  TT_RUNTIME_DEBUG
  TT_RUNTIME_ENABLE_PERF_TRACE
  TT_RUNTIME_WORKAROUNDS
)

foreach(OPTION ${TT_RUNTIME_OPTIONS})
  if (${OPTION})
    add_definitions(-D${OPTION})
  endif()
endforeach()

add_subdirectory(lib)
add_subdirectory(tools)
if (TTMLIR_ENABLE_RUNTIME_TESTS)
    add_subdirectory(test)
endif()
