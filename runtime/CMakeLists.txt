if (TTMLIR_ENABLE_RUNTIME AND (TT_RUNTIME_ENABLE_TTNN OR TT_RUNTIME_ENABLE_TTMETAL))
  set(DEVICE_RUNTIME_ENABLED ON)
else()
  set(DEVICE_RUNTIME_ENABLED OFF)
endif()

if (TTMLIR_ENABLE_RUNTIME AND TT_RUNTIME_ENABLE_DISTRIBUTED)
  set(DISTRIBUTED_RUNTIME_ENABLED ON)
else()
  set(DISTRIBUTED_RUNTIME_ENABLED OFF)
endif()

set(TT_RUNTIME_OPTIONS
  DEVICE_RUNTIME_ENABLED
  TT_RUNTIME_ENABLE_TTNN
  TT_RUNTIME_ENABLE_TTMETAL
  TT_RUNTIME_ENABLE_DISTRIBUTED
  TT_RUNTIME_DEBUG
  TT_RUNTIME_ENABLE_PERF_TRACE
  TTMLIR_ENABLE_RUNTIME_TESTS
)

foreach(OPTION ${TT_RUNTIME_OPTIONS})
  if (${OPTION})
    add_definitions(-D${OPTION})
  endif()
endforeach()

set(METAL_LIBS TTMETAL_LIBRARY DEVICE_LIBRARY TTNN_LIBRARY TT_STL_LIBRARY TRACY_LIBRARY)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(tools)

if (TT_RUNTIME_DEBUG)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  add_library(TTRuntimeChisel STATIC lib/common/chisel_bridge.cpp)
  set_property(TARGET TTRuntimeChisel PROPERTY CXX_STANDARD 20)
  target_include_directories(TTRuntimeChisel PUBLIC
    ${Python3_INCLUDE_DIRS}
  )
  target_link_libraries(TTRuntimeChisel PUBLIC ttmlir_runtime_bindings)
  target_link_libraries(TTRuntimeChisel PUBLIC
    ${Python3_LIBRARIES}
  )
  target_link_libraries(TTRuntimeChisel PUBLIC coverage_config)
endif()

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
