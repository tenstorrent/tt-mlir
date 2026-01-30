file(GLOB PY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

add_custom_target(copy_python_files ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/python_packages/tt_profiler
  COMMAND ${CMAKE_COMMAND} -E copy ${PY_FILES} ${CMAKE_BINARY_DIR}/python_packages/tt_profiler
)

nanobind_add_module(
  _ttmlir_profiler
  __init__.cpp
  tt_profiler.cpp
)

target_include_directories(_ttmlir_profiler
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(_ttmlir_profiler PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/tt_profiler
)