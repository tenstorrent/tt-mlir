add_custom_target(ttrt-copy-files
  COMMAND
    cp -r ${CMAKE_CURRENT_SOURCE_DIR}/* ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(ttrt
  COMMAND TTMLIR_ENABLE_RUNTIME=${TTMLIR_ENABLE_RUNTIME} SOURCE_ROOT=${TTMLIR_SOURCE_DIR} python -m pip install .
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "python ttrt package"
  DEPENDS ttrt-copy-files
)
add_dependencies(ttrt TTRuntime)
