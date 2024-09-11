add_custom_target(ttrt-copy-files
  COMMAND
    cp -r ${CMAKE_CURRENT_SOURCE_DIR}/* ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(ttrt
  COMMAND rm -f build/*.whl
  COMMAND python -m pip install -r requirements.txt
  COMMAND TTMLIR_ENABLE_RUNTIME=${TTMLIR_ENABLE_RUNTIME}
          TT_RUNTIME_ENABLE_TTNN=${TT_RUNTIME_ENABLE_TTNN}
          TT_RUNTIME_ENABLE_TTMETAL=${TT_RUNTIME_ENABLE_TTMETAL}
          TT_RUNTIME_ENABLE_PERF_TRACE=${TT_RUNTIME_ENABLE_PERF_TRACE}
          TTMLIR_VERSION_MAJOR=${TTMLIR_VERSION_MAJOR}
          TTMLIR_VERSION_MINOR=${TTMLIR_VERSION_MINOR}
          TTMLIR_VERSION_PATCH=${TTMLIR_VERSION_PATCH}
          SOURCE_ROOT=${TTMLIR_SOURCE_DIR}
          python -m pip wheel . --wheel-dir build --verbose
  COMMAND python -m pip install build/*.whl --force-reinstall
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "python ttrt package"
  DEPENDS ttrt-copy-files
)

add_dependencies(ttrt TTRuntime)
