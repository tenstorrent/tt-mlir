set(TTRT_SOURCES)
file(GLOB_RECURSE TTRT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach (filename ${TTRT_FILES})
    cmake_path(RELATIVE_PATH filename BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE relative_filename)
    configure_file("${filename}" "${CMAKE_CURRENT_BINARY_DIR}/${relative_filename}" COPYONLY)
    list(APPEND TTRT_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${relative_filename}")
endforeach (filename)

add_custom_command(
  COMMAND rm -rf build
  COMMAND python -m pip install -r "${CMAKE_CURRENT_BINARY_DIR}/requirements.txt"
  COMMAND TTMLIR_ENABLE_RUNTIME=${TTMLIR_ENABLE_RUNTIME}
          TT_RUNTIME_ENABLE_TTNN=${TT_RUNTIME_ENABLE_TTNN}
          TT_RUNTIME_ENABLE_TTMETAL=${TT_RUNTIME_ENABLE_TTMETAL}
          TTMLIR_ENABLE_RUNTIME_TESTS=${TTMLIR_ENABLE_RUNTIME_TESTS}
          TT_RUNTIME_ENABLE_PERF_TRACE=${TT_RUNTIME_ENABLE_PERF_TRACE}
          TT_RUNTIME_DEBUG=${TT_RUNTIME_DEBUG}
          TTMLIR_BINARY_DIR=${TTMLIR_BINARY_DIR}
          TTMLIR_VERSION_MAJOR=${TTMLIR_VERSION_MAJOR}
          TTMLIR_VERSION_MINOR=${TTMLIR_VERSION_MINOR}
          TTMLIR_VERSION_PATCH=${TTMLIR_VERSION_PATCH}
          CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
          SOURCE_ROOT=${TTMLIR_SOURCE_DIR}
          python -m pip wheel . --wheel-dir build --verbose
  COMMAND python -m pip install build/*.whl --force-reinstall
  COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/build/.installed
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS ${TTRT_SOURCES} ${TTRT_FILES} TTMLIRRuntime TTRuntimeTTNNTypes TTRuntimeTTNNUtils _ttmlir_runtime
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build/.installed
)

add_custom_target(ttrt
  COMMENT "python ttrt package"
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/build/.installed)
