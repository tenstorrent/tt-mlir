include(AddMLIRPython)

set(TTRT_SOURCES)
file(GLOB_RECURSE TTRT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach (filename ${TTRT_FILES})
    cmake_path(RELATIVE_PATH filename BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE relative_filename)
    configure_file("${filename}" "${CMAKE_CURRENT_BINARY_DIR}/${relative_filename}" COPYONLY)
    list(APPEND TTRT_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${relative_filename}")
endforeach (filename)

# Create symlink for runtime SO in source repo
add_custom_command(
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${TTMLIR_BINARY_DIR}/runtime/python/_ttmlir_runtime.cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu.so
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/_ttmlir_runtime.cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu.so
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS _ttmlir_runtime
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/runtime/_ttmlir_runtime.cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu.so
  COMMENT "Creating symlink for ttrt runtime SO"
)

# Editable install (only needs to run once)
add_custom_command(
  COMMAND python -m pip install -r "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
  COMMAND TTMLIR_VERSION_MAJOR=${TTMLIR_VERSION_MAJOR}
          TTMLIR_VERSION_MINOR=${TTMLIR_VERSION_MINOR}
          TTMLIR_VERSION_PATCH=${TTMLIR_VERSION_PATCH}
          SOURCE_ROOT=${TTMLIR_SOURCE_DIR}
          TTMLIR_BINARY_DIR=${TTMLIR_BINARY_DIR}
          python -m pip install -e ${CMAKE_CURRENT_SOURCE_DIR} --no-deps
  COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/.editable_installed
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/runtime/_ttmlir_runtime.cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu.so
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/.editable_installed
  COMMENT "Installing ttrt in editable mode"
)

add_custom_target(ttrt ALL
  COMMENT "python ttrt package"
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/.editable_installed)

declare_mlir_python_sources(TTRTSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
    library_tweaks.py

    # binary subpackage files
    binary/__init__.py
    binary/stats.py

    # common subpackage files
    common/__init__.py
    common/api.py
    common/callback.py
    common/check.py
    common/perf.py
    common/query.py
    common/read.py
    common/run.py
    common/util.py
    common/emitpy.py
    common/emitc.py

    # runtime subpackage files
    runtime/__init__.py
)

add_mlir_python_modules(TTRTPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/ttrt"
  INSTALL_PREFIX "python_packages/ttrt"
  DECLARED_SOURCES TTRTSources
)
