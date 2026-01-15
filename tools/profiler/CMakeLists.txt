include(AddMLIRPython)

declare_mlir_python_sources(ProfilerSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
    profiler.py
)

add_mlir_python_modules(ProfilerPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/profiler"
  INSTALL_PREFIX "python_packages/profiler"
  DECLARED_SOURCES ProfilerSources
)
