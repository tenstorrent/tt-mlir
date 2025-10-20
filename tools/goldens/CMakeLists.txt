include(AddMLIRPython)

declare_mlir_python_sources(GoldenSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
    custom_goldens.py
    mappings.py
    utils.py
)

add_mlir_python_modules(GoldenPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/goldens"
  INSTALL_PREFIX "python_packages/goldens"
  DECLARED_SOURCES GoldenSources
)
