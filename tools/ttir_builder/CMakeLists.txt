include(AddMLIRPython)

set(TTIRBUILDER_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
declare_mlir_python_sources(TTIRBuilder)

declare_mlir_python_sources(TTIRBuilderSources
  ROOT_DIR "${TTIRBUILDER_ROOT_DIR}"
  ADD_TO_PARENT TTIRBuilder
  SOURCES
    __init__.py
    ccl_golden.py
    utils.py
    builder.py
)

add_mlir_python_modules(TTIRBuilderModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/ttir_builder"
    INSTALL_PREFIX "python_packages/ttir_builder"
    DECLARED_SOURCES TTIRBuilderSources
)
