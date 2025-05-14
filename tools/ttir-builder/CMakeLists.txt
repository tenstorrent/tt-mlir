include(AddMLIRPython)

declare_mlir_python_sources(TTIRBuilder)

declare_mlir_python_sources(TTIRBuilderSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT TTIRBuilder
  SOURCES
    ccl_golden.py
    test_optimizer_overrides.py
    test_utils.py
    ttir_builder.py
)

add_mlir_python_modules(TTIRBuilderModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/ttir_builder"
    INSTALL_PREFIX "python_packages/ttir_builder"
    DECLARED_SOURCES TTIRBuilderSources
)
