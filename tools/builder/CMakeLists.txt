include(AddMLIRPython)

declare_mlir_python_sources(BuilderSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    # builder
    base/__init__.py
    base/builder.py
    base/builder_golden.py
    base/builder_utils.py

    # ttir subpackage files
    ttir/__init__.py
    ttir/ttir_builder.py

    # stablehlo subpackage files
    stablehlo/__init__.py
    stablehlo/stablehlo_builder.py
)

add_mlir_python_modules(BuilderPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/builder"
  INSTALL_PREFIX "python_packages/builder"
  DECLARED_SOURCES BuilderSources
)
