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
    ttir/ttir_builder_utils.py

    # stablehlo subpackage files
    stablehlo/__init__.py
    stablehlo/stablehlo_builder.py

    # d2m subpackage files
    d2m/__init__.py
    d2m/d2m_builder.py
)

add_mlir_python_modules(BuilderPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/builder"
  INSTALL_PREFIX "python_packages/builder"
  DECLARED_SOURCES BuilderSources
)
