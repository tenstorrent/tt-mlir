include(AddMLIRPython)

declare_mlir_python_sources(RuntimeSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    # runtime
    base/__init__.py
    base/runtime.py
    base/runtime_utils.py
    base/runtime_runtime.py
    base/runtime_apis.py
    base/runtime_enums.py

    # ttir subpackage files
    ttir/__init__.py
    ttir/ttir_runtime.py

    # stablehlo subpackage files
    stablehlo/__init__.py
    stablehlo/stablehlo_runtime.py
    stablehlo/shardy_parallelization_utils.py

    # d2m subpackage files
    d2m/__init__.py
    d2m/d2m_runtime.py

    # ttnn subpackage files
    ttnn/__init__.py
    ttnn/ttnn_runtime.py
)

add_mlir_python_modules(RuntimePythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/runtime"
  INSTALL_PREFIX "python_packages/runtime"
  DECLARED_SOURCES RuntimeSources
)
