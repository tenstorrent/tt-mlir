include(AddMLIRPython)

declare_mlir_python_sources(ChiselSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/chisel"
  SOURCES
    __init__.py
    bind.py
    callbacks.py
    context.py
    checker.py
    ops.py
    executor.py
    exceptions.py
    tensors.py
    op_configs.py
    utils.py
)

add_mlir_python_modules(ChiselPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/chisel"
  INSTALL_PREFIX "python_packages/chisel"
  DECLARED_SOURCES ChiselSources
)
