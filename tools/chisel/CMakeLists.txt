include(AddMLIRPython)

declare_mlir_python_sources(ChiselSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/chisel"
  SOURCES
    __init__.py
    bind.py
    callbacks.py
    context.py
    ops.py
    executor.py
    exceptions.py
    op_handlers.py
    op_configs.py
    recorder.py
    report.py
    safety.py
    utils.py
    validators.py
)

add_mlir_python_modules(ChiselPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/chisel"
  INSTALL_PREFIX "python_packages/chisel"
  DECLARED_SOURCES ChiselSources
)
