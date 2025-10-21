include(AddMLIRPython)

set(TTPYKERNEL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

declare_mlir_python_sources(TTPykernelSources
    ROOT_DIR "${TTPYKERNEL_ROOT_DIR}"
    SOURCES
        __init__.py
        api.py
        d2m_api.py
        _src/base_ast.py
        _src/kernel_ast.py
        _src/kernel_types.py
        _src/kernel_op.py
        _src/d2m_ast.py
        _src/stream.py
        _src/utils.py
)

add_mlir_python_modules(TTPykernelModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/pykernel"
    INSTALL_PREFIX "python_packages/pykernel"
    DECLARED_SOURCES TTPykernelSources
)

add_dependencies(TTPykernelModules TTMLIRPythonModules)

add_custom_target(pykernel COMMENT "pykernel" DEPENDS TTPykernelModules)

if (TTMLIR_ENABLE_RUNTIME)
  add_dependencies(pykernel TTMLIRRuntime)
endif()
