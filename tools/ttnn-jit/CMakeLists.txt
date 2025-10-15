include(AddMLIRPython)

add_subdirectory(csrc)

set(TTNNJIT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

declare_mlir_python_sources(TTNNJITSources
    ROOT_DIR "${TTNNJIT_ROOT_DIR}"
    SOURCES
        api.py
        __init__.py
        _src/__init__.py
        _src/ttir_ast.py
        _src/utils.py
        _src/dispatch_op.py
)

add_mlir_python_modules(TTNNJITModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/ttnn_jit"
    INSTALL_PREFIX "python_packages/ttnn_jit"
    DECLARED_SOURCES TTNNJITSources
)
