include(AddMLIRPython)

set(TTD2MJIT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

declare_mlir_python_sources(TTD2MJitSources
    ROOT_DIR "${TTD2MJIT_ROOT_DIR}"
    SOURCES
        __init__.py
        api.py
        _src/__init__.py
        _src/ast.py
        _src/builder.py
        _src/config.py
        _src/errors.py
        _src/rewrite.py
        _src/tensor_layout.py
        _src/utils.py
        patterns/__init__.py
        patterns/eltwise_add_exp_to_kernel.py
        patterns/eltwise_exp_to_kernel.py
)

add_mlir_python_modules(TTD2MJitModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/d2m_jit"
    INSTALL_PREFIX "python_packages/d2m_jit"
    DECLARED_SOURCES TTD2MJitSources
)

add_dependencies(TTD2MJitModules TTMLIRPythonModules)

add_custom_target(d2m-jit COMMENT "d2m-jit" DEPENDS TTD2MJitModules)

if (TTMLIR_ENABLE_RUNTIME)
  # Both the C++ static library (TTMLIRRuntime) and the Python extension
  # (_ttmlir_runtime). builder.py imports `_ttmlir_runtime` directly; without
  # this dep a `cmake --build BUILD_DIR --target d2m-jit` would skip the .so
  # and to_host() would fail at runtime with "ttmlir runtime is not available
  # in this build". Mirrors what tools/ttnn-jit does.
  add_dependencies(d2m-jit TTMLIRRuntime _ttmlir_runtime)
endif()
