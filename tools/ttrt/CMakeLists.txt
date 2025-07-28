include(AddMLIRPython)

declare_mlir_python_sources(TTRT)

declare_mlir_python_sources(TTRTSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT TTRT
  SOURCES
    __init__.py
    api.py
    callback.py
    check.py
    perf.py
    query.py
    read.py
    run.py
    stats.py
    util.py
    library_tweaks.py
)

add_mlir_python_modules(TTRTModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/ttrt"
    INSTALL_PREFIX "python_packages/ttrt"
    DECLARED_SOURCES TTRTSources
)
