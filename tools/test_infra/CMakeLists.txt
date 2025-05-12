include(AddMLIRPython)

declare_mlir_python_sources(TTMLIRPythonTestInfra)

declare_mlir_python_sources(TTMLIRPythonTestInfraSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT TTMLIRPythonTestInfra
  SOURCES
    ccl_golden.py
    test_optimizer_overrides.py
    test_utils.py
    ttir_builder.py
)

add_mlir_python_modules(TTMLIRPythonTestInfraModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/test_infra"
    INSTALL_PREFIX "python_packages/test_infra"
    DECLARED_SOURCES TTMLIRPythonTestInfraSources
)
