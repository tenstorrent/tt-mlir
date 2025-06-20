include(AddMLIRPython)

declare_mlir_python_sources(TTMLIRPythonOpByOpInfra)

declare_mlir_python_sources(TTMLIRPythonOpByOpInfraSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/op_by_op_infra"
  ADD_TO_PARENT TTMLIRPythonOpByOpInfra
  SOURCES
    __init__.py
    execution_result.py
    mlir_module_executor.py
    mlir_module_splitter.py
    pydantic_models.py
    utils.py
    workflow_internal.py
    workflow.py
)

add_mlir_python_modules(TTMLIRPythonOpByOpInfraModules
    ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/op_by_op_infra"
    INSTALL_PREFIX "python_packages/op_by_op_infra"
    DECLARED_SOURCES TTMLIRPythonOpByOpInfraSources
)
