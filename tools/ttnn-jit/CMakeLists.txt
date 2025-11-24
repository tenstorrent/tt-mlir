
option(USE_TTNN_JIT_WHEEL "Develop with ttnn-jit wheel" ON)

add_subdirectory(csrc)

set(TTNNJIT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if (USE_TTNN_JIT_WHEEL)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ttnn_jit
    COMPONENT TTNNJITWheel
    EXCLUDE_FROM_ALL
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "build" EXCLUDE
    PATTERN "dist" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.egg-info" EXCLUDE
    PATTERN "setup.py" EXCLUDE
  )
  install(FILES
      $<TARGET_FILE:JITCPP>
      $<TARGET_FILE:_ttnn_jit>
      $<TARGET_FILE:TTMLIRRuntime>
      $<TARGET_FILE:_ttmlir_runtime>
    DESTINATION ttnn_jit/runtime
    COMPONENT TTNNJITWheel
    EXCLUDE_FROM_ALL
  )
  install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/ttmlir/
    DESTINATION ttnn_jit/runtime/ttmlir
    COMPONENT TTNNJITWheel
    EXCLUDE_FROM_ALL
  )

  add_custom_command(
    COMMAND rm -rf build ttnn_jit.egg-info
    COMMAND python -m pip install build patchelf
    COMMAND TTMLIR_DEV_BUILD=ON python -m build --wheel --outdir build
    COMMAND python -m pip install build/ttnn_jit-*.whl --force-reinstall
    COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/build/.installed
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS _ttnn_jit
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build/.ttnn-jit-installed
  )

  add_custom_target(ttnn-jit ALL
    COMMENT "ttnn-jit package"
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/build/.ttnn-jit-installed _ttnn_jit _ttmlir_runtime TTMLIRPythonModules)
else ()
  include(AddMLIRPython)
  declare_mlir_python_sources(TTNNJITSources
    ROOT_DIR "${TTNNJIT_ROOT_DIR}"
    SOURCES
        api.py
        __init__.py
        _src/__init__.py
        _src/jit.py
        _src/ttir_ast.py
        _src/utils.py
        _src/dispatch_op.py
        _src/ir_generator.py
        _src/graph_trace_compiler.py
        _src/tensor_translator.py
        _src/supported_ops.py
  )

  add_mlir_python_modules(TTNNJITModules
      ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/ttnn_jit"
      INSTALL_PREFIX "python_packages/ttnn_jit"
      DECLARED_SOURCES TTNNJITSources
  )
  add_custom_target(ttnn-jit ALL
    COMMENT "ttnn-jit package"
    DEPENDS TTNNJITModules _ttnn_jit _ttmlir_runtime TTMLIRPythonModules)
endif()
