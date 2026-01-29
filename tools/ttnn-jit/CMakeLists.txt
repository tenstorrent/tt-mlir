
option(USE_TTNN_JIT_WHEEL "Develop with ttnn-jit wheel" ON)

add_subdirectory(csrc)

set(TTNNJIT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if (USE_TTNN_JIT_WHEEL)
  set(JIT_WHEEL_RPATHS
    "$ORIGIN;$ORIGIN/../../ttnn/build/lib;$ORIGIN/../../../../../../build/lib;${TTMETAL_LIBRARY_DIR}"
    CACHE STRING "RPATHs for ttnn-jit wheel"
  )

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
    DESTINATION ttnn_jit/
    COMPONENT TTNNJITWheel
    EXCLUDE_FROM_ALL
  )

  # Build ttnn-jit wheel with --find-links pointing to ttmlir wheel location
  set(TTMLIR_WHEEL_DIR "${CMAKE_SOURCE_DIR}/python/dist")
  add_custom_command(
    COMMAND rm -rf build ttnn_jit.egg-info
    COMMAND python -m pip install build patchelf
    COMMAND TTMLIR_DEV_BUILD=ON python -m build --wheel --outdir build
    COMMAND python -m pip install --find-links ${TTMLIR_WHEEL_DIR} build/ttnn_jit-*.whl --force-reinstall
    COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/build/.installed
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS _ttnn_jit ttmlir-wheel
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build/.ttnn-jit-installed
  )

  add_custom_target(ttnn-jit-deps
    COMMENT "ttnn-jit dependencies"
    DEPENDS _ttnn_jit _ttmlir_runtime TTMLIRPythonModules
  )

  add_custom_target(ttnn-jit ALL
    COMMENT "ttnn-jit package"
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/build/.ttnn-jit-installed ttnn-jit-deps)

  set_target_properties(JITCPP PROPERTIES
    INSTALL_RPATH "${JIT_WHEEL_RPATHS}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  set_target_properties(_ttnn_jit PROPERTIES
    INSTALL_RPATH "${JIT_WHEEL_RPATHS}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
else ()
  set_target_properties(JITCPP PROPERTIES
    INSTALL_RPATH "$ORIGIN;${TTMETAL_LIBRARY_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  set_target_properties(_ttnn_jit PROPERTIES
    INSTALL_RPATH "$ORIGIN;$<TARGET_FILE_DIR:JITCPP>"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  include(AddMLIRPython)
  declare_mlir_python_sources(TTNNJITSources
    ROOT_DIR "${TTNNJIT_ROOT_DIR}"
    SOURCES
        api.py
        __init__.py
        _src/__init__.py
        _src/jit.py
        _src/utils.py
        _src/dispatch_op.py
        _src/ir_generator.py
        _src/tensor_translator.py
        _src/supported_ops.py
        _src/conversions.py
        _src/jit_functions.py
        _src/tracing_compiler.py
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
