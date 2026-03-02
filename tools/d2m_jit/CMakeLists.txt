
option(USE_D2M_JIT_WHEEL "Develop with d2m-jit wheel" ON)

add_subdirectory(csrc)

set(D2MJIT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if (USE_D2M_JIT_WHEEL)
  set(JIT_WHEEL_RPATHS
    "$ORIGIN;$ORIGIN/../../ttnn/build/lib;$ORIGIN/../../../../../../build/lib;${TTMETAL_LIBRARY_DIR}"
    CACHE STRING "RPATHs for d2m-jit wheel"
  )

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION d2m_jit
    COMPONENT D2MJITWheel
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
      $<TARGET_FILE:_d2m_jit>
      $<TARGET_FILE:TTMLIRRuntime>
      $<TARGET_FILE:_ttmlir_runtime>
    DESTINATION d2m_jit
    COMPONENT D2MJITWheel
    EXCLUDE_FROM_ALL
  )

  # Build d2m-jit wheel with --find-links pointing to ttmlir wheel location
  set(TTMLIR_WHEEL_DIR "${CMAKE_SOURCE_DIR}/python/dist")
  add_custom_command(
    COMMAND rm -rf build d2m_jit.egg-info
    COMMAND python -m pip install build patchelf
    COMMAND TTMLIR_DEV_BUILD=ON python -m build --wheel --outdir build
    COMMAND python -m pip install --find-links ${TTMLIR_WHEEL_DIR} build/d2m_jit-*.whl --force-reinstall
    COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/build/.installed
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS _d2m_jit ttmlir-wheel
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build/.d2m-jit-installed
  )

  add_custom_target(d2m-jit-deps
    COMMENT "d2m-jit dependencies"
    DEPENDS _d2m_jit _ttmlir_runtime TTMLIRPythonModules
  )

  add_custom_target(d2m-jit ALL
    COMMENT "d2m-jit package"
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/build/.d2m-jit-installed d2m-jit-deps)

  set_target_properties(JITCPP PROPERTIES
    INSTALL_RPATH "${JIT_WHEEL_RPATHS}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  set_target_properties(_d2m_jit PROPERTIES
    INSTALL_RPATH "${JIT_WHEEL_RPATHS}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
else ()
  set_target_properties(JITCPP PROPERTIES
    INSTALL_RPATH "$ORIGIN;${TTMETAL_LIBRARY_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  set_target_properties(_d2m_jit PROPERTIES
    INSTALL_RPATH "$ORIGIN;$<TARGET_FILE_DIR:JITCPP>"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  include(AddMLIRPython)
  declare_mlir_python_sources(D2MJITSources
    ROOT_DIR "${D2MJIT_ROOT_DIR}"
    SOURCES
        api.py
        __init__.py
        _src/ast_compiler.py
        _src/mlir_generator.py
  )

  add_mlir_python_modules(D2MJITModules
      ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/d2m_jit"
      INSTALL_PREFIX "python_packages/d2m_jit"
      DECLARED_SOURCES D2MJITSources
  )
  add_custom_target(d2m-jit ALL
    COMMENT "d2m-jit package"
    DEPENDS D2MJITModules _d2m_jit _ttmlir_runtime TTMLIRPythonModules)
endif()
