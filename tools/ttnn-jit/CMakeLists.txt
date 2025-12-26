
option(TTNN_JIT_DEV "ttnn-jit dev flow with editable wheel install" OFF)

add_subdirectory(csrc)

set(TTNNJIT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

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
  COMMAND python -m pip install build
  COMMAND TTMLIR_DEV_BUILD=ON python -m build --wheel --outdir build
  COMMAND python -m pip install build/ttnn_jit-*.whl --force-reinstall
  COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/build/.installed
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS _ttnn_jit
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build/.ttnn-jit-installed
)

add_custom_command(
  COMMAND python -m pip install --upgrade "setuptools>=64.0" build
  COMMAND python -m pip install -e . --no-build-isolation
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS _ttnn_jit
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build/.ttnn-jit-installed-editable
)

add_custom_target(ttnn-jit-deps
  COMMENT "ttnn-jit dependencies"
  DEPENDS _ttnn_jit _ttmlir_runtime TTMLIRPythonModules
)

if (TTNN_JIT_DEV)
  add_custom_target(ttnn-jit ALL
    COMMENT "ttnn-jit dev package"
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/build/.ttnn-jit-installed-editable ttnn-jit-deps)
else()
  add_custom_target(ttnn-jit ALL
    COMMENT "ttnn-jit package"
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/build/.ttnn-jit-installed ttnn-jit-deps)
endif()

set_target_properties(JITCPP PROPERTIES
  INSTALL_RPATH "${JIT_WHEEL_RPATHS}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
set_target_properties(_ttnn_jit PROPERTIES
  INSTALL_RPATH "${JIT_WHEEL_RPATHS}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
