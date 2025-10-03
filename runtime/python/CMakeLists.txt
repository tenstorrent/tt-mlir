# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

find_package(nanobind CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(TTMLIR_RUNTIME_PYTHON_SRCS
  __init__.cpp
  runtime/runtime.cpp
  binary/binary.cpp
)

if (TTMLIR_ENABLE_RUNTIME_TESTS)
  list(APPEND TTMLIR_RUNTIME_PYTHON_SRCS runtime/test.cpp)
endif()

nanobind_add_module(
  _ttmlir_runtime
  ${TTMLIR_RUNTIME_PYTHON_SRCS}
)

set_target_properties(_ttmlir_runtime PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "$ORIGIN;$<TARGET_FILE_DIR:TTMLIRRuntime>"
  BUILD_WITH_INSTALL_RPATH FALSE
)

target_include_directories(_ttmlir_runtime
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)

target_link_libraries(_ttmlir_runtime
  PRIVATE
  TTMLIRRuntime
)
add_dependencies(_ttmlir_runtime TTMLIRRuntime)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(nanobind-static PRIVATE
      -Wno-cast-qual
      -Wno-zero-length-array
      -Wno-nested-anon-types
      -Wno-c++98-compat-extra-semi
      -Wno-covered-switch-default
  )
endif()

if (TTMLIR_ENABLE_RUNTIME)
  # Isolate utils.cpp to a separate TU to avoid pulling in pybinds.
  add_library(_mlir_runtime_utils OBJECT runtime/utils.cpp)
  target_include_directories(_mlir_runtime_utils
    SYSTEM PRIVATE ${pybind11_INCLUDE_DIRS}
    PRIVATE
      ${PROJECT_SOURCE_DIR}/runtime/include
      ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
      ${PROJECT_SOURCE_DIR}/runtime/python/include
  )
  target_link_libraries(_mlir_runtime_utils
    PRIVATE
    TTRuntimeTTNNUtils
    nanobind-static
  )
  target_sources(_ttmlir_runtime PRIVATE $<TARGET_OBJECTS:_mlir_runtime_utils>)
  set_property(TARGET _mlir_runtime_utils PROPERTY CXX_STANDARD 20)
endif()

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options("nanobind-static" PRIVATE
      -Wno-cast-qual
      -Wno-zero-length-array
      -Wno-nested-anon-types
      -Wno-c++98-compat-extra-semi
      -Wno-covered-switch-default
  )
endif()

# Copy Python scripts to build directory for C++ runtime to use
add_custom_command(TARGET _ttmlir_runtime POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts
    ${CMAKE_CURRENT_BINARY_DIR}/python/scripts
  COMMENT "Copying Python runtime scripts to build directory"
)

# Install the Python runtime module
install(TARGETS _ttmlir_runtime
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT SharedLib
)

# Install Python runtime scripts to build directory (for embedded Python to find)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/../runtime/python
  COMPONENT SharedLib
  FILES_MATCHING PATTERN "*.py"
)
