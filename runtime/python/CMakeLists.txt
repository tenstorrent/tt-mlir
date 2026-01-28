# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

find_package(Python3 REQUIRED COMPONENTS Interpreter Development Development.Module)
if (NOT Python3_LIBRARIES)
  message(FATAL_ERROR "python libraries not found")
endif()
list(APPEND CMAKE_PREFIX_PATH "${Python3_SITEARCH}" "${Python3_SITELIB}")
if (NOT TARGET Python::Module AND TARGET Python3::Module)
  add_library(Python::Module ALIAS Python3::Module)
endif()
if (NOT DEFINED Python_EXECUTABLE AND DEFINED Python3_EXECUTABLE)
  set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "" FORCE)
endif()
if (NOT DEFINED Python_INCLUDE_DIRS AND DEFINED Python3_INCLUDE_DIRS)
  set(Python_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}" CACHE PATH "" FORCE)
endif()
if (NOT DEFINED Python_VERSION AND DEFINED Python3_VERSION)
  set(Python_VERSION "${Python3_VERSION}" CACHE PATH "" FORCE)
endif()
find_package(nanobind CONFIG REQUIRED)

set(TTMLIR_RUNTIME_PYTHON_SRCS
  __init__.cpp
  runtime/runtime.cpp
  binary/binary.cpp
)

if (TTMLIR_ENABLE_RUNTIME_TESTS)
  list(APPEND TTMLIR_RUNTIME_PYTHON_SRCS runtime/test.cpp)
endif()

# Add macOS stubs when runtime is disabled but pykernel is enabled
if (APPLE AND NOT TTMLIR_ENABLE_RUNTIME AND TTMLIR_ENABLE_PYKERNEL)
  list(APPEND TTMLIR_RUNTIME_PYTHON_SRCS runtime/stubs_macos.cpp)
endif()

nanobind_add_module(
  _ttmlir_runtime
  ${TTMLIR_RUNTIME_PYTHON_SRCS}
)

if (TTMLIR_ENABLE_TTNN_JIT)
  # ttnn-jit wheel requires custom rpaths when installed into a tt-metal dev venv
  set_target_properties(_ttmlir_runtime PROPERTIES
    INSTALL_RPATH "${JIT_WHEEL_RPATHS}"
    BUILD_RPATH "$ORIGIN;$<TARGET_FILE_DIR:TTMLIRRuntime>"
    BUILD_WITH_INSTALL_RPATH FALSE
  )
else()
  set_target_properties(_ttmlir_runtime PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH "$ORIGIN;$<TARGET_FILE_DIR:TTMLIRRuntime>"
    BUILD_WITH_INSTALL_RPATH FALSE
  )
endif()

target_include_directories(_ttmlir_runtime
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
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
  add_library(_mlir_runtime_utils OBJECT runtime/utils.cpp)
  target_include_directories(_mlir_runtime_utils
    PRIVATE
      ${PROJECT_SOURCE_DIR}/runtime/include
      ${PROJECT_BINARY_DIR}/runtime/include
      ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
      ${PROJECT_SOURCE_DIR}/runtime/python/include
  )
  target_include_directories(_mlir_runtime_utils SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
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
