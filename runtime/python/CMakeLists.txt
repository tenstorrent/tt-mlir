# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

find_package(nanobind CONFIG REQUIRED)

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

set_target_properties(_ttmlir_runtime PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(_ttmlir_runtime PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

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

install(TARGETS _ttmlir_runtime
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT SharedLib
)
