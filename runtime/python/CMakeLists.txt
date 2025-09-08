# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

find_package(nanobind CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(TTMLIR_RUNTIME_PYTHON_SRCS
  __init__.cpp
  runtime/runtime.cpp
  runtime/utils.cpp
  binary/binary.cpp
)

set_source_files_properties(runtime/utils.cpp PROPERTIES INCLUDE_DIRECTORIES "${pybind11_INCLUDE_DIRS}")

if (TTMLIR_ENABLE_RUNTIME_TESTS)
  list(APPEND TTMLIR_RUNTIME_PYTHON_SRCS runtime/test.cpp)
endif()

nanobind_add_module(
  _ttmlir_runtime
  ${TTMLIR_RUNTIME_PYTHON_SRCS}
)

set_property(TARGET _ttmlir_runtime PROPERTY CXX_STANDARD 20)

set_target_properties(_ttmlir_runtime PROPERTIES
  INSTALL_RPATH "$ORIGIN;$<TARGET_FILE_DIR:TTMLIRRuntime>"
  BUILD_RPATH "$<TARGET_FILE_DIR:TTMLIRRuntime>"
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
  TTRuntimeTTNNUtils
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

# if (TTMLIR_ENABLE_PYKERNEL)
#   set(RT_PKG "${TTMLIR_PYTHON_PACKAGES_DIR}/tt_runtime")
#   file(MAKE_DIRECTORY "${RT_PKG}")
#   file(WRITE "${RT_PKG}/__init__.py" "from ._ttmlir_runtime import runtime as runtime, binary as binary\n")

#   # Create symlinks to runtime pybinds and its dependencies.
#   add_custom_command(TARGET _ttmlir_runtime POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:_ttmlir_runtime> "${RT_PKG}/$<TARGET_FILE_NAME:_ttmlir_runtime>"
#     COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:TTMLIRRuntime> "${RT_PKG}/$<TARGET_FILE_NAME:TTMLIRRuntime>"
#     COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:TTMETAL_LIBRARY> "${RT_PKG}/$<TARGET_FILE_NAME:TTMETAL_LIBRARY>"
#     COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:DEVICE_LIBRARY> "${RT_PKG}/$<TARGET_FILE_NAME:DEVICE_LIBRARY>"
#     COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:TTNN_LIBRARY> "${RT_PKG}/$<TARGET_FILE_NAME:TTNN_LIBRARY>"
#     COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:TT_STL_LIBRARY> "${RT_PKG}/$<TARGET_FILE_NAME:TT_STL_LIBRARY>"
#   )
# endif()
