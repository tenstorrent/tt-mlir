# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

################################################################################
# Python Model Runner Library
################################################################################

set(PYTHON_RUNNER_LIBRARY_NAME tt-alchemist-python-runner)

# We embed CPython (Py_Initialize, etc.), so request the "embed" development
# target. nanobind additionally expects FindPython-provided variables/targets.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)

# Ensure nanobind_build_library() is available (no-op if already found).
find_package(nanobind CONFIG REQUIRED)

# Use the same nanobind that tt-metal uses for ABI compatibility with the ttnn
# Python module.  NB_INTERNALS_VERSION must match; otherwise each library gets
# its own nb_internals and cross-library type lookups (e.g. ttnn::Tensor) fail
# at runtime with a segfault in nb_type_c2p.
#
# The pip-installed nanobind may be a different version (e.g. 2.11.0 / ABI v18)
# than what tt-metal vendors via CPM (e.g. 2.10.2 / ABI v17).  We build a
# separate nanobind-static library from tt-metal's source to guarantee the same
# ABI as _ttnn.so.
if(DEFINED ENV{CPM_SOURCE_CACHE})
  set(_CPM_CACHE $ENV{CPM_SOURCE_CACHE})
else()
  set(_CPM_CACHE "${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache")
endif()
# CPM uses content-hashed directory names; glob for the nanobind source dir.
file(GLOB _NB_DIRS "${_CPM_CACHE}/nanobind/*")
list(FILTER _NB_DIRS EXCLUDE REGEX "cmake\\.lock$")
if(_NB_DIRS)
  list(GET _NB_DIRS 0 _NB_TTMETAL_DIR)
  # Temporarily swap NB_DIR (CACHE INTERNAL set by nanobind-config.cmake) so
  # nanobind_build_library compiles from tt-metal's nanobind source tree.
  set(_ORIG_NB_DIR "${NB_DIR}")
  set(NB_DIR "${_NB_TTMETAL_DIR}" CACHE INTERNAL "" FORCE)
  message(STATUS "PythonModelRunner: building nanobind-static-compat from ${NB_DIR}")
  nanobind_build_library(nanobind-static-compat)
  set(NB_DIR "${_ORIG_NB_DIR}" CACHE INTERNAL "" FORCE)
  set(_NB_TARGET nanobind-static-compat)
else()
  message(WARNING "PythonModelRunner: tt-metal nanobind not found in CPM cache, "
                  "falling back to default nanobind-static (ABI may mismatch)")
  nanobind_build_library(nanobind-static)
  set(_NB_TARGET nanobind-static)
endif()

add_library(${PYTHON_RUNNER_LIBRARY_NAME}
  SHARED
  python_runner.cpp
)

# Set C++20 standard (required by tt-metal headers)
set_property(TARGET ${PYTHON_RUNNER_LIBRARY_NAME} PROPERTY CXX_STANDARD 20)

target_include_directories(${PYTHON_RUNNER_LIBRARY_NAME}
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Treat TTNN/tt-metal headers as system headers to avoid warnings promoted to
# errors by this repo's -Werror settings.
target_include_directories(${PYTHON_RUNNER_LIBRARY_NAME}
  SYSTEM PRIVATE
    ${TTMETAL_INCLUDE_DIRS}
)

add_dependencies(${PYTHON_RUNNER_LIBRARY_NAME} tt-metal)

target_link_libraries(${PYTHON_RUNNER_LIBRARY_NAME}
  PUBLIC
    TTMLIRRuntime
  PRIVATE
    ${_NB_TARGET}
    Python::Python
    ${TTMETAL_LIBRARY_PATH}
    "LINKER:${TTNN_LIBRARY_PATH}"
    ${DEVICE_LIBRARY_PATH}
    ${TT_STL_LIBRARY_PATH}
    ${FMT_LIBRARY_PATH}
)

set_target_properties(${PYTHON_RUNNER_LIBRARY_NAME} PROPERTIES
  BUILD_RPATH "$ORIGIN;$<TARGET_FILE_DIR:TTMLIRRuntime>"
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH FALSE
)

install(FILES $<TARGET_FILE:${PYTHON_RUNNER_LIBRARY_NAME}>
  DESTINATION lib
  COMPONENT SharedLib
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python_runner.hpp
  DESTINATION include/tools/tt-alchemist/python_runner
  COMPONENT SharedLib
)
