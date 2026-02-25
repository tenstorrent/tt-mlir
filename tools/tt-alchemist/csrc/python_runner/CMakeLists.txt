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

# Use the same nanobind that tt-metal uses for ABI compatibility with the ttnn
# Python module.  NB_INTERNALS_VERSION must match; otherwise each library gets
# its own nb_internals and cross-library type lookups (e.g. ttnn::Tensor) fail.
if(DEFINED ENV{CPM_SOURCE_CACHE})
  set(_CPM_CACHE $ENV{CPM_SOURCE_CACHE})
else()
  set(_CPM_CACHE "${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache")
endif()
file(GLOB _NANOBIND_CMAKE_DIR "${_CPM_CACHE}/nanobind/*/cmake")
if(_NANOBIND_CMAKE_DIR)
  list(GET _NANOBIND_CMAKE_DIR 0 _NANOBIND_CMAKE_DIR)
  set(nanobind_DIR "${_NANOBIND_CMAKE_DIR}")
  message(STATUS "PythonModelRunner: using tt-metal's nanobind for ABI compat: ${nanobind_DIR}")
endif()

find_package(nanobind CONFIG REQUIRED)
nanobind_build_library(nanobind-static)

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
    nanobind-static
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
