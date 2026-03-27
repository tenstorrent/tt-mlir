# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

################################################################################
# Python Model Runner Library
################################################################################

set(PYTHON_RUNNER_LIBRARY_NAME tt-alchemist-python-runner)

include(GNUInstallDirs)

# We embed CPython (Py_Initialize, etc.), so request the "embed" development
# target. nanobind additionally expects FindPython-provided variables/targets.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)

find_package(nanobind CONFIG REQUIRED)
nanobind_build_library(nanobind-static-python-runner)

# Suppress warnings from nanobind source that are promoted to errors by -Werror.
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(nanobind-static-python-runner PRIVATE
      -Wno-cast-qual
      -Wno-zero-length-array
      -Wno-nested-anon-types
      -Wno-c++98-compat-extra-semi
      -Wno-covered-switch-default
  )
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
    nanobind-static-python-runner
  PRIVATE
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
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT SharedLib
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python_runner.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tools/tt-alchemist/python_runner
  COMPONENT SharedLib
)
