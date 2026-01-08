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

find_package(nanobind CONFIG REQUIRED)
nanobind_build_library(nanobind-static)

add_library(${PYTHON_RUNNER_LIBRARY_NAME}
  SHARED
  python_runner.cpp
)

# Set C++20 standard (required by tt-metal headers)
set_property(TARGET ${PYTHON_RUNNER_LIBRARY_NAME} PROPERTY CXX_STANDARD 20)

# Suppress warnings from tt-metal headers that leak through macro expansion
# and template instantiation (SYSTEM includes don't fully prevent these)
target_compile_options(${PYTHON_RUNNER_LIBRARY_NAME}
  PRIVATE
    -Wno-deprecated-declarations
    -Wno-gnu-zero-variadic-macro-arguments
)

target_include_directories(${PYTHON_RUNNER_LIBRARY_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/tools/tt-alchemist/python_runner>
  SYSTEM PUBLIC
    ${TTMETAL_INCLUDE_DIRS}
)

target_link_libraries(${PYTHON_RUNNER_LIBRARY_NAME}
  PUBLIC
    nanobind-static
    Python::Python
    ${TTMETAL_LIBRARY_DIR}/libtt_metal.so
    ${TTMETAL_LIBRARY_DIR}/_ttnncpp.so
    ${TTMETAL_LIBRARY_DIR}/libtt_stl.so
    ${TTMETAL_LIBRARY_DIR}/libfmt$<IF:$<CONFIG:Debug>,d,>.so
)

set_target_properties(${PYTHON_RUNNER_LIBRARY_NAME} PROPERTIES
  BUILD_RPATH "$ORIGIN"
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
