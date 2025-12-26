# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

################################################################################
# Python Model Runner Library
################################################################################

# Issue(vkovacevic): https://github.com/tenstorrent/tt-mlir/issues/6443
# Python embedding via pybind11::embed is incompatible with nanobind.
# When tt-metal is built with EXPERIMENTAL_NANOBIND_BINDINGS=ON (default),
# the type casters registered by nanobind cannot be used by pybind11's py::cast.
# Skip building this library when nanobind is enabled.
message(STATUS "Skipping python_runner: incompatible with nanobind (EXPERIMENTAL_NANOBIND_BINDINGS=ON)")
return()

set(PYTHON_RUNNER_LIBRARY_NAME tt-alchemist-python-runner)

# Find pybind11 for Python embedding support (used by python runner)
find_package(pybind11 CONFIG REQUIRED)

add_library(${PYTHON_RUNNER_LIBRARY_NAME}
  SHARED
  EXCLUDE_FROM_ALL
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
    ${CMAKE_CURRENT_SOURCE_DIR}  # For python_runner.hpp
  SYSTEM PUBLIC
    ${pybind11_INCLUDE_DIRS}
    ${TTMETAL_INCLUDE_DIRS}
)

# Find Python for linking
find_package(Python3 COMPONENTS Development REQUIRED)

target_link_libraries(${PYTHON_RUNNER_LIBRARY_NAME}
  PUBLIC
    pybind11::embed
    Python3::Python
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
  OPTIONAL
)
