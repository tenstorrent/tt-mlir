# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Tests for PythonModelRunner API

# Find Python for linking (needed for pybind11)
find_package(Python3 COMPONENTS Development REQUIRED)

# Simple test (no device/tensor required)
add_executable(test_python_runner_simple
  EXCLUDE_FROM_ALL
  test_python_runner_simple.cpp
)

set_property(TARGET test_python_runner_simple PROPERTY CXX_STANDARD 20)

target_compile_options(test_python_runner_simple
  PRIVATE
    -Wno-deprecated-declarations
    -Wno-gnu-zero-variadic-macro-arguments
)

target_include_directories(test_python_runner_simple
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/python_runner
)

target_link_libraries(test_python_runner_simple
  PRIVATE
    tt-alchemist-python-runner
)

# Pass test directory path to the executable
target_compile_definitions(test_python_runner_simple
  PRIVATE
    TEST_DIR_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
)

# Set RPATH to find all required libraries at runtime
set_target_properties(test_python_runner_simple PROPERTIES
  BUILD_WITH_INSTALL_RPATH FALSE
  BUILD_RPATH "${CMAKE_BINARY_DIR}/tools/tt-alchemist/csrc/python_runner:${TTMETAL_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${TTMETAL_LIBRARY_DIR}"
)

install(TARGETS test_python_runner_simple
  RUNTIME DESTINATION tools/tt-alchemist/test
  OPTIONAL
)

# Full test with tensors (requires device)
add_executable(test_python_runner
  EXCLUDE_FROM_ALL
  test_python_runner.cpp
)

set_property(TARGET test_python_runner PROPERTY CXX_STANDARD 20)


target_include_directories(test_python_runner
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/python_runner
)

target_include_directories(test_python_runner
  SYSTEM PRIVATE
    ${TTMETAL_INCLUDE_DIRS}
)

target_link_libraries(test_python_runner
  PRIVATE
    tt-alchemist-python-runner
    ${TTMETAL_LIBRARY_PATH}
    "LINKER:${TTNN_LIBRARY_PATH}"
    ${TT_STL_LIBRARY_PATH}
    ${FMT_LIBRARY_PATH}
)

# Path to tt-metal source directory
set(TTMETAL_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal")

# Pass paths to test executable so it can set up Python environment
target_compile_definitions(test_python_runner
  PRIVATE
    TT_METAL_HOME_PATH="${TTMETAL_SOURCE_DIR}"
    TTNN_PYTHON_PATH="${TTMETAL_SOURCE_DIR}/ttnn"
    TTMETAL_BUILD_LIB_PATH="${TTMETAL_LIBRARY_DIR}"
    TEST_DIR_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
)

# Set RPATH to find all required libraries at runtime
set_target_properties(test_python_runner PROPERTIES
  BUILD_WITH_INSTALL_RPATH FALSE
  BUILD_RPATH "${CMAKE_BINARY_DIR}/tools/tt-alchemist/csrc/python_runner:${TTMETAL_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${TTMETAL_LIBRARY_DIR}"
)

install(TARGETS test_python_runner
  RUNTIME DESTINATION tools/tt-alchemist/test
  OPTIONAL
)

# Install Python test modules needed by the C++ tests
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/simple_test_model.py
  ${CMAKE_CURRENT_SOURCE_DIR}/test_model.py
  DESTINATION tools/tt-alchemist/test
  OPTIONAL
)
