# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

################################################################################
# Python Model Runner Library
################################################################################

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
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
  SYSTEM PUBLIC
    ${pybind11_INCLUDE_DIRS}
)

# Find Python for linking
find_package(Python3 COMPONENTS Development REQUIRED)

target_include_directories(${PYTHON_RUNNER_LIBRARY_NAME}
  SYSTEM PUBLIC
    ${TTMETAL_INCLUDE_DIRS}
)

target_link_libraries(${PYTHON_RUNNER_LIBRARY_NAME}
  PUBLIC
    pybind11::embed
    Python3::Python
    TTRuntimeTypes
    TTRuntimeTTNNUtils
    TTRuntimeTTNNTypes
    TTRuntimeUtils
    TTRuntimeContext
    ${TTMETAL_LIBRARY_DIR}/libtt_metal.so
    ${TTMETAL_LIBRARY_DIR}/_ttnncpp.so
    ${TTMETAL_LIBRARY_DIR}/libtt_stl.so
)

add_dependencies(${PYTHON_RUNNER_LIBRARY_NAME} FBS_GENERATION)

set_target_properties(${PYTHON_RUNNER_LIBRARY_NAME} PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH FALSE
)

# Copy header to tt-alchemist include directory during build
add_custom_command(TARGET ${PYTHON_RUNNER_LIBRARY_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/python_runner.hpp
    ${TT_ALCHEMIST_ROOT_BINARY_DIR}/include/python_runner.hpp
  COMMENT "Copying python_runner.hpp to tt-alchemist include directory"
)

install(FILES $<TARGET_FILE:${PYTHON_RUNNER_LIBRARY_NAME}>
  DESTINATION lib
  OPTIONAL
)
