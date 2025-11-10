# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Find pybind11 and Python
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Define the module name
set(MODULE_NAME example_module)

# Create a shared library instead of using pybind11_add_module
# This allows us to export the entrypoint symbol for dlopen
add_library(${MODULE_NAME} SHARED example_module.cpp)

# Link against Python and pybind11
target_link_libraries(${MODULE_NAME}
    PRIVATE
    pybind11::embed  # Use embed target for embedded interpreter
    Python3::Python
)

# Include pybind11 headers
target_include_directories(${MODULE_NAME}
    PRIVATE
    ${pybind11_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Set C++ standard
set_target_properties(${MODULE_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Set the output name to match Python module naming convention
set_target_properties(${MODULE_NAME} PROPERTIES
    PREFIX ""
    SUFFIX "${Python3_SOABI}.so"
)

# Export all symbols (needed for dlopen to find entrypoint)
set_target_properties(${MODULE_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET default
)

# Set RPATH for runtime dependencies
set_target_properties(${MODULE_NAME} PROPERTIES
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH "$ORIGIN"
)

# Optional: Link against other libraries if needed
# Uncomment and modify if you need to link against MLIR or other libraries
# target_link_libraries(${MODULE_NAME}
#   PRIVATE
#     TTMLIRCompilerStatic
#     MLIRSupport
# )

# Optional: Add include directories if needed
# target_include_directories(${MODULE_NAME}
#   PRIVATE
#     ${PROJECT_SOURCE_DIR}/include
# )

# Define the destination directory for the .so file
set(TEMPLATE_DEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../templates/python/local)

# Add a custom command to copy the .so file to the templates directory
add_custom_command(TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:${MODULE_NAME}>
        ${TEMPLATE_DEST_DIR}/$<TARGET_FILE_NAME:${MODULE_NAME}>
    COMMENT "Copying ${MODULE_NAME}.so to templates/python/local"
)

# Also copy to the Python build directory if building the wheel
set(PYTHON_BUILD_TEMPLATE_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../python/tt_alchemist/templates/python/local)
add_custom_command(TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_BUILD_TEMPLATE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:${MODULE_NAME}>
        ${PYTHON_BUILD_TEMPLATE_DIR}/$<TARGET_FILE_NAME:${MODULE_NAME}>
    COMMENT "Copying ${MODULE_NAME}.so to Python build templates directory"
)

