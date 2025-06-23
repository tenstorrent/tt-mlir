# Configure envs
#
if(NOT DEFINED ENV{TT_MLIR_HOME})
  message(FATAL_ERROR "TT_MLIR_HOME environment variable is not set")
endif()
if(NOT DEFINED ENV{TT_ALCHEMIST_LIB_DIR})
  message(FATAL_ERROR "TT_ALCHEMIST_LIB_DIR environment variable is not set")
endif()

# Set TT_ALCHEMIST_HOME
#
set(TT_ALCHEMIST_HOME $ENV{TT_MLIR_HOME}/tools/tt-alchemist)

# Define target names
#
set(LIBRARY_NAME tt-alchemist-lib)

# Sources
#
set(LIB_SOURCES
  lib/tt_alchemist.cpp
)

# Add library
#
add_library(${LIBRARY_NAME}
  SHARED
  EXCLUDE_FROM_ALL
  ${LIB_SOURCES}
)

# Add dependencies
#
add_dependencies(${LIBRARY_NAME} TTMLIRCompiler)

# Include dirs
target_include_directories(${LIBRARY_NAME}
  PUBLIC
    ${TT_ALCHEMIST_HOME}/include
    ${TT_MLIR_HOME}/include
)

target_link_directories(${LIBRARY_NAME}
  PUBLIC
    $ENV{TT_MLIR_HOME}/lib
)

# Link libraries
target_link_libraries(${LIBRARY_NAME}
  PRIVATE
    # TTMLIRCompiler
    TTMLIRCompilerStatic
)

# Disable RTTI and set visibility
target_compile_options(${LIBRARY_NAME}
  PRIVATE
    -fno-rtti
    -fvisibility=hidden
)

# Define export macro
target_compile_definitions(${LIBRARY_NAME}
  PRIVATE
    TT_ALCHEMIST_EXPORT=__attribute__\(\(visibility\(\"default\"\)\)\)
)

# Install the library
install(TARGETS ${LIBRARY_NAME}
  EXPORT ${LIBRARY_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)


# ======================================


# Python wheel building and installation
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Copy Python files to build directory
set(PYTHON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../python)
set(PYTHON_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/python)

# Create Python build directory
add_custom_command(
  OUTPUT ${PYTHON_BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_BUILD_DIR}
  COMMENT "Creating Python build directory"
)

# Find all Python files in the source directory
file(GLOB_RECURSE PYTHON_SOURCE_FILES
  ${PYTHON_SRC_DIR}/*.py
  ${PYTHON_SRC_DIR}/pyproject.toml
  ${PYTHON_SRC_DIR}/setup.py
)

# Copy Python files
add_custom_command(
  OUTPUT ${PYTHON_BUILD_DIR}/pyproject.toml
  DEPENDS ${PYTHON_SOURCE_FILES} ${PYTHON_BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PYTHON_SRC_DIR} ${PYTHON_BUILD_DIR}
  COMMENT "Copying Python files to build directory"
)

# Build wheel
add_custom_target(build-wheel-tt-alchemist
  DEPENDS ${PYTHON_BUILD_DIR}/pyproject.toml ${LIBRARY_NAME}
  COMMAND ${CMAKE_COMMAND} -E env TT_ALCHEMIST_LIB_DIR=$<TARGET_FILE:${LIBRARY_NAME}>
          ${Python3_EXECUTABLE} -m pip wheel --no-deps -w ${CMAKE_CURRENT_BINARY_DIR}/dist ${PYTHON_BUILD_DIR}
  WORKING_DIRECTORY ${PYTHON_BUILD_DIR}
  COMMENT "Building tt-alchemist wheel"
)

# Install tt-alchemist wheel
add_custom_target(tt-alchemist
  DEPENDS build-wheel-tt-alchemist
  COMMAND ${Python3_EXECUTABLE} -m pip install --force-reinstall ${CMAKE_CURRENT_BINARY_DIR}/dist/*.whl
  COMMENT "Installing tt-alchemist wheel"
)
