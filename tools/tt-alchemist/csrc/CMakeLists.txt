# Set tt-alchemist root dirs
#
set(TT_ALCHEMIST_ROOT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(TT_ALCHEMIST_ROOT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)

# Set TT_MLIR_HOME - not to be confused with $ENV{TT_MLIR_HOME}
#
set(TT_MLIR_HOME ${PROJECT_SOURCE_DIR})

# Define target names
#
set(TT_ALCHEMIST_LIBRARY_NAME tt-alchemist-lib)

# Sources
#
set(LIB_SOURCES
  lib/tt_alchemist.cpp
  lib/model_to_cpp.cpp
  lib/model_to_python.cpp
  lib/generate_cpp.cpp
  lib/generate_python.cpp
  lib/utils.cpp
)

# Add library
#
add_library(${TT_ALCHEMIST_LIBRARY_NAME}
  SHARED
  ${LIB_SOURCES}
)

# Include dirs
#
target_include_directories(${TT_ALCHEMIST_LIBRARY_NAME}
  PUBLIC
    ${TT_ALCHEMIST_ROOT_SOURCE_DIR}/include  # External headers
  PRIVATE
    ${TT_ALCHEMIST_ROOT_SOURCE_DIR}/csrc/include  # Internal headers
    ${TT_MLIR_HOME}/include
)

# Find pybind11 for Python embedding support (used by python runner)
find_package(pybind11 CONFIG REQUIRED)

target_link_directories(${TT_ALCHEMIST_LIBRARY_NAME}
  PUBLIC
    ${PROJECT_BINARY_DIR}/lib
)

# Add dependencies
#
add_dependencies(${TT_ALCHEMIST_LIBRARY_NAME}
  TTMLIRCompiler
)

# Link libraries
target_link_libraries(${TT_ALCHEMIST_LIBRARY_NAME}
  PUBLIC
    TTMLIRCompiler
)

# Disable RTTI and set visibility
target_compile_options(${TT_ALCHEMIST_LIBRARY_NAME}
  PRIVATE
    -fno-rtti
    -fvisibility=hidden  # Hide symbols from other translation units
)

# Define export macro for C API to make symbols visible to other translation units
target_compile_definitions(${TT_ALCHEMIST_LIBRARY_NAME}
  PRIVATE
    TT_ALCHEMIST_EXPORT=__attribute__\(\(visibility\(\"default\"\)\)\)
)

# Set RPATH for the library
# At runtime, alchemist lib will look in its own directory for
# dependencies (libTTMLIRCompiler.so, libtt_metal.so, etc.)
set_target_properties(${TT_ALCHEMIST_LIBRARY_NAME} PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH FALSE
  # Build library directly into build/lib/ alongside TTMLIRCompiler.so
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
)

# Add interface link options to help external projects find transitive dependencies at link time
target_link_options(${TT_ALCHEMIST_LIBRARY_NAME}
  INTERFACE
    -Wl,-rpath-link,${PROJECT_BINARY_DIR}/lib
    -Wl,-rpath-link,${TTNN_INSTALL_DIR}/lib
)

################################################################################
# Python Model Runner Library (separate subdirectory due to RTTI requirements)
################################################################################

add_subdirectory(python_runner)

################################################################################
# Install headers and templates
################################################################################

# Install headers
add_custom_target(tt-alchemist-install-public-headers
  # Clean old headers directory to avoid stale files
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${TT_ALCHEMIST_ROOT_BINARY_DIR}/include
  # Create new headers directory
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TT_ALCHEMIST_ROOT_BINARY_DIR}/include
  # Copy only .hpp files from external headers (filters out README.md, etc.)
  COMMAND ${CMAKE_COMMAND}
    -DSOURCE_DIR=${TT_ALCHEMIST_ROOT_SOURCE_DIR}/include/tt-alchemist
    -DDEST_DIR=${TT_ALCHEMIST_ROOT_BINARY_DIR}/include
    -P ${CMAKE_CURRENT_SOURCE_DIR}/CopyHppFiles.cmake

  COMMENT "Copying tt-alchemist headers"
)

# Install templates
add_custom_target(tt-alchemist-install-templates
  # Clean old templates directory to avoid stale files
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${TT_ALCHEMIST_ROOT_BINARY_DIR}/templates || cmake -E true
  # Create new templates directories
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TT_ALCHEMIST_ROOT_BINARY_DIR}/templates
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TT_ALCHEMIST_ROOT_BINARY_DIR}/templates/cpp
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TT_ALCHEMIST_ROOT_SOURCE_DIR}/templates/cpp/local ${TT_ALCHEMIST_ROOT_BINARY_DIR}/templates/cpp/local
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TT_ALCHEMIST_ROOT_SOURCE_DIR}/templates/cpp/standalone ${TT_ALCHEMIST_ROOT_BINARY_DIR}/templates/cpp/standalone
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TT_ALCHEMIST_ROOT_BINARY_DIR}/templates/python
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TT_ALCHEMIST_ROOT_SOURCE_DIR}/templates/python/local ${TT_ALCHEMIST_ROOT_BINARY_DIR}/templates/python/local
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TT_ALCHEMIST_ROOT_SOURCE_DIR}/templates/python/standalone ${TT_ALCHEMIST_ROOT_BINARY_DIR}/templates/python/standalone

  COMMENT "Copying tt-alchemist templates"
)

# Include cpp standalone component installation utilities
include(${CMAKE_CURRENT_SOURCE_DIR}/InstalCppStandaloneComponents.cmake)

# Target to install all components for cpp standalone
add_custom_target(tt-alchemist-install-cpp-standalone
  DEPENDS
    tt-alchemist-install-templates
    ${TTNN_INSTALL_DIR}/.metalium-runtime-installed
    ${TTNN_INSTALL_DIR}/.metalium-dev-installed
    ${TTNN_INSTALL_DIR}/.ttnn-runtime-installed
    ${TTNN_INSTALL_DIR}/.ttnn-dev-installed
    ${TTNN_INSTALL_DIR}/.missing-headers-installed
    ${TTNN_INSTALL_DIR}/.all-components-installed
  COMMENT "Installing all required TT-NN components for C++ standalone mode"
)

# Useful for other project to use tt-alchemist as a dependency, like tt-xla
add_custom_target(tt-alchemist-package ALL
  DEPENDS
    ${TT_ALCHEMIST_LIBRARY_NAME}
    tt-alchemist-install-public-headers
    tt-alchemist-install-cpp-standalone
  COMMENT "Packaging tt-alchemist"
)

# Default target: just build the library and package
add_custom_target(tt-alchemist ALL
  DEPENDS tt-alchemist-package
  COMMENT "Building tt-alchemist library and package"
)

################################################################################
# Python Wheel Building
################################################################################

if(TTMLIR_ENABLE_ALCHEMIST_WHEEL)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  # Set Python wheel paths
  set(PYTHON_SRC_DIR ${TT_ALCHEMIST_ROOT_SOURCE_DIR}/python)
  set(PYTHON_WHEEL_DIR ${TT_ALCHEMIST_ROOT_BINARY_DIR}/wheel)

  # Target to prepare wheel directory structure with symlinks for editable install
  add_custom_target(tt-alchemist-prepare-wheel
    DEPENDS
      ${TT_ALCHEMIST_LIBRARY_NAME}
      tt-alchemist-package

    # Clean and create wheel directory
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${PYTHON_WHEEL_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_WHEEL_DIR}

    # Copy Python source files
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PYTHON_SRC_DIR} ${PYTHON_WHEEL_DIR}

    # Create symlink to lib directory (contains all .so files)
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${PROJECT_BINARY_DIR}/lib
      ${PYTHON_WHEEL_DIR}/tt_alchemist/lib

    # Create symlink to templates
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${TT_ALCHEMIST_ROOT_BINARY_DIR}/templates
      ${PYTHON_WHEEL_DIR}/tt_alchemist/templates

    # Create symlink to ttnn-install
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${TTNN_INSTALL_DIR}
      ${PYTHON_WHEEL_DIR}/tt_alchemist/ttnn_install

    COMMENT "Preparing wheel directory with symlinks for editable install"
  )

  # Editable install (uses symlinks, no copying)
  add_custom_target(tt-alchemist-install-wheel-editable
    DEPENDS tt-alchemist-prepare-wheel
    COMMAND ${Python3_EXECUTABLE} -m pip install -e ${PYTHON_WHEEL_DIR} --no-deps
    COMMENT "Installing tt-alchemist in editable mode"
  )

  # Build wheel (for distribution, copies files instead of symlinks)
  add_custom_target(tt-alchemist-build-wheel-distribution
    DEPENDS tt-alchemist-prepare-wheel

    # Clean previous build artifacts
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${PYTHON_WHEEL_DIR}/build
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${PYTHON_WHEEL_DIR}/tt_alchemist.egg-info
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/dist
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/dist

    # For distribution wheel, copy required ttnn-install .so files into lib directory
    # This consolidates all shared libraries into one location for simpler RPATH
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TTNN_INSTALL_DIR}/lib/_ttnncpp.so ${PYTHON_WHEEL_DIR}/tt_alchemist/lib/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TTNN_INSTALL_DIR}/lib/libtt_metal.so ${PYTHON_WHEEL_DIR}/tt_alchemist/lib/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TTNN_INSTALL_DIR}/lib/libdevice.so ${PYTHON_WHEEL_DIR}/tt_alchemist/lib/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TTNN_INSTALL_DIR}/lib/libtt_stl.so ${PYTHON_WHEEL_DIR}/tt_alchemist/lib/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TTNN_INSTALL_DIR}/lib/libtracy.so.0.10.0 ${PYTHON_WHEEL_DIR}/tt_alchemist/lib/

    # Build wheel
    COMMAND ${Python3_EXECUTABLE} -m pip wheel
      -w ${CMAKE_CURRENT_BINARY_DIR}/dist
      ${PYTHON_WHEEL_DIR}
      --no-deps

    WORKING_DIRECTORY ${PYTHON_WHEEL_DIR}
    COMMENT "Building tt-alchemist wheel"
  )

  # Install wheel (non-editable)
  add_custom_target(tt-alchemist-install-wheel-distribution
    DEPENDS tt-alchemist-build-wheel-distribution
    COMMAND ${Python3_EXECUTABLE} -m pip install
      --force-reinstall
      ${CMAKE_CURRENT_BINARY_DIR}/dist/*.whl
    COMMENT "Installing tt-alchemist wheel"
  )

  # Override default target to add editable wheel install
  add_dependencies(tt-alchemist tt-alchemist-install-wheel-editable)
endif()

################################################################################
# Tests
################################################################################

add_subdirectory(${TT_ALCHEMIST_ROOT_SOURCE_DIR}/test ${CMAKE_CURRENT_BINARY_DIR}/test)
