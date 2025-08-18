# Set TT_ALCHEMIST_HOME
#
set(TT_ALCHEMIST_HOME ${PROJECT_SOURCE_DIR}/tools/tt-alchemist)

# Set TT_MLIR_HOME - not to be confused with $ENV{TT_MLIR_HOME}
#
set(TT_MLIR_HOME ${PROJECT_SOURCE_DIR})

# Define target names
#
set(ALCHEMIST_LIBRARY_NAME tt-alchemist-lib)

# Sources
#
set(LIB_SOURCES
  lib/tt_alchemist.cpp
  lib/model_to_cpp.cpp
  lib/model_to_python.cpp
  lib/generate_cpp.cpp
  lib/generate_python.cpp
)

# Add library
#
add_library(${ALCHEMIST_LIBRARY_NAME}
  SHARED
  EXCLUDE_FROM_ALL
  ${LIB_SOURCES}
)

# Add dependencies
#
add_dependencies(${ALCHEMIST_LIBRARY_NAME}
  TTMLIRCompiler
)

# Include dirs
target_include_directories(${ALCHEMIST_LIBRARY_NAME}
  PUBLIC
    ${TT_ALCHEMIST_HOME}/include  # External headers
  PRIVATE
    ${TT_ALCHEMIST_HOME}/csrc/include  # Internal headers
    ${TT_MLIR_HOME}/include
)

target_link_directories(${ALCHEMIST_LIBRARY_NAME}
  PUBLIC
    ${PROJECT_BINARY_DIR}/lib
)

# Link libraries
target_link_libraries(${ALCHEMIST_LIBRARY_NAME}
  PRIVATE
    TTMLIRCompilerStatic
)

# Disable RTTI and set visibility
target_compile_options(${ALCHEMIST_LIBRARY_NAME}
  PRIVATE
    -fno-rtti
    -fvisibility=hidden
)

# Define export macro
target_compile_definitions(${ALCHEMIST_LIBRARY_NAME}
  PRIVATE
    TT_ALCHEMIST_EXPORT=__attribute__\(\(visibility\(\"default\"\)\)\)
)

# Set RPATH for the library
# At runtime, libtt-alchemist-lib.so will look in its own directory for
# dependencies (libtt_metal.so, etc.)
set_target_properties(${ALCHEMIST_LIBRARY_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)

# Install the library
install(TARGETS ${ALCHEMIST_LIBRARY_NAME}
  EXPORT ${ALCHEMIST_LIBRARY_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
  COMPONENT tt-alchemist-runtime
)

# Install the main library header
install(FILES
  ${TT_ALCHEMIST_HOME}/include/tt-alchemist/tt_alchemist_c_api.hpp
  DESTINATION include/tt-alchemist
  COMPONENT tt-alchemist-dev
)

# Install Python API
install(DIRECTORY
  ${TT_ALCHEMIST_HOME}/python/tt_alchemist/
  DESTINATION python/tt_alchemist
  COMPONENT tt-alchemist-python
  FILES_MATCHING PATTERN "*.py"
  PATTERN "cli.py" EXCLUDE
)

# Install tt-alchemist library in lib subdirectory relative to Python API
install(TARGETS ${ALCHEMIST_LIBRARY_NAME}
  LIBRARY DESTINATION python/tt_alchemist/lib
  COMPONENT tt-alchemist-python
)

# Install tt-metal dependencies in lib subdirectory relative to Python API
# These files are needed at runtime for the tt-alchemist library to function
if(DEFINED TTMETAL_LIBRARY_DIR AND EXISTS ${TTMETAL_LIBRARY_DIR})
  install(FILES
    ${TTMETAL_LIBRARY_DIR}/libtt_metal.so
    ${TTMETAL_LIBRARY_DIR}/_ttnncpp.so
    ${TTMETAL_LIBRARY_DIR}/libdevice.so
    DESTINATION python/tt_alchemist/lib
    COMPONENT tt-alchemist-python
    OPTIONAL
  )
endif()

# Install templates directory at the same level as lib folder
install(DIRECTORY
  ${TT_ALCHEMIST_HOME}/templates/
  DESTINATION python/tt_alchemist/templates
  COMPONENT tt-alchemist-python
)

# Create convenience targets for installing specific components
add_custom_target(install-tt-alchemist-runtime
  COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --component tt-alchemist-runtime --prefix ${CMAKE_INSTALL_PREFIX}
  DEPENDS ${ALCHEMIST_LIBRARY_NAME}
  COMMENT "Installing tt-alchemist runtime components (library and dependencies)"
)

add_custom_target(install-tt-alchemist-dev
  COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --component tt-alchemist-dev --prefix ${CMAKE_INSTALL_PREFIX}
  COMMENT "Installing tt-alchemist development headers"
)

add_custom_target(install-tt-alchemist-python
  COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --component tt-alchemist-python --prefix ${CMAKE_INSTALL_PREFIX}
  COMMENT "Installing tt-alchemist Python API"
)

add_custom_target(install-tt-alchemist-all
  DEPENDS install-tt-alchemist-runtime install-tt-alchemist-dev install-tt-alchemist-python install-templates
  COMMENT "Installing all tt-alchemist components including templates"
)

# Include component installation utilities
include(${CMAKE_CURRENT_SOURCE_DIR}/InstallComponents.cmake)

# Python wheel building and installation
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Copy Python files to build directory
set(PYTHON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../python)
set(PYTHON_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/../python)

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
)

# Copy Python files
add_custom_command(
  OUTPUT ${PYTHON_BUILD_DIR}/pyproject.toml
  DEPENDS ${PYTHON_SOURCE_FILES} ${PYTHON_BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PYTHON_SRC_DIR} ${PYTHON_BUILD_DIR}
  COMMENT "Copying Python files to build directory"
)

# Create lib directory in the Python package
add_custom_command(
  OUTPUT ${PYTHON_BUILD_DIR}/tt_alchemist/lib
  DEPENDS ${PYTHON_BUILD_DIR}/pyproject.toml
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_BUILD_DIR}/tt_alchemist/lib
  COMMENT "Creating lib directory in Python package"
)

# Copy all required shared libraries to the Python package
# TODO: Eventually leverage these libs from ttnn-install for smaller footprint
add_custom_command(
  OUTPUT
    ${PYTHON_BUILD_DIR}/tt_alchemist/lib/libtt-alchemist-lib.so
    ${PYTHON_BUILD_DIR}/tt_alchemist/lib/libtt_metal.so
    ${PYTHON_BUILD_DIR}/tt_alchemist/lib/_ttnncpp.so
    ${PYTHON_BUILD_DIR}/tt_alchemist/lib/libdevice.so
  DEPENDS ${ALCHEMIST_LIBRARY_NAME} ${PYTHON_BUILD_DIR}/tt_alchemist/lib
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${ALCHEMIST_LIBRARY_NAME}> ${PYTHON_BUILD_DIR}/tt_alchemist/lib/
  COMMAND ${CMAKE_COMMAND} -E copy ${TTMETAL_LIBRARY_DIR}/libtt_metal.so ${PYTHON_BUILD_DIR}/tt_alchemist/lib/
  COMMAND ${CMAKE_COMMAND} -E copy ${TTMETAL_LIBRARY_DIR}/_ttnncpp.so ${PYTHON_BUILD_DIR}/tt_alchemist/lib/
  COMMAND ${CMAKE_COMMAND} -E copy ${TTMETAL_LIBRARY_DIR}/libdevice.so ${PYTHON_BUILD_DIR}/tt_alchemist/lib/
  COMMENT "Copying shared libraries to Python package"
)

# Copy all templates and install components for cpp standalone
add_custom_target(tt_alchemist_copy_templates
  DEPENDS install-ttnn-components
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_BUILD_DIR}/tt_alchemist/templates
  # C++ templates
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_BUILD_DIR}/tt_alchemist/templates/cpp
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../templates/cpp/local ${PYTHON_BUILD_DIR}/tt_alchemist/templates/cpp/local
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../templates/cpp/standalone ${PYTHON_BUILD_DIR}/tt_alchemist/templates/cpp/standalone
  # Python templates
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_BUILD_DIR}/tt_alchemist/templates/python
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../templates/python/local ${PYTHON_BUILD_DIR}/tt_alchemist/templates/python/local
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../templates/python/standalone ${PYTHON_BUILD_DIR}/tt_alchemist/templates/python/standalone
  # Copy cpp standalone libs
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TTNN_INSTALL_DIR} ${PYTHON_BUILD_DIR}/tt_alchemist/templates/cpp/standalone/ttnn-install
  COMMENT "Copying templates and TT-NN components into Python package"
)

# Build wheel
add_custom_target(build-wheel-tt-alchemist
  DEPENDS
    ${PYTHON_BUILD_DIR}/pyproject.toml
    ${PYTHON_BUILD_DIR}/tt_alchemist/lib/libtt-alchemist-lib.so
    ${PYTHON_BUILD_DIR}/tt_alchemist/lib/libtt_metal.so
    ${PYTHON_BUILD_DIR}/tt_alchemist/lib/_ttnncpp.so
    ${PYTHON_BUILD_DIR}/tt_alchemist/lib/libdevice.so
    tt_alchemist_copy_templates
  COMMAND ${Python3_EXECUTABLE} -m pip wheel -w ${CMAKE_CURRENT_BINARY_DIR}/dist ${PYTHON_BUILD_DIR}
  WORKING_DIRECTORY ${PYTHON_BUILD_DIR}
  COMMENT "Building tt-alchemist wheel"
)

# Install tt-alchemist wheel
add_custom_target(tt-alchemist
  DEPENDS build-wheel-tt-alchemist
  COMMAND ${Python3_EXECUTABLE} -m pip install --force-reinstall ${CMAKE_CURRENT_BINARY_DIR}/dist/*.whl
  COMMENT "Installing tt-alchemist wheel"
)

# Install templates directly under install/templates using proper CMake install
# This ensures templates are always copied during installation
install(DIRECTORY
  ${TT_ALCHEMIST_HOME}/templates/
  DESTINATION templates
  COMPONENT tt-alchemist-templates
  FILES_MATCHING
  PATTERN "*"
  PATTERN ".git*" EXCLUDE
)

# Create convenience target for installing templates
add_custom_target(install-templates
  COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --component tt-alchemist-templates --prefix ${CMAKE_INSTALL_PREFIX}
  COMMENT "Installing templates to install/templates"
)
