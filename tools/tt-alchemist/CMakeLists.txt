cmake_minimum_required(VERSION 3.16)

# tt-alchemist: A user-friendly abstraction layer for tt-mlir
set(TTALCHEMIST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TTALCHEMIST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(TTALCHEMIST_ENABLE_TESTS "Enable tests for tt-alchemist" ON)
option(TTALCHEMIST_ENABLE_PYTHON "Enable Python bindings for tt-alchemist" ON)

# Include directories
include_directories(
  ${TTALCHEMIST_SOURCE_DIR}/include
  ${TTALCHEMIST_BINARY_DIR}/include
)

# Add source directories
add_subdirectory(src)

# Add Python bindings if enabled
if(TTALCHEMIST_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

# Add tests if enabled
if(TTALCHEMIST_ENABLE_TESTS)
  add_subdirectory(test)
endif()

# Install templates
install(
  DIRECTORY ${TTALCHEMIST_SOURCE_DIR}/templates/
  DESTINATION share/tt-alchemist/templates
  COMPONENT TTAlchemist
)
