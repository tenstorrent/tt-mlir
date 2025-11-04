cmake_minimum_required(VERSION 3.24)
project(tt-alchemist-example)

# Require TTMLIR_BUILD_DIR to be set
if(NOT DEFINED TTMLIR_BUILD_DIR)
  message(FATAL_ERROR
    "TTMLIR_BUILD_DIR must be set. Example:\n"
    "  cmake -DTTMLIR_BUILD_DIR=/path/to/tt-mlir/build .."
  )
endif()

# Verify tt-alchemist-package was built
set(TEMPLATES_DIR "${TTMLIR_BUILD_DIR}/tools/tt-alchemist/templates")
if(NOT EXISTS "${TEMPLATES_DIR}")
  message(FATAL_ERROR
    "Templates directory not found at: ${TEMPLATES_DIR}\n"
    "Please build tt-alchemist-package first:\n"
    "  cmake --build ${TTMLIR_BUILD_DIR} --target tt-alchemist-package"
  )
endif()

message(STATUS "Using tt-mlir build from: ${TTMLIR_BUILD_DIR}")
message(STATUS "Templates found at: ${TEMPLATES_DIR}")

# Add the build/lib directory to library search path
link_directories(${TTMLIR_BUILD_DIR}/lib)

# Create an example executable
add_executable(example_usage example.cpp)

# Link against tt-alchemist-lib
target_link_libraries(example_usage PRIVATE tt-alchemist-lib TTMLIRCompiler)

# Include directories
target_include_directories(example_usage PRIVATE
  ${TTMLIR_BUILD_DIR}/../tools/tt-alchemist/include
)

# Set RPATH so it finds libraries at runtime
set_target_properties(example_usage PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${TTMLIR_BUILD_DIR}/lib"
)
