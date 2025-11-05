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
set(HEADERS_DIR "${TTMLIR_BUILD_DIR}/tools/tt-alchemist/include")

if(NOT EXISTS "${TEMPLATES_DIR}")
  message(FATAL_ERROR
    "Templates directory not found at: ${TEMPLATES_DIR}\n"
    "Please build tt-alchemist-package first:\n"
    "  cmake --build ${TTMLIR_BUILD_DIR} --target tt-alchemist-package"
  )
endif()

if(NOT EXISTS "${HEADERS_DIR}")
  message(FATAL_ERROR
    "Headers directory not found at: ${HEADERS_DIR}\n"
    "Please build tt-alchemist-package first:\n"
    "  cmake --build ${TTMLIR_BUILD_DIR} --target tt-alchemist-package"
  )
endif()

message(STATUS "Using tt-mlir build from: ${TTMLIR_BUILD_DIR}")
message(STATUS "Templates found at: ${TEMPLATES_DIR}")
message(STATUS "Headers found at: ${HEADERS_DIR}")

# Enable RPATH for build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add the build/lib directory to library search path
link_directories(${TTMLIR_BUILD_DIR}/lib)

# Create an example executable
add_executable(example_usage example.cpp)

# Link against tt-alchemist-lib
target_link_libraries(example_usage PRIVATE tt-alchemist-lib TTMLIRCompiler)

# Include directories
target_include_directories(example_usage PRIVATE
  ${HEADERS_DIR}
)

# Set RPATH so it finds libraries at runtime
set_target_properties(example_usage PROPERTIES
  BUILD_RPATH "${TTMLIR_BUILD_DIR}/lib:${TTMLIR_BUILD_DIR}/tools/tt-alchemist/ttnn-install/lib"
  INSTALL_RPATH "${TTMLIR_BUILD_DIR}/lib:${TTMLIR_BUILD_DIR}/tools/tt-alchemist/ttnn-install/lib"
)

# Add rpath-link to help linker find transitive dependencies at link time
target_link_options(example_usage PRIVATE
  -Wl,-rpath-link,${TTMLIR_BUILD_DIR}/lib
  -Wl,-rpath-link,${TTMLIR_BUILD_DIR}/tools/tt-alchemist/ttnn-install/lib
)
