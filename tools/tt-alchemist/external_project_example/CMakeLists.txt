cmake_minimum_required(VERSION 3.24)
project(tt-alchemist-example)

# Require TTMLIR_BUILD_DIR to be set
if(NOT DEFINED TTMLIR_BUILD_DIR)
  message(FATAL_ERROR
    "TTMLIR_BUILD_DIR must be set. Example:\n"
    "  cmake -DTTMLIR_BUILD_DIR=/path/to/tt-mlir/build .."
  )
endif()

# Set headers directory
set(HEADERS_DIR "${TTMLIR_BUILD_DIR}/tools/tt-alchemist/include")

# Add the build/lib directory to library search path
link_directories(${TTMLIR_BUILD_DIR}/lib)

# Create an example executable
add_executable(example_usage example.cpp)

# Link against tt-alchemist-lib
target_link_libraries(example_usage PRIVATE tt-alchemist-lib)

# Include directories
target_include_directories(example_usage PRIVATE
  ${HEADERS_DIR}
)
