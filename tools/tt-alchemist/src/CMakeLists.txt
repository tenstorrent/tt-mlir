# Source files
set(TTALCHEMIST_SRCS
  compiler.cpp
  solution_generator.cpp
  runtime.cpp
  tt_alchemist.cpp
)

# Create the shared library
add_library(tt-alchemist SHARED ${TTALCHEMIST_SRCS})

# Set include directories
target_include_directories(tt-alchemist
  PUBLIC
    ${TTALCHEMIST_SOURCE_DIR}/include
  PRIVATE
    ${TTALCHEMIST_SOURCE_DIR}/src/include
)

# Link against required libraries
target_link_libraries(tt-alchemist
  PRIVATE
    MLIRTTNNDialect
    MLIRTTNNTransforms
)

# Install the library
install(
  TARGETS tt-alchemist
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  COMPONENT TTAlchemist
)

# Install the headers
install(
  DIRECTORY ${TTALCHEMIST_SOURCE_DIR}/include/
  DESTINATION include
  COMPONENT TTAlchemist
)
