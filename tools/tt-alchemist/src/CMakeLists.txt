# Configure envs
#
if(NOT DEFINED ENV{TT_MLIR_HOME})
  message(FATAL_ERROR "TT_MLIR_HOME environment variable is not set")
endif()

# Set TT_ALCHEMIST_HOME
#
set(TT_ALCHEMIST_HOME $ENV{TT_MLIR_HOME}/tools/tt-alchemist)

# Define lib name
#
set(LIBRARY_NAME tt-alchemist)

# Sources
#
set(SOURCES
  lib/tt_alchemist.cpp
)

# Add library
#
add_library(${LIBRARY_NAME}
  SHARED
  EXCLUDE_FROM_ALL
  ${SOURCES}
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
  INCLUDES DESTINATION include
)
