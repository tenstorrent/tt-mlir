cmake_minimum_required(VERSION 3.24)

project(
    tt_crank
    VERSION 0.1.0
    DESCRIPTION "Compiler & runtime frontend for Tenstorrent hardware"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_testing()

option(TT_CRANK_BUILD_TESTS "Build tt-crank unit tests" ON)
option(TT_CRANK_BUILD_PYTHON_EXT "Build Python extension modules" ON)
option(TT_CRANK_TRACY_ZONES "Compile tt-crank with tracy zones" ON)

# Mark `tt-mlir` include files as system includes - to ignore warnings on them.
include(cmake/SystemIncludes.cmake)
mark_as_system_includes(TTMLIRCompiler TTMLIRRuntime)

set(TT_CRANK_TT_METAL_HOME "${CMAKE_SOURCE_DIR}/third_party/tt-metal/src/tt-metal")

# Silence the per-file "-- Installing:"/"-- Up-to-date:" messages on install.
# Set before any add_subdirectory so it applies to third-party install rules too.
set(CMAKE_INSTALL_MESSAGE NEVER)

include(cmake/CompilerOptions.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/StaticAnalyzers.cmake)

# TODO(pilkic): `tt-mlir` should properly set warnings (using an interface, not
# setting global compile options).
get_directory_property(_tp_compile_options COMPILE_OPTIONS)
list(REMOVE_ITEM _tp_compile_options -Werror)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${_tp_compile_options}")

add_subdirectory(third_party)
add_subdirectory(src)

include(cmake/TTsim.cmake)
add_subdirectory(python)

if(TT_CRANK_BUILD_TESTS)
    add_subdirectory(tests)
endif()
