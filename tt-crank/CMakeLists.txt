cmake_minimum_required(VERSION 3.24)

project(
    tt_kurbla
    VERSION 0.1.0
    DESCRIPTION "Compiler & runtime frontend for Tenstorrent hardware"
    LANGUAGES CXX
)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    enable_testing()
endif()

option(TT_KURBLA_BUILD_TESTS "Build tt-kurbla unit tests" ${PROJECT_IS_TOP_LEVEL})

include(cmake/Cache.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/StaticAnalyzers.cmake)
include(cmake/Ttsim.cmake)

add_subdirectory(third_party)
add_subdirectory(src)

if(TT_KURBLA_BUILD_TESTS AND PROJECT_IS_TOP_LEVEL)
    add_subdirectory(tests)
endif()
