cmake_minimum_required(VERSION 3.20.0)
project(tt-mlir LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/modules)

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter --system-header-prefix=/opt/ttmlir-toolchain)

include(TTMLIRBuildTypes)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

include(FindMLIR)
include(TTMLIRVersion)

set(TTMLIR_TOOLCHAIN "/opt/ttmlir-toolchain" CACHE STRING "Path to the toolchain")
set(TTMLIR_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TTMLIR_BINARY_DIR ${PROJECT_BINARY_DIR})
set(LLVM_LIT_TOOLS_DIR "${TTMLIR_TOOLCHAIN}/src/llvm-project/llvm/utils/lit")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${TTMLIR_SOURCE_DIR}/include)
include_directories(${TTMLIR_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
include(TTMLIRPythonSitePackages)

add_subdirectory(third_party)
add_subdirectory(include)
add_subdirectory(lib)
if(MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  set(TTMLIR_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
  add_subdirectory(python)
endif()
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(runtime)

include(LintTools)
