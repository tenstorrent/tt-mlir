cmake_minimum_required(VERSION 3.20.0)
project(tt-mlir LANGUAGES CXX C)

if (NOT DEFINED ENV{TTMLIR_ENV_ACTIVATED})
  message(FATAL_ERROR "tt-mlir environment not activated. Please run 'source env/activate'.")
endif()

option(TT_RUNTIME_ENABLE_PERF_TRACE "Enable performance mode" OFF)
option(TTMLIR_ENABLE_RUNTIME "Enable runtime" OFF)
option(TTMLIR_ENABLE_PYKERNEL "Enable python kernels" OFF)
option(TTMLIR_ENABLE_STABLEHLO "Enable StableHLO support" OFF)
option(TTMLIR_ENABLE_OPMODEL "Enable OpModel support" OFF)
option(TTMLIR_ENABLE_SHARED_LIB "Enable Shared lib building" ON)
option(TTMLIR_ENABLE_DEBUG_STRINGS "Enable debug strings in flatbuffer" ON)
option(TTMLIR_ENABLE_EXPLORER "Enable cloning and building the explorer tool" ON)

if (TTMLIR_ENABLE_DEBUG_STRINGS)
  add_compile_definitions(TTMLIR_ENABLE_DEBUG_STRINGS)
endif()

if (NOT TTMLIR_ENABLE_RUNTIME)
  set (TTMLIR_ENABLE_SHARED_LIB OFF)
endif()

if (TTMLIR_ENABLE_STABLEHLO)
  add_compile_definitions(TTMLIR_ENABLE_STABLEHLO)
endif()

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TTMLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "Enable Python bindings")

if (APPLE)
  set(TTMLIR_ENABLE_OPMODEL OFF)
  message(WARNING "TTNNOpModelLib is disabled on Apple platforms. Optimizer will not get true performance.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/modules)

if (TT_RUNTIME_ENABLE_PERF_TRACE)
  add_compile_options(-DTRACY_ENABLE=ON)
endif()

if (NOT DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})
    message(FATAL_ERROR "TTMLIR_TOOLCHAIN_DIR environment variable not set. Please run 'source env/activate'.")
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter --system-header-prefix=ENV{TTMLIR_TOOLCHAIN_DIR})

include(TTMLIRBuildTypes)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

# If there are multiple python env we need to explicitly set the python3 executable
set(Python3_EXECUTABLE $ENV{TTMLIR_VENV_DIR}/bin/python3)

include(FindMLIR)
include(TTMLIRVersion)

if (CMAKE_BUILD_TYPE STREQUAL "Asan")
  # Address sanitization does not work with python bindings.
  if(MLIR_ENABLE_BINDINGS_PYTHON AND TTMLIR_ENABLE_BINDINGS_PYTHON)
    message(FATAL_ERROR "TTMLIR_ENABLE_BINDINGS_PYTHON must be set to OFF when building TTMLIR with address sanitization")
  endif()
  # Clang only links in the sanitizer runtime for the main executable, not for the shared libraries.
  if (TTMLIR_ENABLE_SHARED_LIB)
    message(FATAL_ERROR "TTMLIR_ENABLE_SHARED_LIB must be set to OFF when building TTMLIR with address sanitization")
  endif()
endif()

set(TTMLIR_TOOLCHAIN_DIR $ENV{TTMLIR_TOOLCHAIN_DIR})
set(TTMLIR_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TTMLIR_BINARY_DIR ${PROJECT_BINARY_DIR})
set(LLVM_TOOLS_DIR "${TTMLIR_TOOLCHAIN_DIR}/bin")
set(LLVM_LIT_TOOLS_DIR "${TTMLIR_TOOLCHAIN_DIR}/src/llvm-project/llvm/utils/lit")
set(SHLIBEXT ".so")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${TTMLIR_SOURCE_DIR}/include)
include_directories(${TTMLIR_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
include(TTMLIRPythonSitePackages)

add_subdirectory(third_party)
if (TTMLIR_ENABLE_STABLEHLO)
  set(STABLEHLO_BUILD_EMBEDDED ON)
  add_subdirectory(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo ${CMAKE_CURRENT_BINARY_DIR}/stablehlo EXCLUDE_FROM_ALL)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/stablehlo)
  include_directories(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
endif()
add_subdirectory(include)
add_subdirectory(lib)
if(MLIR_ENABLE_BINDINGS_PYTHON AND TTMLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  set(TTMLIR_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
  add_subdirectory(python)
endif()
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(runtime)
add_subdirectory(docs)
add_subdirectory(cmake/modules)

include(LintTools)
