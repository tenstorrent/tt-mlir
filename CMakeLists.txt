cmake_minimum_required(VERSION 3.24.0)

if(NOT DEFINED CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER clang CACHE STRING "C Compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER clang++ CACHE STRING "C++ Compiler" FORCE)
endif()

project(tt-mlir LANGUAGES CXX C)

if (NOT DEFINED ENV{TTMLIR_ENV_ACTIVATED})
  message(FATAL_ERROR "tt-mlir environment not activated. Please run 'source env/activate'.")
endif()

option(TT_RUNTIME_ENABLE_PERF_TRACE "Enable performance mode" OFF)
option(TTMLIR_ENABLE_RUNTIME "Enable runtime" OFF)
option(TTMLIR_ENABLE_PYKERNEL "Enable python kernels" OFF)
option(TTMLIR_ENABLE_TTNN_JIT "Enable ttnn-jit" OFF)
option(TTMLIR_ENABLE_STABLEHLO "Enable StableHLO support" OFF)
option(TTMLIR_ENABLE_TTIRTONVVM "Enable lowering TTIR to NVVM" OFF)
option(TTMLIR_ENABLE_OPMODEL "Enable OpModel support" OFF)
option(TTMLIR_ENABLE_OPMODEL_TESTS "Enable OpModel tests" ON)
option(TTMLIR_ENABLE_SHARED_LIB "Enable Shared lib building" ON)
option(TTMLIR_ENABLE_EXPLORER "Enable cloning and building the explorer tool" ON)
option(TTMLIR_ENABLE_ALCHEMIST "Enable tt-alchemist tool" ON)
option(CODE_COVERAGE "Enable coverage reporting" OFF)

if (TTMLIR_ENABLE_RUNTIME)
  option(TT_RUNTIME_ENABLE_TTNN "Enable TTNN Runtime (requires TTMLIR_ENABLE_RUNTIME)" ON)
  option(TT_RUNTIME_ENABLE_TTMETAL "Enable TTMetal Runtime (requires TTMLIR_ENABLE_RUNTIME)" ON)
  option(TT_RUNTIME_ENABLE_DISTRIBUTED "Enable Distributed Runtime (requires TTMLIR_ENABLE_RUNTIME)" ON)
else()
  set(TT_RUNTIME_ENABLE_TTNN OFF CACHE BOOL "Enable TTNN Runtime (requires TTMLIR_ENABLE_RUNTIME)" FORCE)
  set(TT_RUNTIME_ENABLE_TTMETAL OFF CACHE BOOL "Enable TTMetal Runtime (requires TTMLIR_ENABLE_RUNTIME)" FORCE)
  set(TT_RUNTIME_ENABLE_DISTRIBUTED OFF CACHE BOOL "Enable Distributed Runtime (requires TTMLIR_ENABLE_RUNTIME)" FORCE)
endif()
option(TTMLIR_ENABLE_RUNTIME_TESTS "Enable runtime tests" OFF)
option(TT_RUNTIME_DEBUG "Enable debug tools in runtime" OFF)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Asan")
  set(TT_RUNTIME_DEBUG ON)
endif()

if (CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
endif()

if (TTMLIR_ENABLE_STABLEHLO)
  add_compile_definitions(TTMLIR_ENABLE_STABLEHLO)
endif()

if (TTMLIR_ENABLE_TTIRTONVVM)
  add_compile_definitions(TTMLIR_ENABLE_TTIRTONVVM)
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

# Enable debug asserts and debug logs only in Debug mode
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(TT_BUILD_DEBUG TTMLIR_ENABLE_DEBUG_LOGS)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter --system-header-prefix=ENV{TTMLIR_TOOLCHAIN_DIR})

set(CMAKE_LINKER_TYPE DEFAULT)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE CLANG_VERSION_INFO ERROR_QUIET)
  string(REGEX MATCH "version ([0-9]+)\\.[0-9]+" CLANG_VERSION "${CLANG_VERSION_INFO}")
  set(CLANG_VERSION_MAJOR ${CMAKE_MATCH_1})
  set(LD_LLD_EXECUTABLE_VERSIONED "ld.lld-${CLANG_VERSION_MAJOR}")
  find_program(LLD NAMES ${LD_LLD_EXECUTABLE_VERSIONED} ld.lld)
  if(LLD)
    execute_process(COMMAND ${LLD} --version OUTPUT_VARIABLE LLD_VERSION_INFO ERROR_QUIET)
    string(REGEX MATCH "LLD ([0-9]+)\\.[0-9]+" LLD_VERSION "${LLD_VERSION_INFO}")
    set(LLD_VERSION_MAJOR ${CMAKE_MATCH_1})
    if (CLANG_VERSION_MAJOR EQUAL LLD_VERSION_MAJOR)
      message(STATUS "Using LLD linker: ${LLD}")
      set(CMAKE_LINKER_TYPE LLD)
    endif()
  endif()
endif()

include(TTMLIRBuildTypes)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

# If there are multiple python env we need to explicitly set the python3 executable
set(Python3_EXECUTABLE $ENV{TTMLIR_VENV_DIR}/bin/python3)

include(FindMLIR)
include(TTMLIRVersion)

add_library(coverage_config INTERFACE)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Add required flags (GCC & LLVM/Clang)
  target_compile_options(coverage_config INTERFACE
    -O0        # no optimization
    -g         # generate debug info
    --coverage # sets all required flags
    -fPIC
  )
  target_link_options(coverage_config INTERFACE --coverage)
endif()

install(TARGETS coverage_config EXPORT TTNNOpModelLibTargets)
install(TARGETS coverage_config EXPORT MLIRTTNNDialect)
install(TARGETS coverage_config EXPORT TTMLIRTTNNUtils)

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
include_directories(${TTMLIR_TOOLCHAIN_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
include(TTMLIRPythonSitePackages)

add_subdirectory(third_party)
if (TTMLIR_ENABLE_STABLEHLO)
  set(STABLEHLO_BUILD_EMBEDDED ON)
  set(STABLEHLO_ENABLE_BINDINGS_PYTHON ON)
  include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/shardy)
  include_directories(SYSTEM ${TTMLIR_TOOLCHAIN_DIR}/src/shardy)
  include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/stablehlo)
  include_directories(SYSTEM ${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo)
  add_subdirectory(${TTMLIR_TOOLCHAIN_DIR}/src/shardy ${CMAKE_CURRENT_BINARY_DIR}/shardy EXCLUDE_FROM_ALL)
  add_subdirectory(${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo ${CMAKE_CURRENT_BINARY_DIR}/stablehlo EXCLUDE_FROM_ALL)
  install(DIRECTORY shardy ${CMAKE_CURRENT_BINARY_DIR}/shardy/shardy
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT SharedLib
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
  )
  install(DIRECTORY stablehlo ${CMAKE_CURRENT_BINARY_DIR}/stablehlo/stablehlo
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT SharedLib
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
  )
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
