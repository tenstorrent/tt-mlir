cmake_minimum_required(VERSION 3.20.0)
project(tt-mlir LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter --system-header-prefix=/opt/ttmlir-toolchain)

# Asan build type
set(CMAKE_C_FLAGS_ASAN
    "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer" CACHE STRING
    "Flags used by the C compiler for Asan build type or configuration." FORCE)
set(CMAKE_CXX_FLAGS_ASAN
    "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer" CACHE STRING
    "Flags used by the C++ compiler for Asan build type or configuration." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_ASAN
    "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address" CACHE STRING
    "Linker flags to be used to create executables for Asan build type." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_ASAN
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address" CACHE STRING
    "Linker lags to be used to create shared libraries for Asan build type." FORCE)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

# Find MLIR and LLVM
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE TTMLIR_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions("-DTTMLIR_GIT_HASH=${TTMLIR_GIT_HASH}")
add_definitions("-DTTMLIR_VERSION_MAJOR=0")
add_definitions("-DTTMLIR_VERSION_MINOR=0")
add_definitions("-DTTMLIR_VERSION_RELEASE=0")

set(TTMLIR_TOOLCHAIN "/opt/ttmlir-toolchain" CACHE STRING "Path to the toolchain")
set(TTMLIR_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TTMLIR_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${TTMLIR_SOURCE_DIR}/include)
include_directories(${TTMLIR_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
if(MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  set(TTMLIR_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
  add_subdirectory(python)
endif()
add_subdirectory(test)
add_subdirectory(tools)

# clang-tidy setup
add_custom_target(clang-tidy-filter-out-external-srcs COMMAND python3 ${TTMLIR_SOURCE_DIR}/tools/scripts/filter-compile-commands.py ${TTMLIR_BINARY_DIR}/compile_commands.json ${TTMLIR_SOURCE_DIR})
add_custom_target(clang-tidy COMMAND run-clang-tidy -p ${PROJECT_BINARY_DIR} -warnings-as-errors '*' DEPENDS clang-tidy-filter-out-external-srcs)
add_custom_target(clang-format COMMAND git-clang-format)
