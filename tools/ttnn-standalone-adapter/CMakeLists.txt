cmake_minimum_required(VERSION 3.16)
project(ttnn-standalone-adapter CXX)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_VERBOSE_MAKEFILE on)

if(DEFINED ENV{CONFIG})
    message(STATUS "CONFIG is set, CMAKE_BUILD_TYPE being set to $ENV{CONFIG}")
    set(CMAKE_BUILD_TYPE $ENV{CONFIG})
elseif(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release build is the default" FORCE)
endif()

# Handle env:ARCH_NAME
#
# Due to ttmetal's folder structure, we need to do some overriding for ARCH_NAME (e.g. wormhole_b0 -> wormhole)
#
if ("$ENV{ARCH_NAME}" STREQUAL "grayskull")
  set(ARCH_NAME "grayskull")
  set(ARCH_EXTRA_DIR "grayskull")
elseif ("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
  set(ARCH_NAME "wormhole")
  set(ARCH_EXTRA_DIR "wormhole/wormhole_b0_defines")
elseif ("$ENV{ARCH_NAME}" STREQUAL "blackhole")
  set(ARCH_NAME "blackhole")
  set(ARCH_EXTRA_DIR "blackhole")
else()
  message(FATAL_ERROR "Unsupported ARCH_NAME: $ENV{ARCH_NAME}")
endif()

# Find clang 17
#
find_program(CLANG_17 clang++-17)
if(CLANG_17)
    message(STATUS "Found Clang-17 here: ${CLANG_17}")
    set(CMAKE_CXX_COMPILER "${CLANG_17}")
else()
    message(WARNING "Clang++-17 not found!!!")
endif()

if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()
if("$ENV{TT_METAL_BUILD_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_BUILD_HOME is not set")
endif()
if("$ENV{ARCH_NAME}" STREQUAL "")
    message(FATAL_ERROR "ARCH_NAME is not set")
endif()

message($ENV{TT_METAL_HOME}/tt_metal/third_party/src/firmware/riscv/$ENV{ARCH_NAME})

# Directories to search for headers
#
set(INCLUDE_DIRS
)

# Link directories
#
set(LINK_DIRS
)

# Libraries to link
#
set(LINK_LIBS
)

# Compile definitions
#
set(COMPILE_DEFS
)

#              _                _
#    __ _   __| |  __ _  _ __  | |_  ___  _ __
#   / _` | / _` | / _` || '_ \ | __|/ _ \| '__|
#  | (_| || (_| || (_| || |_) || |_|  __/| |
#   \__,_| \__,_| \__,_|| .__/  \__|\___||_|
#                       |_|

add_library(ttnn-standalone-adapter STATIC adapter.cpp)
set_property(TARGET ttnn-standalone-adapter PROPERTY CXX_STANDARD 20)

target_include_directories(ttnn-standalone-adapter PRIVATE ${INCLUDE_DIRS})
target_link_directories(ttnn-standalone-adapter PRIVATE ${LINK_DIRS})
target_link_libraries(ttnn-standalone-adapter PRIVATE ${LINK_LIBS})
target_compile_definitions(ttnn-standalone-adapter PRIVATE ${COMPILE_DEFS})
