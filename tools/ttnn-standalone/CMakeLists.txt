cmake_minimum_required(VERSION 3.16)
project(ttnn-standalone CXX)

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
if("$ENV{ARCH_NAME}" STREQUAL "")
    message(FATAL_ERROR "ARCH_NAME is not set")
endif()

# Check if TT_METAL_HOME_ORIGINAL is set
if(DEFINED ENV{TT_METAL_HOME_ORIGINAL})
    # Save the current value of TT_METAL_HOME
    set(ORIGINAL_TT_METAL_HOME $ENV{TT_METAL_HOME})

    # Swap TT_METAL_HOME with TT_METAL_HOME_ORIGINAL
    set(ENV{TT_METAL_HOME} $ENV{TT_METAL_HOME_ORIGINAL})
    message(STATUS "Swapped TT_METAL_HOME with TT_METAL_HOME_ORIGINAL.")
endif()

message($ENV{TT_METAL_HOME}/tt_metal/third_party/src/firmware/riscv/$ENV{ARCH_NAME})

# Directories to search for headers
#
set(INCLUDE_DIRS
    # TODO: Remove these when ttmetal removes the dependencies from public facing headers
    $ENV{TT_METAL_HOME}/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f
    $ENV{TT_METAL_HOME}/.cpmcache/fmt/73b5ec45edbd92babfd91c3777a9e1ab9cac8238/include
    $ENV{TT_METAL_HOME}/.cpmcache/magic_enum/4d76fe0a5b27a0e62d6c15976d02b33c54207096/include
    $ENV{TT_METAL_HOME}/.cpmcache/boost_core/e679bef5c160cf29d0f37d549881dc5f5a58c332/include
    $ENV{TT_METAL_HOME}/.cpmcache/json/230202b6f5267cbf0c8e5a2f17301964d95f83ff/include

    # Metalium
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api
    $ENV{TT_METAL_HOME}/tt_metal/third_party/fmt
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/${ARCH_NAME}
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/${ARCH_EXTRA_DIR}
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/${ARCH_NAME}
    $ENV{TT_METAL_HOME}/tt_metal/third_party/magic_enum
    $ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public

    # TTNN
    $ENV{TT_METAL_HOME}/ttnn/cpp
    $ENV{TT_METAL_HOME}/ttnn/cpp/ttnn
    $ENV{TT_METAL_HOME}/ttnn/cpp/ttnn/deprecated
)

# Link directories
#
set(LINK_DIRS
    $ENV{TT_METAL_HOME}-build/lib
)

# Libraries to link
#
set(LINK_LIBS
    tt_metal
    device
    yaml-cpp
    pthread

    # TTNN
    # _ttnn  # Why doesn't this work?
    $ENV{TT_METAL_HOME}-build/lib/_ttnn.so
)

# Compile definitions
#
set(COMPILE_DEFS
    FMT_HEADER_ONLY  # needed for fmt lib
)

#       _                     _         _
#  ___ | |_  __ _  _ __    __| |  __ _ | |  ___   _ __    ___
# / __|| __|/ _` || '_ \  / _` | / _` || | / _ \ | '_ \  / _ \
# \__ \| |_| (_| || | | || (_| || (_| || || (_) || | | ||  __/
# |___/ \__|\__,_||_| |_| \__,_| \__,_||_| \___/ |_| |_| \___|

add_executable(ttnn-standalone ttnn-standalone.cpp)
set_property(TARGET ttnn-standalone PROPERTY CXX_STANDARD 20)

target_include_directories(ttnn-standalone PRIVATE ${INCLUDE_DIRS})
target_link_directories(ttnn-standalone PRIVATE ${LINK_DIRS})
target_link_libraries(ttnn-standalone PRIVATE ${LINK_LIBS})
target_compile_definitions(ttnn-standalone PRIVATE ${COMPILE_DEFS})

target_precompile_headers(ttnn-standalone PRIVATE ttnn-precompiled.hpp)

#      _         _  _  _
#   __| | _   _ | |(_)| |__
#  / _` || | | || || || '_ \
# | (_| || |_| || || || |_) |
#  \__,_| \__, ||_||_||_.__/
#         |___/

add_library(ttnn-dylib SHARED ttnn-dylib.cpp)
set_property(TARGET ttnn-dylib PROPERTY CXX_STANDARD 20)

target_include_directories(ttnn-dylib PRIVATE ${INCLUDE_DIRS})
target_link_directories(ttnn-dylib PRIVATE ${LINK_DIRS})
target_link_libraries(ttnn-dylib PRIVATE ${LINK_LIBS})
set_target_properties(ttnn-dylib PROPERTIES PUBLIC_HEADER ttnn-dylib.h)

target_precompile_headers(ttnn-dylib PRIVATE ttnn-dylib.hpp)
