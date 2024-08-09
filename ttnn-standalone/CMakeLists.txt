cmake_minimum_required(VERSION 3.16)
project(ttnn-standalone CXX)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_VERBOSE_MAKEFILE on)

if(DEFINED ENV{CONFIG})
    message(STATUS "CONFIG is set, CMAKE_BUILD_TYPE being set to $ENV{CONFIG}")
    set(CMAKE_BUILD_TYPE $ENV{CONFIG})
elseif(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release build is the default" FORCE)
endif()

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

message($ENV{TT_METAL_HOME}/tt_metal/third_party/src/firmware/riscv/$ENV{ARCH_NAME})

add_executable(ttnn-standalone ttnn-standalone.cpp)
set_property(TARGET ttnn-standalone PROPERTY CXX_STANDARD 20)

target_include_directories(ttnn-standalone PRIVATE
    # reflect
    $ENV{TT_METAL_HOME}/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f

    # Metalium
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/fmt
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
    # $ENV{TT_METAL_HOME}/tt_metal/hw/inc/$ENV{ARCH_NAME}
    # $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole_b0
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc
    # $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/$ENV{ARCH_NAME}
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/wormhole
    $ENV{TT_METAL_HOME}/tt_metal/third_party/magic_enum

    # TTNN
    $ENV{TT_METAL_HOME}/ttnn/cpp
    $ENV{TT_METAL_HOME}/ttnn/cpp/ttnn
    $ENV{TT_METAL_HOME}/ttnn/cpp/ttnn/deprecated
)

target_link_directories(ttnn-standalone PRIVATE
    $ENV{TT_METAL_HOME}-build/lib
)

target_link_libraries(ttnn-standalone PRIVATE
    # Metalium
    tt_metal
    yaml-cpp
    pthread

    # TTNN
    # _ttnn  # Why doesn't this work?
    $ENV{TT_METAL_HOME}-build/lib/_ttnn.so
)

target_compile_definitions(ttnn-standalone PRIVATE
    FMT_HEADER_ONLY
)

target_precompile_headers(ttnn-standalone PRIVATE pch.hpp)
