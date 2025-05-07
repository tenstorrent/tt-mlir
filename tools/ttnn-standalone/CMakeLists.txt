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
if("$ENV{TT_METAL_BUILD_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_BUILD_HOME is not set")
endif()
if("$ENV{ARCH_NAME}" STREQUAL "")
    message(FATAL_ERROR "ARCH_NAME is not set")
endif()

message($ENV{TT_METAL_HOME}/tt_metal/third_party/src/firmware/riscv/$ENV{ARCH_NAME})

if (DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_SOURCE_CACHE $ENV{CPM_SOURCE_CACHE})
else()
  set(CPM_SOURCE_CACHE $ENV{TT_METAL_HOME}/.cpmcache)
endif()
message(STATUS "Setting tt-metal CPM cache to: ${CPM_SOURCE_CACHE}")

# Directories to search for headers
#
set(INCLUDE_DIRS
    # TODO: Remove these when ttmetal removes the dependencies from public facing headers
    ${CPM_SOURCE_CACHE}/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f
    ${CPM_SOURCE_CACHE}/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6/include
    ${CPM_SOURCE_CACHE}/magic_enum/4d76fe0a5b27a0e62d6c15976d02b33c54207096/include
    ${CPM_SOURCE_CACHE}/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/core/include
    ${CPM_SOURCE_CACHE}/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include
    ${CPM_SOURCE_CACHE}/xtensor/c46fa4d34d719a7efc5f310a89f10bdf1e167ebc/include
    ${CPM_SOURCE_CACHE}/xtensor-blas/0047541801f16eb7eefdacfdf251c5f06c8f10d0/include
    ${CPM_SOURCE_CACHE}/xtl/47c2dd39d2d18cfa94b1db25e7dace936e428f78/include

    # Metalium
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/api
    $ENV{TT_METAL_HOME}/tt_metal/hostdevcommon/api
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/${ARCH_EXTRA_DIR}
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/${ARCH_NAME}
    $ENV{TT_METAL_HOME}/tt_metal/include
    $ENV{TT_METAL_HOME}/tt_stl
    $ENV{TT_METAL_HOME}/tt_stl/tt_stl
    $ENV{TT_METAL_HOME}/tt_metal/third_party/fmt
    $ENV{TT_METAL_HOME}/tt_metal/third_party/magic_enum
    $ENV{TT_METAL_HOME}/tt_metal/third_party/taskflow
    $ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/${ARCH_NAME}

    # TTNN
    $ENV{TT_METAL_HOME}/ttnn
    $ENV{TT_METAL_HOME}/ttnn/cpp
    $ENV{TT_METAL_HOME}/ttnn/cpp/ttnn
    $ENV{TT_METAL_HOME}/ttnn/cpp/ttnn/deprecated

    # TT_MLIR Runtime
    $ENV{TT_MLIR_HOME}/runtime/lib/ttnn
)

# Link directories
#
set(LINK_DIRS
    $ENV{TT_METAL_BUILD_HOME}/lib
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
    $ENV{TT_METAL_BUILD_HOME}/lib/_ttnn.so
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

if(NOT EXISTS "ttnn-dylib.cpp")
    file(TOUCH "ttnn-dylib.cpp")
endif()
add_library(ttnn-dylib SHARED ttnn-dylib.cpp)
set_property(TARGET ttnn-dylib PROPERTY CXX_STANDARD 20)

target_include_directories(ttnn-dylib PRIVATE ${INCLUDE_DIRS})
target_link_directories(ttnn-dylib PRIVATE ${LINK_DIRS})
target_link_libraries(ttnn-dylib PRIVATE ${LINK_LIBS})
set_target_properties(ttnn-dylib PROPERTIES PUBLIC_HEADER ttnn-dylib.h)
target_compile_options(ttnn-dylib PRIVATE -mavx -mavx2)

target_precompile_headers(ttnn-dylib PRIVATE ttnn-precompiled.hpp)
