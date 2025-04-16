option(TTNN_STANDALONE_USE_INSTALL_PATHS "Use install paths" OFF)

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

# Find clang 17
#
find_program(CLANG_17 clang++-17)
if(CLANG_17)
    message(STATUS "Found Clang-17 here: ${CLANG_17}")
    set(CMAKE_CXX_COMPILER "${CLANG_17}")
else()
    message(WARNING "Clang++-17 not found!!!")
endif()

# This tool can be compiled in two ways:
# 1. Local build (default)
# 2. Using install paths - initially used for frontends like FFE
if (NOT ${TTNN_STANDALONE_USE_INSTALL_PATHS} STREQUAL ON)
    # LOCAL BUILD
    message(STATUS "Building locally")

    # Check if env TT_METAL_HOME is set
    if(NOT DEFINED ENV{TT_METAL_HOME})
        message(FATAL_ERROR "TT_METAL_HOME is not set")
    endif()

    set(METAL_ROOT_DIR "$ENV{TT_METAL_HOME}")
    set(METAL_LIBS_DIR "$ENV{TT_METAL_HOME}-build/lib")
else()
    # INSTALL PATH BUILD
    message(STATUS "Building using install paths")

    if (NOT DEFINED CMAKE_INSTALL_PREFIX)
        message(FATAL_ERROR "CMAKE_INSTALL_PREFIX is not set. Aborting!")
    endif()

    # message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

    set(METAL_ROOT_DIR "${CMAKE_INSTALL_PREFIX}/tt-metal")
    set(METAL_LIBS_DIR "${CMAKE_INSTALL_PREFIX}/lib")
endif()
message(STATUS "METAL_ROOT_DIR: ${METAL_ROOT_DIR}")
message(STATUS "METAL_LIBS_DIR: ${METAL_LIBS_DIR}")

# Directories to search for headers
#
set(INCLUDE_DIRS
    # TODO: Remove these when ttmetal removes the dependencies from public facing headers
    ${METAL_ROOT_DIR}/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f
    ${METAL_ROOT_DIR}/.cpmcache/fmt/73b5ec45edbd92babfd91c3777a9e1ab9cac8238/include
    ${METAL_ROOT_DIR}/.cpmcache/magic_enum/4d76fe0a5b27a0e62d6c15976d02b33c54207096/include
    ${METAL_ROOT_DIR}/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/core/include
    ${METAL_ROOT_DIR}/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include
    ${METAL_ROOT_DIR}/.cpmcache/xtensor/4a957e26c765b48cbec4a4235fe9e518d5a85d3d/include
    ${METAL_ROOT_DIR}/.cpmcache/xtensor-blas/190c3a4314355b67291a7d78b20a2100de3f8f54/include
    ${METAL_ROOT_DIR}/.cpmcache/xtl/0918808959d33a292c551b9f014a0e808bc4a95c/include

    # Metalium
    ${METAL_ROOT_DIR}
    ${METAL_ROOT_DIR}/tt_metal
    ${METAL_ROOT_DIR}/tt_metal/api
    ${METAL_ROOT_DIR}/tt_metal/hostdevcommon/api
    ${METAL_ROOT_DIR}/tt_metal/hw/inc
    ${METAL_ROOT_DIR}/tt_metal/hw/inc/grayskull
    ${METAL_ROOT_DIR}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
    ${METAL_ROOT_DIR}/tt_metal/hw/inc/wormhole
    ${METAL_ROOT_DIR}/tt_metal/hw/inc/blackhole
    ${METAL_ROOT_DIR}/tt_metal/include
    ${METAL_ROOT_DIR}/tt_stl
    ${METAL_ROOT_DIR}/tt_stl/tt_stl
    ${METAL_ROOT_DIR}/tt_metal/third_party/fmt
    ${METAL_ROOT_DIR}/tt_metal/third_party/magic_enum
    ${METAL_ROOT_DIR}/tt_metal/third_party/taskflow
    ${METAL_ROOT_DIR}/tt_metal/third_party/tracy/public
    ${METAL_ROOT_DIR}/tt_metal/third_party/umd
    ${METAL_ROOT_DIR}/tt_metal/third_party/umd/device/api
    ${METAL_ROOT_DIR}/tt_metal/third_party/umd/src/firmware/riscv/grayskull
    ${METAL_ROOT_DIR}/tt_metal/third_party/umd/src/firmware/riscv/wormhole
    ${METAL_ROOT_DIR}/tt_metal/third_party/umd/src/firmware/riscv/blackhole

    # TTNN
    ${METAL_ROOT_DIR}/ttnn
    ${METAL_ROOT_DIR}/ttnn/cpp
    ${METAL_ROOT_DIR}/ttnn/cpp/ttnn
    ${METAL_ROOT_DIR}/ttnn/cpp/ttnn/deprecated

    # TT_MLIR Runtime
    $ENV{TT_MLIR_HOME}/runtime/lib/ttnn
)

# Link directories
#
set(LINK_DIRS
    ${METAL_LIBS_DIR}
)

# Libraries to link
#
set(LINK_LIBS
    tt_metal
    device

    # TTNN
    # _ttnn  # Why doesn't this work?
    ${METAL_LIBS_DIR}/_ttnn.so
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

# Test compile_so
add_executable(test_compile_so
    test_compile_so.cpp
    compile_so.cpp
)
