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

# Directories to search for headers
#
set(INCLUDE_DIRS
    # TODO: Remove these when ttmetal removes the dependencies from public facing headers
    $ENV{TT_METAL_HOME}/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f
    $ENV{TT_METAL_HOME}/.cpmcache/fmt/73b5ec45edbd92babfd91c3777a9e1ab9cac8238/include
    $ENV{TT_METAL_HOME}/.cpmcache/magic_enum/4d76fe0a5b27a0e62d6c15976d02b33c54207096/include
    $ENV{TT_METAL_HOME}/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/core/include
    $ENV{TT_METAL_HOME}/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include
    $ENV{TT_METAL_HOME}/.cpmcache/xtensor/4a957e26c765b48cbec4a4235fe9e518d5a85d3d/include
    $ENV{TT_METAL_HOME}/.cpmcache/xtensor-blas/190c3a4314355b67291a7d78b20a2100de3f8f54/include
    $ENV{TT_METAL_HOME}/.cpmcache/xtl/0918808959d33a292c551b9f014a0e808bc4a95c/include

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
