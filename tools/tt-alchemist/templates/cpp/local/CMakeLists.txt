cmake_minimum_required(VERSION 3.16)
project(ttnn-local CXX)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(DEFINED ENV{CONFIG})
    message(STATUS "CONFIG is set, CMAKE_BUILD_TYPE being set to $ENV{CONFIG}")
    set(CMAKE_BUILD_TYPE $ENV{CONFIG})
elseif(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release build is the default" FORCE)
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(WARNING "Not using the Clang compiler!")
else()
  message(STATUS "Found Clang-${CLANG_VERSION_MAJOR} here: ${CMAKE_CXX_COMPILER}")
  if(CLANG_VERSION_MAJOR LESS 17)
    message(WARNING "Clang version is less than 17!")
  endif()
endif()

#############################
# Set env pointer variables #
#############################

# TT_METAL_HOME env variable needs to be set
#
if(NOT DEFINED ENV{TT_METAL_HOME})
    message(WARNING "TT_METAL_HOME is not set. Skipping build.")
    return()
endif()

###########################
# Calculate env variables #
###########################

# Metal src dir and lib dir
#
set(METAL_SRC_DIR "$ENV{TT_METAL_HOME}" CACHE PATH "Path to the tt-metal root src directory")
if(DEFINED ENV{TT_METAL_LIB})
  set(METAL_LIB_DIR "$ENV{TT_METAL_LIB}" CACHE PATH "Path to the tt-metal lib directory")
else()
  set(METAL_LIB_DIR "$ENV{TT_METAL_HOME}/build/lib" CACHE PATH "Path to the tt-metal lib directory")
endif()

# CPM Cache
#
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
    ${CPM_SOURCE_CACHE}/reflect/f93e77475670eaeacf332927dfe8b50e3f3812e0
    ${CPM_SOURCE_CACHE}/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6/include
    ${CPM_SOURCE_CACHE}/enchantum/2fb7ab238e36c101b9848892ddb6382276b65837/enchantum/include
    ${CPM_SOURCE_CACHE}/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/core/include
    ${CPM_SOURCE_CACHE}/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include
    ${CPM_SOURCE_CACHE}/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include
    ${CPM_SOURCE_CACHE}/tt-logger/5d45c661affa374c5656335a77010ab503f96a99/include

    # Metalium
    ${METAL_SRC_DIR}
    ${METAL_SRC_DIR}/tt_metal
    ${METAL_SRC_DIR}/tt_metal/api
    ${METAL_SRC_DIR}/tt_metal/hostdevcommon/api
    ${METAL_SRC_DIR}/tt_metal/hw/inc
    ${METAL_SRC_DIR}/tt_metal/hw/inc/grayskull
    ${METAL_SRC_DIR}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
    ${METAL_SRC_DIR}/tt_metal/hw/inc/wormhole
    ${METAL_SRC_DIR}/tt_metal/hw/inc/blackhole
    ${METAL_SRC_DIR}/tt_metal/include
    ${METAL_SRC_DIR}/tt_stl
    ${METAL_SRC_DIR}/tt_stl/tt_stl
    ${METAL_SRC_DIR}/tt_metal/third_party/fmt
    ${METAL_SRC_DIR}/tt_metal/third_party/enchantum
    ${METAL_SRC_DIR}/tt_metal/third_party/taskflow
    ${METAL_SRC_DIR}/tt_metal/third_party/tracy/public
    ${METAL_SRC_DIR}/tt_metal/third_party/umd
    ${METAL_SRC_DIR}/tt_metal/third_party/umd/device/api
    ${METAL_SRC_DIR}/tt_metal/third_party/umd/src/firmware/riscv/grayskull
    ${METAL_SRC_DIR}/tt_metal/third_party/umd/src/firmware/riscv/wormhole
    ${METAL_SRC_DIR}/tt_metal/third_party/umd/src/firmware/riscv/blackhole

    # TTNN
    ${METAL_SRC_DIR}/ttnn
    ${METAL_SRC_DIR}/ttnn/api
    ${METAL_SRC_DIR}/ttnn/core
    ${METAL_SRC_DIR}/ttnn/cpp
    ${METAL_SRC_DIR}/ttnn/cpp/ttnn
)

# Link directories
#
set(LINK_DIRS
    ${METAL_LIB_DIR}
)

# Libraries to link
#
set(LINK_LIBS
    tt_metal
    device
    ${METAL_LIB_DIR}/_ttnncpp.so
)

# Compile definitions
#
set(COMPILE_DEFS
    FMT_HEADER_ONLY  # needed for fmt lib
)

#  _                 _
# | | ___   ___ __ _| |
# | |/ _ \ / __/ _` | |
# | | (_) | (_| (_| | |
# |_|\___/ \___\__,_|_|

add_executable(ttnn-local ttnn-local.cpp)
set_property(TARGET ttnn-local PROPERTY CXX_STANDARD 20)
set_property(TARGET ttnn-local PROPERTY EXCLUDE_FROM_ALL TRUE)

target_include_directories(ttnn-local PRIVATE ${INCLUDE_DIRS})
target_link_directories(ttnn-local PRIVATE ${LINK_DIRS})
target_link_libraries(ttnn-local PRIVATE ${LINK_LIBS})
target_compile_definitions(ttnn-local PRIVATE ${COMPILE_DEFS})

target_precompile_headers(ttnn-local PRIVATE ttnn-precompiled.hpp)
