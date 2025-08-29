cmake_minimum_required(VERSION 3.16)
project(ttnn-standalone CXX)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(TTNNCompileSo SHARED compile_so.cpp)
set_target_properties(TTNNCompileSo PROPERTIES
    PUBLIC_HEADER compile_so.hpp
)

install(
  TARGETS TTNNCompileSo
  COMPONENT TTNNStandalone
  PUBLIC_HEADER DESTINATION include/ttmlir/tools/ttnn-standalone
)

install(
  PROGRAMS
    run
    ci_compile_dylib.py
  DESTINATION tools/ttnn-standalone
  COMPONENT TTNNStandalone
)

install(
  FILES
    CMakeLists.txt
    README.md
    ttnn-precompiled.hpp
    ttnn-standalone.cpp
    workarounds.hpp
    compile_so.hpp
    compile_so.cpp
  DESTINATION tools/ttnn-standalone
  COMPONENT TTNNStandalone
)

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

# TT_METAL_HOME env variable needs to be set
if(NOT DEFINED ENV{TT_METAL_HOME})
    message(WARNING "TT_METAL_HOME is not set in ttnn-standalone. Skipping build.")
    return()
endif()

# Metal dir variables
set(METAL_SRC_DIR "$ENV{TT_METAL_HOME}" CACHE PATH "Path to the tt-metal root src directory")
if(DEFINED ENV{TT_METAL_LIB})
  set(METAL_LIB_DIR "$ENV{TT_METAL_LIB}" CACHE PATH "Path to the tt-metal lib directory")
else()
  set(METAL_LIB_DIR "$ENV{TT_METAL_HOME}/build/lib" CACHE PATH "Path to the tt-metal lib directory")
endif()

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
    ${CPM_SOURCE_CACHE}/tt-logger/48ec77b63c35cf84c47678990aa3603b492d0c7c/include

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
    tt_stl
    ${METAL_LIB_DIR}/_ttnncpp.so
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
set_property(TARGET ttnn-standalone PROPERTY EXCLUDE_FROM_ALL TRUE)
set_target_properties(ttnn-standalone PROPERTIES INSTALL_RPATH "$ORIGIN:${METAL_LIB_DIR}")
set_target_properties(ttnn-standalone PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

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
set_property(TARGET ttnn-dylib PROPERTY EXCLUDE_FROM_ALL TRUE)

target_include_directories(ttnn-dylib PRIVATE ${INCLUDE_DIRS})
target_link_directories(ttnn-dylib PRIVATE ${LINK_DIRS})
target_link_libraries(ttnn-dylib PRIVATE ${LINK_LIBS})
set_target_properties(ttnn-dylib PROPERTIES PUBLIC_HEADER ttnn-dylib.h)
set_target_properties(ttnn-dylib PROPERTIES INSTALL_RPATH "$ORIGIN:${METAL_LIB_DIR}")
set_target_properties(ttnn-dylib PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  target_compile_options(ttnn-dylib PRIVATE -march=x86-64-v3)
endif()

target_precompile_headers(ttnn-dylib PRIVATE ttnn-precompiled.hpp)
