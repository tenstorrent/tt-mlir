cmake_minimum_required(VERSION 3.16)
project(ttnn-standalone CXX)

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

# Add local ttnn-install to CMAKE_PREFIX_PATH if it exists
set(TTNN_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttnn-install")
if(EXISTS ${TTNN_INSTALL_DIR})
    list(APPEND CMAKE_PREFIX_PATH ${TTNN_INSTALL_DIR})
endif()

# Find TT-NN package (required for standalone mode)
find_package(xtensor REQUIRED)  # TODO: remove once fixed in metal
find_package(TT-NN REQUIRED)

# All dependencies are handled by the TT-NN package

#       _                     _         _
#  ___ | |_  __ _  _ __    __| |  __ _ | |  ___   _ __    ___
# / __|| __|/ _` || '_ \  / _` | / _` || | / _ \ | '_ \  / _ \
# \__ \| |_| (_| || | | || (_| || (_| || || (_) || | | ||  __/
# |___/ \__|\__,_||_| |_| \__,_| \__,_||_| \___/ |_| |_| \___|

add_executable(ttnn-standalone ttnn-standalone.cpp)
set_property(TARGET ttnn-standalone PROPERTY CXX_STANDARD 20)
set_property(TARGET ttnn-standalone PROPERTY EXCLUDE_FROM_ALL TRUE)

# Link with TTNN which provides all necessary dependencies
target_link_libraries(ttnn-standalone PRIVATE TTNN::TTNN)

target_precompile_headers(ttnn-standalone PRIVATE ttnn-precompiled.hpp)
