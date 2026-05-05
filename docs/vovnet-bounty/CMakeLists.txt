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

# Reuse the ttnn-install from the v3 sibling directory.
set(TTNN_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vovnet_standalone_v3/ttnn-install")
if(EXISTS ${TTNN_INSTALL_DIR})
    list(APPEND CMAKE_PREFIX_PATH ${TTNN_INSTALL_DIR})
endif()

find_package(xtensor REQUIRED)  # TODO: remove once fixed in metal
find_package(TT-NN REQUIRED)

add_executable(ttnn-standalone
    main.cpp
    model_ttnn.cpp
    inputs.cpp
    consteval.cpp
    utils.cpp
)
set_property(TARGET ttnn-standalone PROPERTY CXX_STANDARD 20)
set_property(TARGET ttnn-standalone PROPERTY EXCLUDE_FROM_ALL TRUE)

target_link_libraries(ttnn-standalone PRIVATE TTNN::TTNN)

target_precompile_headers(ttnn-standalone PRIVATE ttnn-precompiled.hpp)
