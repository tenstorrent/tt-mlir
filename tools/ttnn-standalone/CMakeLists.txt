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

# Link directories - use standardized directory from third_party/CMakeLists.txt
#
set(LINK_DIRS
    ${TTMETAL_LIBRARY_DIR}
)

# Use standardized library targets from third_party/CMakeLists.txt
set(TTNN_LIBS TTMETAL_LIBRARY DEVICE_LIBRARY TTNN_LIBRARY TT_STL_LIBRARY TRACY_LIBRARY)

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
set_target_properties(ttnn-standalone PROPERTIES INSTALL_RPATH "$ORIGIN:${TTMETAL_LIBRARY_DIR}")
set_target_properties(ttnn-standalone PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

# Add dependency on tt-metal external project
add_dependencies(ttnn-standalone tt-metal)

# Debug: Check if the required header file exists
set(BFLOAT16_HEADER "${CMAKE_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/api/tt-metalium/bfloat16.hpp")
if(EXISTS "${BFLOAT16_HEADER}")
    message(STATUS "Found bfloat16.hpp at: ${BFLOAT16_HEADER}")
else()
    message(WARNING "bfloat16.hpp not found at: ${BFLOAT16_HEADER}")
endif()

target_include_directories(ttnn-standalone SYSTEM PRIVATE
    "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>"
    "${CMAKE_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/api"
)
target_link_directories(ttnn-standalone PRIVATE ${LINK_DIRS})
target_link_libraries(ttnn-standalone PRIVATE ${TTNN_LIBS})
target_compile_definitions(ttnn-standalone PRIVATE ${COMPILE_DEFS})

target_precompile_headers(ttnn-standalone PRIVATE ttnn-precompiled.hpp)

# Install the ttnn-standalone executable
install(
  TARGETS ttnn-standalone
  DESTINATION tools/ttnn-standalone
  COMPONENT TTNNStandalone
)

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

target_include_directories(ttnn-dylib SYSTEM PRIVATE
    "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>"
    "${CMAKE_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/api"
)
target_link_directories(ttnn-dylib PRIVATE ${LINK_DIRS})
target_link_libraries(ttnn-dylib PRIVATE ${TTNN_LIBS})
set_target_properties(ttnn-dylib PROPERTIES PUBLIC_HEADER ttnn-dylib.h)
set_target_properties(ttnn-dylib PROPERTIES INSTALL_RPATH "$ORIGIN:${TTMETAL_LIBRARY_DIR}")
set_target_properties(ttnn-dylib PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  target_compile_options(ttnn-dylib PRIVATE -march=x86-64-v3)
endif()

target_precompile_headers(ttnn-dylib PRIVATE ttnn-precompiled.hpp)
