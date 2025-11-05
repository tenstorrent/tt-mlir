cmake_minimum_required(VERSION 3.20)
project(TestTTMLIRInstall C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TTMLIR REQUIRED CONFIG HINTS ${TTMLIR_DIR})
if(NOT TTMLIR_FOUND)
  message(FATAL_ERROR "TTMLIR not found")
endif()

message(STATUS "TTMLIR found in: ${TTMLIR_DIR}")
message(STATUS "TTMLIR_DIR: ${TTMLIR_DIR}")
message(STATUS "TTMLIR_INCLUDE_DIRS: ${TTMLIR_INCLUDE_DIRS}")
message(STATUS "TTMLIR_LIBRARY_DIRS: ${TTMLIR_LIBRARY_DIRS}")
message(STATUS "TTMLIR_TOOLS_DIR: ${TTMLIR_TOOLS_DIR}")
add_executable(test_install test_install.cpp)

# Link against a basic TTMLIR library that has no StableHLO dependencies.
target_link_libraries(test_install PRIVATE
  TTMLIR::MLIRTTCoreDialect
  MLIRFuncDialect
)
