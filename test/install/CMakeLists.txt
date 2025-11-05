cmake_minimum_required(VERSION 3.20)
project(TestTTMLIRInstall)

# Allow specifying TTMLIR install location via TTMLIR_DIR or CMAKE_PREFIX_PATH
if(TTMLIR_DIR)
  set(CMAKE_PREFIX_PATH ${TTMLIR_DIR} ${CMAKE_PREFIX_PATH})
endif()

find_package(TTMLIR REQUIRED CONFIG)

add_executable(test_install test_install.cpp)
target_link_libraries(test_install PRIVATE
  TTMLIR::TTMLIRCompilerStatic
)
