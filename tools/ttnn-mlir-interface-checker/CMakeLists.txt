cmake_minimum_required(VERSION 3.16)
project(ttnn-mlir-interface-checker CXX)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the source files for the application
set(SOURCES
    ttnn-mlir-interface-checker.cpp
)

# Add an executable target
add_executable(${PROJECT_NAME} ${SOURCES})



target_link_libraries(${PROJECT_NAME} PRIVATE TTMETAL_LIBRARY TTNN_LIBRARY TTNN_MLIR_INTERFACE_LIBRARY)

# Get the public header directory of the MLIR interface library
get_target_property(MLIR_INTERFACE_PUBLIC_HEADER TTNN_MLIR_INTERFACE_LIBRARY PUBLIC_HEADER)
get_filename_component(MLIR_INTERFACE_PUBLIC_HEADER_INCLUDE ${MLIR_INTERFACE_PUBLIC_HEADER} DIRECTORY)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${MLIR_INTERFACE_PUBLIC_HEADER_INCLUDE}
)
