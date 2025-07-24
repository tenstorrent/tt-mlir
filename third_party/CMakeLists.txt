include(FetchContent)

set(TT_METAL_VERSION "d78b2f9ae3d33c468d71c99ee5e8ab7f3606c89d")

if(TARGET pybind11::pybind11_headers)
  message(FATAL_ERROR "pybind11::pybind11_headers already exists, TODO: find a way to skip CPM")
endif()

set(ENABLE_TRACY ${TT_RUNTIME_ENABLE_PERF_TRACE})
set(ENABLE_DISTRIBUTED ${TT_RUNTIME_ENABLE_DISTRIBUTED})
set(TT_INSTALL ON)

FetchContent_Declare(
  tt-metal
  GIT_REPOSITORY https://github.com/tenstorrent/tt-metal.git
  GIT_TAG ${TT_METAL_VERSION}
  SYSTEM
)

message(STATUS "Downloading tt-metal...")
FetchContent_MakeAvailable(tt-metal)

install(CODE "
set(CMAKE_BINARY_DIR \"${CMAKE_BINARY_DIR}\")
message(STATUS \"Installing metalium dependencies for SharedLib component...\")
execute_process(
  COMMAND \${CMAKE_COMMAND} --install \${CMAKE_BINARY_DIR} --prefix \${CMAKE_INSTALL_PREFIX} --component ttnn-runtime
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND \${CMAKE_COMMAND} --install \${CMAKE_BINARY_DIR} --prefix \${CMAKE_INSTALL_PREFIX} --component tracy
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND \${CMAKE_COMMAND} --install \${CMAKE_BINARY_DIR} --prefix \${CMAKE_INSTALL_PREFIX} --component metalium-runtime
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND \${CMAKE_COMMAND} --install \${CMAKE_BINARY_DIR} --prefix \${CMAKE_INSTALL_PREFIX} --component umd-runtime
  COMMAND_ERROR_IS_FATAL ANY
)
" COMPONENT SharedLib)
