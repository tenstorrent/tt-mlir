include(FetchContent)

set(TT_METAL_VERSION "98c764028d2001d60f66eabcb0311c248bc7006b")

if(TARGET pybind11::pybind11_headers)
  message(FATAL_ERROR "pybind11::pybind11_headers already exists, TODO: find a way to skip CPM")
endif()

set(ENABLE_TRACY ${TT_RUNTIME_ENABLE_PERF_TRACE})
set(ENABLE_DISTRIBUTED ${TT_RUNTIME_ENABLE_DISTRIBUTED})
set(TT_INSTALL ON)

if (TTMLIR_ENABLE_RUNTIME)
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
else()
  add_library(TTMetaliumRuntimeDisabled INTERFACE)
  add_library(TTNN::CPP ALIAS TTMetaliumRuntimeDisabled)
  add_library(TTNN::PYTHON ALIAS TTMetaliumRuntimeDisabled)
  add_library(TT::Metalium ALIAS TTMetaliumRuntimeDisabled)
endif()
