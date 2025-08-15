if ((NOT TTMLIR_ENABLE_RUNTIME OR (NOT TT_RUNTIME_ENABLE_TTNN AND NOT TT_RUNTIME_ENABLE_TTMETAL)) AND NOT TT_RUNTIME_ENABLE_CUDA)
  message(FATAL_ERROR "Runtime tests require at least one backend runtime to be enabled")
endif()

enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(gtest PRIVATE -Wno-covered-switch-default)
endif()

include(GoogleTest)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
if (NOT Python3_LIBRARIES)
  message(FATAL_ERROR "python libraries not found")
endif()

find_library(FLATBUFFERS_LIB flatbuffers PATHS ${TTMLIR_TOOLCHAIN_DIR}/lib)
if (NOT FLATBUFFERS_LIB)
  message(FATAL_ERROR "flatbuffers library not found")
endif()

add_library(TTRuntimeGTestLib INTERFACE)
target_include_directories(TTRuntimeGTestLib INTERFACE
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
    ${TTMLIR_TOOLCHAIN}/include
)
target_include_directories(TTRuntimeGTestLib SYSTEM INTERFACE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

target_link_libraries(TTRuntimeGTestLib INTERFACE
    TTBinary
    TTRuntimeTypes
    TTRuntimeUtils
    TTRuntimeSocket
    TTMLIRRuntime
    TTRuntimeDistributedController
    TTRuntimeDistributedWorker
    TTNN_LIBRARY
    TTMETAL_LIBRARY
    DEVICE_LIBRARY
    TT_STL_LIBRARY
    ${Python3_LIBRARIES}
    ${FLATBUFFERS_LIB}
    GTest::gtest_main
)

target_link_libraries(TTRuntimeGTestLib INTERFACE TRACY_LIBRARY)


function(add_runtime_gtest test_name)
  if(ARGC EQUAL 1)
    set(source_files ${test_name}.cpp)
  else()
    set(source_files ${ARGN})
  endif()

  add_executable(${test_name} ${source_files})
  set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 20)
  add_dependencies(${test_name} TTRuntimeGTestLib)
  target_link_libraries(${test_name} PRIVATE TTRuntimeGTestLib)
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

function(add_cuda_runtime_gtest test_name)
  if(ARGC EQUAL 1)
    set(source_files ${test_name}.cpp)
  else()
    set(source_files ${ARGN})
  endif()

  add_executable(${test_name} ${source_files})
  set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 20)
  target_include_directories(${test_name} PRIVATE
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/CUDA
    ${TTMLIR_TOOLCHAIN}/include
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  add_dependencies(${test_name} TTRuntimeCUDA)
  target_link_libraries(${test_name} PRIVATE
    TTRuntimeCUDA
    TTBinary
    TTMLIRRuntime
    GTest::gtest_main
  )

  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_subdirectory(common)

if (TT_RUNTIME_ENABLE_TTNN)
  add_subdirectory(ttnn)
endif()

if (TT_RUNTIME_ENABLE_TTMETAL)
  add_subdirectory(ttmetal)
endif()

if (TT_RUNTIME_ENABLE_CUDA)
  find_package(CUDAToolkit)
  if (CUDAToolkit_FOUND)
    add_subdirectory(cuda)
  endif()
endif()
