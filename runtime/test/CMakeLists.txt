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

add_library(TTRuntimeTEST INTERFACE)
add_dependencies(TTRuntimeTEST TTRuntimeTTNN TTRuntimeTTMetal TTRuntime)
target_include_directories(TTRuntimeTEST INTERFACE
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
    ${TTMLIR_TOOLCHAIN}/include
)
target_link_libraries(TTRuntimeTEST INTERFACE
    tt_eager
    tt_metal
    TTRuntime
    TTRuntimeTTNN
    TTRuntimeTTMetal
    ${Python3_LIBRARIES}
    ${FLATBUFFERS_LIB}
    GTest::gtest_main
)

function(add_runtime_gtest test_name)
  add_executable(${test_name} ${ARGN})
  add_dependencies(${test_name} TTRuntimeTEST)
  target_link_libraries(${test_name} PRIVATE TTRuntimeTEST)
  gtest_discover_tests(${test_name})
endfunction()

add_subdirectory(ttnn)
add_subdirectory(ttmetal)
