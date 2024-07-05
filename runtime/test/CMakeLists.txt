# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

function(add_runtime_gtest test_name)
  add_executable(${test_name} ${ARGN})
  target_link_libraries(${test_name} PRIVATE GTest::gtest_main)
  gtest_discover_tests(${test_name})
endfunction()

add_subdirectory(ttnn)
add_subdirectory(ttmetal)

