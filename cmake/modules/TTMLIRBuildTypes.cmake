# Asan build type
set(CMAKE_C_FLAGS_ASAN
    "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer -shared-libasan" CACHE STRING
    "Flags used by the C compiler for Asan build type or configuration." FORCE)
set(CMAKE_CXX_FLAGS_ASAN
    "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer -shared-libasan" CACHE STRING
    "Flags used by the C++ compiler for Asan build type or configuration." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_ASAN
    "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address -shared-libasan" CACHE STRING
    "Linker flags to be used to create executables for Asan build type." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_ASAN
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address -shared-libasan" CACHE STRING
    "Linker lags to be used to create shared libraries for Asan build type." FORCE)

# Code coverage build type
set(CMAKE_C_FLAGS_COVERAGE
    "${CMAKE_C_FLAGS_DEBUG} --coverage" CACHE STRING
    "Flags used by the C compiler for code coverage build type or configuration." FORCE)
set(CMAKE_CXX_FLAGS_COVERAGE
    "${CMAKE_CXX_FLAGS_DEBUG} --coverage" CACHE STRING
    "Flags used by the C++ compiler for code coverage build type or configuration." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE
    "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage" CACHE STRING
    "Linker flags to be used to create executables for code coverage build type." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} --coverage" CACHE STRING
    "Linker lags to be used to create shared libraries for code coverage build type." FORCE)

  # Assert build type
set(CMAKE_C_FLAGS_ASSERT
  "${CMAKE_C_FLAGS_RELEASE}" CACHE STRING
    "Flags used by the C compiler for Assert build type or configuration." FORCE)
set(CMAKE_CXX_FLAGS_ASSERT
  "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING
    "Flags used by the C++ compiler for Assert build type or configuration." FORCE)
