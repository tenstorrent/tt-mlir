# Options
option(TTMLIR_ENABLE_RUNTIME_TESTS "Enable runtime tests" OFF)

add_subdirectory(lib)
add_subdirectory(tools)
if (TTMLIR_ENABLE_RUNTIME_TESTS)
    add_subdirectory(test)
endif()
