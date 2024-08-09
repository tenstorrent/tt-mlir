# Options
option(TTMLIR_ENABLE_RUNTIME_TESTS "Enable runtime tests" OFF)
option(TT_RUNTIME_ENABLE_TTNN "Enable TTNN Runtime" ON)
option(TT_RUNTIME_ENABLE_TTMETAL "Enable TTMetal Runtime" ON)

add_subdirectory(lib)
add_subdirectory(tools)
if (TTMLIR_ENABLE_RUNTIME_TESTS)
    add_subdirectory(test)
endif()
