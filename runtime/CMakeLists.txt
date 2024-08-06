# Options
option(TTMLIR_ENABLE_RUNTIME_TESTS "Enable runtime tests" OFF)
option(TT_RUNTIME_ENABLE_TTNN "Enable TTNN Runtime" OFF)
option(TT_RUNTIME_ENABLE_TTMETAL "Enable TTMetal Runtime" OFF)

if (NOT TT_RUNTIME_ENABLE_TTNN AND NOT TT_RUNTIME_ENABLE_TTMETAL)
  # Default to TTNN
  set(TT_RUNTIME_ENABLE_TTNN ON)
endif()

add_subdirectory(lib)
add_subdirectory(tools)
if (TTMLIR_ENABLE_RUNTIME_TESTS)
    add_subdirectory(test)
endif()
