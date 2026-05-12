include(FetchContent)

set(INSTALL_GTEST OFF CACHE BOOL "Disable installing GoogleTest as part of tt-kurbla")

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

# Don't run clang-tidy on gtest's own translation units when TT_KURBLA_ENABLE_CLANG_TIDY=ON.
foreach(_gtest_tgt gtest gmock gtest_main gmock_main)
    if(TARGET ${_gtest_tgt})
        set_target_properties(${_gtest_tgt} PROPERTIES CXX_CLANG_TIDY "")
    endif()
endforeach()

include(GoogleTest)

add_executable(tt_kurbla_unit_tests
    version_test.cpp
    engine_compile_test.cpp
    engine_execution_payload_test.cpp
    # Static-init env shim — inert unless TT_KURBLA_USE_SIMULATOR is set at
    # runtime. Always linked so any test invocation can flip into ttsim mode
    # via the env var (e.g., `ctest --preset sim`).
    sim_test_env.cpp
)

target_link_libraries(tt_kurbla_unit_tests
    PRIVATE
        tt_kurbla::tt_kurbla
        tt_kurbla_warnings
        tt_kurbla_sanitizers
        GTest::gtest_main
)

target_compile_definitions(tt_kurbla_unit_tests PRIVATE
    TT_KURBLA_SIM_DIR="${TT_KURBLA_SIM_DIR}"
    TT_KURBLA_TT_METAL_HOME="${TT_KURBLA_TT_METAL_HOME}"
)

# The SoC descriptor under TT_KURBLA_SIM_DIR is copied at build time (see
# cmake/Ttsim.cmake); make sure that's run before tests can execute.
add_dependencies(tt_kurbla_unit_tests tt_kurbla_ttsim_stage)

gtest_discover_tests(tt_kurbla_unit_tests)
