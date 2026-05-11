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
)

target_link_libraries(tt_kurbla_unit_tests
    PRIVATE
        tt_kurbla::tt_kurbla
        tt_kurbla_warnings
        tt_kurbla_sanitizers
        GTest::gtest_main
)

# When ttsim is enabled (see cmake/Ttsim.cmake), this TU sets TT_METAL_SIMULATOR
# + friends at static-init time so the runtime opens ttsim instead of a device.
if(TT_KURBLA_ENABLE_SIMULATOR)
    target_sources(tt_kurbla_unit_tests PRIVATE sim_test_env.cpp)
    target_compile_definitions(tt_kurbla_unit_tests PRIVATE
        TT_KURBLA_SIM_DIR="${TT_KURBLA_SIM_DIR}"
        TT_KURBLA_TT_METAL_HOME="${TT_KURBLA_TT_METAL_HOME}"
    )
endif()

gtest_discover_tests(tt_kurbla_unit_tests)
