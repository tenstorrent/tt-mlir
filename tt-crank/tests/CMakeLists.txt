find_package(GTest)

# Don't run clang-tidy on gtest's own translation units when TT_CRANK_ENABLE_CLANG_TIDY=ON.
foreach(_gtest_tgt gtest gmock gtest_main gmock_main)
    if(TARGET ${_gtest_tgt})
        set_target_properties(${_gtest_tgt} PROPERTIES CXX_CLANG_TIDY "")
    endif()
endforeach()

add_executable(tt_crank_unit_tests
    version_test.cpp
    engine_compile_test.cpp
    engine_module_builder_test.cpp
    engine_execution_payload_test.cpp
    device_teardown_env.cpp
)

target_link_libraries(tt_crank_unit_tests
    PRIVATE
        tt_crank::tt_crank
        tt_crank_warnings
        tt_crank_common
        tt_crank_sanitizers
        GTest::gtest_main
)

add_test(NAME tt_crank_unit_tests COMMAND tt_crank_unit_tests)
