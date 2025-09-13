if (TTMLIR_ENABLE_OPMODEL)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TestConversion is used to test MLIR to TTNN types conversion
add_executable(TestConversion
    TestConversion.cpp
)

target_compile_options(TestConversion
    PRIVATE
)

target_include_directories(TestConversion
    PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
    ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN/
)

target_link_libraries(TestConversion
    PRIVATE
    gtest
    gtest_main
    TTNNOpModelLib
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTTransforms
)

endif()
