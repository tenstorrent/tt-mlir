if (TTMLIR_ENABLE_OPMODEL)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(TestOpModelLib
TestOpModelLib.cpp
)

target_compile_options(TestOpModelLib
    PRIVATE
    -fno-rtti
)

target_include_directories(TestOpModelLib
    PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
    ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN/
)

target_link_libraries(TestOpModelLib
    PRIVATE
    gtest
    gtest_main
    TTNNOpModelLib
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTTransforms
)

endif()
