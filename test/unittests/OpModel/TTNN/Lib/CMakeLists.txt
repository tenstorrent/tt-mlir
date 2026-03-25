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

add_executable(TestOpModelLibMockDevice
TestOpModelLibMockDevice.cpp
)

target_compile_options(TestOpModelLibMockDevice
    PRIVATE
    -fno-rtti
)

target_include_directories(TestOpModelLibMockDevice
    PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
    ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN/
)

target_link_libraries(TestOpModelLibMockDevice
    PRIVATE
    gtest
    TTNNOpModelLib
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTTransforms
)

add_executable(TestMockDeviceCrash TestMockDeviceCrash.cpp)
target_compile_options(TestMockDeviceCrash PRIVATE -fno-rtti)
target_include_directories(TestMockDeviceCrash PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
    ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN/
)
target_link_libraries(TestMockDeviceCrash PRIVATE
    gtest TTNNOpModelLib MLIRTTCoreDialect MLIRTTNNDialect MLIRTTTransforms
)

endif()
