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
    TTMLIRTestUtils
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
    TTMLIRTestUtils
)

# Tripwire tests: small, standalone, expected to fail when a specific upstream
# tt-metal change lands.  Built as its own target so it compiles fast and the
# whole file/target can be removed once the upstream change arrives.
add_executable(TestOpModelLibTripwires
TestOpModelLibTripwires.cpp
)

target_compile_options(TestOpModelLibTripwires
    PRIVATE
    -fno-rtti
)

target_include_directories(TestOpModelLibTripwires
    PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
    ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN/
)

target_link_libraries(TestOpModelLibTripwires
    PRIVATE
    gtest
    gtest_main
    TTNNOpModelLib
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTTransforms
    TTMLIRTestUtils
)

endif()
