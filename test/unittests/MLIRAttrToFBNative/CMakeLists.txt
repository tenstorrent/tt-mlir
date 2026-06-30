set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_mlir_unittest(ToNativeConsistencyTests
    ToNativeConsistency.cpp
    OpTPathParity.cpp
    PARTIAL_SOURCES_INTENDED
)

target_compile_options(ToNativeConsistencyTests
    PRIVATE
    -frtti
    -fexceptions
)

target_include_directories(ToNativeConsistencyTests
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../
    ${PROJECT_BINARY_DIR}/include/
)

target_link_libraries(ToNativeConsistencyTests
    PRIVATE
    gtest
    gtest_main
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTTransforms
    TTNNTargetFlatbuffer
    TTNNOpModelLib
)
