if (TTMLIR_ENABLE_OPMODEL)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(TestOpModelInterface
TestOpModelInterface.cpp
)

target_compile_options(TestOpModelInterface
    PRIVATE
    -fno-rtti
)

target_include_directories(TestOpModelInterface
    PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
    ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN/
)

target_link_libraries(TestOpModelInterface
    PRIVATE
    gtest
    gtest_main
    TTNNOpModelLib
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTTransforms
)

add_executable(TestMatmulBlockShardedConstraint
TestMatmulBlockShardedConstraint.cpp
)

target_compile_options(TestMatmulBlockShardedConstraint
    PRIVATE
    -fno-rtti
)

target_include_directories(TestMatmulBlockShardedConstraint
    PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
    ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN/
)

target_link_libraries(TestMatmulBlockShardedConstraint
    PRIVATE
    gtest
    gtest_main
    TTNNOpModelLib
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTTransforms
)

add_executable(TestRMSNormWidthSharded
TestRMSNormWidthSharded.cpp
)

target_compile_options(TestRMSNormWidthSharded
    PRIVATE
    -fno-rtti
)

target_include_directories(TestRMSNormWidthSharded
    PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
    ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN/
)

target_link_libraries(TestRMSNormWidthSharded
    PRIVATE
    gtest
    gtest_main
    TTNNOpModelLib
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTTransforms
)

add_executable(TestD2MOpCostModel
TestD2MOpCostModel.cpp
)

target_compile_options(TestD2MOpCostModel
    PRIVATE
    -fno-rtti
)

target_include_directories(TestD2MOpCostModel
    PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/OpModel/TTNN/
    ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN/
)

target_link_libraries(TestD2MOpCostModel
    PRIVATE
    gtest
    gtest_main
    MLIRTTNNInterfaces
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRTTTransforms
    TTMLIRTTNNUtils
)

endif()
