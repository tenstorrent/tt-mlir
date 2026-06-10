add_mlir_unittest(OptimizerTests
    TestShardSolver.cpp
    TestOptimizerOverrides.cpp
    TestGreedyL1InterleavedPolicy.cpp
    TestLegalLayoutAnalysis.cpp
    TestLegalTensorLayoutAnalysis.cpp
    TestConv2dConfigGenerator.cpp
    TestConv3dConfigHeuristic.cpp
    PARTIAL_SOURCES_INTENDED
)

target_link_libraries(OptimizerTests
    PRIVATE
    MLIRTTNNDialect
    MLIRTTTransforms
    MLIRTTNNAnalysis
    MLIRTTNNValidation
)

if (TTMLIR_ENABLE_OPMODEL)
    add_mlir_unittest(OpModelStrategyTests
        TestOpModelStrategy.cpp
        TestBeamSearch.cpp
        TestReshardCandidates.cpp
        PARTIAL_SOURCES_INTENDED
    )

    target_link_libraries(OpModelStrategyTests
        PRIVATE
        MLIRTTNNDialect
        MLIRTTTransforms
        MLIRTTNNAnalysis
        MLIRTTNNValidation
    )
endif()

add_mlir_unittest(L1SpillManagementTests
    TestL1SpillManagement.cpp
    PARTIAL_SOURCES_INTENDED
)

target_link_libraries(L1SpillManagementTests
    PRIVATE
    MLIRTTNNDialect
    MLIRTTTransforms
    MLIRTTNNAnalysis
    MLIRTTNNValidation
)

if (TTMLIR_ENABLE_OPMODEL)
    # Allocator-backed stateful spill tests. Standalone binary (custom main()
    # registering MockDeviceEnvironment) rather than add_mlir_unittest, so it is
    # NOT wired into MLIRUnitTests/check-ttmlir -- it needs the op-model lib and
    # a mock device, matching the TestOpModelLibMockDevice convention. Runs
    # HW-free (mock device) on any box with the op-model build.
    add_executable(L1SpillManagementMockAllocatorTests
        TestL1SpillManagementMockAllocator.cpp
    )

    target_compile_options(L1SpillManagementMockAllocatorTests
        PRIVATE
        -fno-rtti
    )

    target_include_directories(L1SpillManagementMockAllocatorTests
        PUBLIC
        ${PROJECT_SOURCE_DIR}/test/unittests/OpModel/TTNN/
    )

    target_link_libraries(L1SpillManagementMockAllocatorTests
        PRIVATE
        gtest
        TTNNOpModelLib
        MLIRTTCoreDialect
        MLIRTTNNDialect
        MLIRTTTransforms
        MLIRTTNNAnalysis
        MLIRTTNNValidation
        TTMLIRTestUtils
    )
endif()
