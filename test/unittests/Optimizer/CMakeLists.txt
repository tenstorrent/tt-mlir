add_mlir_unittest(OptimizerTests
    TestOptimizerOverrides.cpp
    TestLegalLayoutAnalysis.cpp
    TestLegalTensorLayoutAnalysis.cpp
    TestConv2dConfigGenerator.cpp
    TestConv3dConfigHeuristic.cpp
    TestPipelineOptionResolution.cpp
    TestOpRuleBook.cpp
    PARTIAL_SOURCES_INTENDED
)

target_link_libraries(OptimizerTests
    PRIVATE
    MLIRTTNNDialect
    MLIRTTTransforms
    MLIRTTNNAnalysis
    MLIRTTNNValidation
    MLIRTTNNPipelines
)

# Opt-in CPU benchmark with a link-wrapped mock backend.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT TTMLIR_ENABLE_OPMODEL)
    add_executable(RotaryReshardSearchBenchmark EXCLUDE_FROM_ALL
        benchmarks/RotaryReshardSearchBenchmark.cpp
    )
    target_compile_options(RotaryReshardSearchBenchmark PRIVATE -fno-rtti)
    target_link_libraries(RotaryReshardSearchBenchmark PRIVATE
        MLIRTTNNAnalysis MLIRTTNNValidation MLIRTTNNDialect MLIRTTTransforms
    )
    target_link_options(RotaryReshardSearchBenchmark PRIVATE
        "LINKER:--wrap=_ZN4mlir2tt4ttnn21shouldExploreReshardsEPNS_9OperationEj"
        "LINKER:--wrap=_ZN4mlir2tt4ttnn24op_constraint_validation17validateOperationEPNS_9OperationEN4llvm8ArrayRefINS1_14TTNNLayoutAttrEEERKNS1_8OpConfigEm"
    )
endif()

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
    # HW-free (mock device) on any box with the op-model build. Invoked in CI by
    # .github/test_scripts/op_model.sh (alongside the other mock-device gtest
    # binaries), since check-ttmlir does not run it.
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
