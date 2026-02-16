add_mlir_unittest(OptimizerTests
    TestShardSolver.cpp
    TestOptimizerOverrides.cpp
    TestGreedyL1InterleavedPolicy.cpp
    TestLegalLayoutAnalysis.cpp
    TestLegalTensorLayoutAnalysis.cpp
    TestConv2dConfigGenerator.cpp
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
