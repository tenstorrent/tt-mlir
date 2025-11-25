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
