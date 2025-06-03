add_mlir_unittest(OptimizerTests
    TestShardSolver.cpp
    TestOptimizerOverrides.cpp
    TestGreedyL1InterleavedPolicy.cpp
    TestLegalTensorLayoutAnalysis.cpp
    TestConv2dConfigGenerator.cpp
    PARTIAL_SOURCES_INTENDED
)

target_link_libraries(OptimizerTests
    PRIVATE
    MLIRTTCoreDialect
    MLIRTTNNAnalysis
    MLIRTTNNPipelines
)
