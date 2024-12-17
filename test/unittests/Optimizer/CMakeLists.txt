add_mlir_unittest(OptimizerTests
    TestShardSolver.cpp
    TestOptimizerOverrides.cpp
    TestGreedyL1InterleavedPolicy.cpp
)

target_link_libraries(OptimizerTests
    PRIVATE
    MLIR
    MLIRTTDialect
    MLIRTTNNAnalysis
    MLIRTTNNPipelines
)
