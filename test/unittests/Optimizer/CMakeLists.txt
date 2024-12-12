add_mlir_unittest(OptimizerTests
    TestShardSolver.cpp
    TestOptimizerOverrides.cpp
    TestL1InterleavedPolicy.cpp
)

target_link_libraries(OptimizerTests
    PRIVATE
    MLIRTTDialect
    MLIRTTNNAnalysis
    MLIRTTNNPipelines
)
