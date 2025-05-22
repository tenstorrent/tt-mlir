# Generate a standalone binary instead an mlir_unittest since TestLayoutAnalysis tests
# needs to run serially. llvm-lit runs tests in parallel
add_executable(OptimizerTests
    TestShardSolver.cpp
    TestOptimizerOverrides.cpp
    TestGreedyL1InterleavedPolicy.cpp
    TestLayoutAnalysis.cpp
)

target_compile_options(OptimizerTests
    PRIVATE
    -fno-rtti
)

target_link_libraries(OptimizerTests
    PRIVATE
    gtest
    gtest_main
    MLIRTTDialect
    MLIRTTNNAnalysis
    MLIRTTNNPipelines
)
