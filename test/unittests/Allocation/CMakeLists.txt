add_mlir_unittest(AllocationTests
    TestAllocation.cpp
)

target_link_libraries(AllocationTests
    PRIVATE
    MLIRTTIRAnalysis
    coverage_config
)
