add_mlir_unittest(SchedulerTests
    TestScheduler.cpp
)

target_link_libraries(SchedulerTests
    PRIVATE
    MLIRTTNNDialect
    MLIRTTTransforms
    MLIRScheduler
)
