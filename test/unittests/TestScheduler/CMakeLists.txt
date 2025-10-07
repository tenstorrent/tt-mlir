add_mlir_unittest(SchedulerTests
    TestScheduler.cpp
)

target_link_libraries(SchedulerTests
    PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTTNNDialect>
    MLIRTTTransforms
    MLIRScheduler
)
