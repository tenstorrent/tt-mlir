add_mlir_unittest(SchedulerTests
    TestScheduler.cpp
)

target_link_libraries(SchedulerTests
    PRIVATE
    MLIRTTCoreDialect
    MLIRTTIRDialect
    MLIRTTNNPipelines
    MLIRScheduler
)

if (TTMLIR_ENABLE_TTNN_PY)
    target_link_libraries(SchedulerTests PRIVATE Python3::Python)
endif()
