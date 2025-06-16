add_mlir_unittest(TTNNToEmitCTests
    TestEmitCConversion.cpp
)

target_link_libraries(TTNNToEmitCTests
    PRIVATE
    MLIRTTNNPipelines
)

if (TTMLIR_ENABLE_TTNN_PY)
    target_link_libraries(TTNNToEmitCTests PRIVATE Python3::Python)
endif()
