add_mlir_unittest(TTNNToEmitCTests
    TestEmitCConversion.cpp
)

target_link_libraries(TTNNToEmitCTests
    PRIVATE
    MLIRIR
)
