set(TESTSUITE AllocationTests)

add_mlir_unittest(${TESTSUITE}
    TestAllocation.cpp
    TestGenericOpBufferAnalysis.cpp
)

target_include_directories(${TESTSUITE}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../
)

target_link_libraries(${TESTSUITE}
  PRIVATE
    TTMLIRTestUtils
    MLIRD2MAnalysis
    MLIRD2MDialect
    MLIRTTCoreDialect
    MLIRTTIRDialect
    MLIRTTNNDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    coverage_config
)
