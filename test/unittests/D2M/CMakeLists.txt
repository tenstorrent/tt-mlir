set(TESTSUITE D2MTests)

add_mlir_unittest(${TESTSUITE}
    TestDstCapacityAnalysis.cpp
    TestGraphColoringStrategy.cpp
    TestD2MUtils.cpp
    TestLoopSemantics.cpp
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
    MLIRD2MTransforms
    MLIRD2MUtils
    MLIRTTCoreDialect
    MLIRTTNNDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    coverage_config
)
