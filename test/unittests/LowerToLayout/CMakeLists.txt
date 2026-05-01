set(TESTSUITE LowerToLayoutTests)

add_mlir_unittest(${TESTSUITE}
    TestPlan.cpp
)

target_include_directories(${TESTSUITE}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../
)

target_link_libraries(${TESTSUITE}
  PRIVATE
    TTMLIRTestUtils
    MLIRD2MTransforms
    MLIRD2MAnalysis
    MLIRD2MDialect
    MLIRD2MUtils
    MLIRTTTransforms
    MLIRTTCoreDialect
    MLIRTTIRDialect
    MLIRTTNNDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    coverage_config
)
