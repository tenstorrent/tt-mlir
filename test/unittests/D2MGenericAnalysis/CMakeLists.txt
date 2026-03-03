set(TESTSUITE D2MGenericAnalysisTests)

add_mlir_unittest(${TESTSUITE}
    TestD2MGenericAnalysis.cpp
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
    MLIRD2MUtils
    MLIRTTCoreDialect
    MLIRTTIRDialect
    MLIRTTNNDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    coverage_config
)
