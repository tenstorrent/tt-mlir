set(TESTSUITE LowerToLayoutTests)

add_mlir_unittest(${TESTSUITE}
    TestLowerToLayoutPlan.cpp
)

target_include_directories(${TESTSUITE}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../
)

target_link_libraries(${TESTSUITE}
  PRIVATE
    TTMLIRTestUtils
    MLIRD2MTransforms
    MLIRD2MDialect
    MLIRTTCoreDialect
    MLIRFuncDialect
    coverage_config
)
