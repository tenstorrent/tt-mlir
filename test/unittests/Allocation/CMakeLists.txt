set(TESTSUITE AllocationTests)

add_mlir_unittest(${TESTSUITE}
    TestAllocation.cpp
    TestBlockFactorAnalysis.cpp
)

target_include_directories(${TESTSUITE}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../
)

target_link_libraries(${TESTSUITE}
  PRIVATE
    TTMLIRTestUtils
    MLIRD2MAnalysis
    coverage_config
)
