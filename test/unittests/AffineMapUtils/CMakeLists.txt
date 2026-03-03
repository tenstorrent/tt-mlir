set(TESTSUITE AffineMapUtilsTests)

add_mlir_unittest(${TESTSUITE}
    TestAffineMapUtils.cpp
)

target_include_directories(${TESTSUITE}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../
)

target_link_libraries(${TESTSUITE}
  PRIVATE
    TTMLIRTestUtils
    MLIRD2MUtils
    MLIRSupport
    MLIRIR
    MLIRAffineDialect
    coverage_config
)
