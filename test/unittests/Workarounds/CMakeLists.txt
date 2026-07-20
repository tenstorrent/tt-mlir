add_mlir_unittest(TTMLIRWorkaroundsTests
    TestWidthShardCoreGrid.cpp
)

target_include_directories(TTMLIRWorkaroundsTests
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../
)

target_link_libraries(TTMLIRWorkaroundsTests
  PRIVATE
    MLIRTTNNDialect
    coverage_config
)
