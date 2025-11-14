add_mlir_unittest(TTMLIRSupportTests
  LoggerTest.cpp
  TTPrintIRInstrumentationTest.cpp
  TestAsserts.cpp
  TestAssertsElision.cpp
  DeathTestAsserts.cpp
)

target_include_directories(TTMLIRSupportTests
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../
)

get_target_property(TTMLIR_SUPPORT_DEPS TTMLIRSupport LINK_LIBRARIES)

target_link_libraries(TTMLIRSupportTests
  PRIVATE
    MLIRArithDialect
    MLIRIR
    MLIRFuncDialect
    MLIRTransforms
    TTMLIRSupport
)
