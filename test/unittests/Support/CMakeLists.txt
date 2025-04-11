add_mlir_unittest(TTMLIRSupportTests
  LoggerTest.cpp
  )

target_link_libraries(TTMLIRSupportTests
  PRIVATE
  MLIRSupport
  )
