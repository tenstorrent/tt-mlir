add_mlir_unittest(EraseInverseOpsTests
  TestConstevalForwardAnalysis.cpp
)

target_link_libraries(EraseInverseOpsTests
  PRIVATE
    MLIRArithDialect
    MLIRIR
    MLIRFuncDialect
    MLIRTransforms
    MLIRParser
    MLIRTTCoreDialect
    MLIRTTIRDialect
    MLIRTTIREraseInverseOps
)
