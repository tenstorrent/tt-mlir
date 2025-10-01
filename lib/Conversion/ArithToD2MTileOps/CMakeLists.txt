add_mlir_conversion_library(TTMLIRArithToD2MTileOps
  ArithToD2MTileOps.cpp

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRArithDialect
)
