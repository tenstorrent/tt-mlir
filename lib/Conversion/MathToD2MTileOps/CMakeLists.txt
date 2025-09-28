add_mlir_conversion_library(TTMLIRMathToD2MTileOps
  MathToD2MTileOps.cpp

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRMathDialect
)
