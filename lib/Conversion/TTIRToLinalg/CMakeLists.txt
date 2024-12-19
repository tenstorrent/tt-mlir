add_mlir_library(TTMLIRTTIRToLinAlg
  TTIRToLinAlg.cpp
  TTIRToLinAlgPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTIRToLinalg

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRArithDialect
)
