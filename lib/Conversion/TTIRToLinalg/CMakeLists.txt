add_mlir_conversion_library(TTMLIRTTIRToLinalg
  TTIRToLinalg.cpp
  TTIRToLinalgPass.cpp

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
