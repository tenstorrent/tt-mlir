add_mlir_conversion_library(TTMLIRTTIRToLinalg
  EltwiseBinary.cpp
  EltwiseUnary.cpp
  Pooling.cpp
  TTIRToLinalg.cpp
  TTIRToLinalgPass.cpp
  Utils.cpp

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
