add_mlir_conversion_library(TTMLIRTTIRToTosa
  TTIRToTosa.cpp
  TTIRToTosaPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir/Conversion/TTIRToTosa

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  MLIRArithDialect
)
