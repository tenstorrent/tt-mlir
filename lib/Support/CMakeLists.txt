add_mlir_library(TTMLIRSupport
  TTPrintIRInstrumentation.cpp
  TTGraphTelemetryInstrumentation.cpp

  ADDITIONAL_HEADER_DIRS
  ${TTMLIR_SOURCE_DIR}/include/ttmlir/Support

  DEPENDS
  MLIRPass

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRSupport
  MLIRGraphTelemetry
  MLIRGraphTelemetryVisitors
)
