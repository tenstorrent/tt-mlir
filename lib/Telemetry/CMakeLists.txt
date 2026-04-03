add_mlir_library(MLIRGraphTelemetry
  IRSerializer.cpp
  TelemetryWriter.cpp
  TelemetryVisitor.cpp

  ADDITIONAL_HEADER_DIRS
  ${TTMLIR_SOURCE_DIR}/include/ttmlir/Telemetry

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRCallInterfaces
  MLIRFunctionInterfaces
  MLIRSupport
)

add_subdirectory(Visitors)
