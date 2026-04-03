add_mlir_library(MLIRGraphTelemetryVisitors
  TTNNLayoutVisitor.cpp
  MemoryConfigVisitor.cpp
  RegisterVisitors.cpp

  LINK_LIBS PUBLIC
  MLIRGraphTelemetry
  MLIRFuncDialect
  MLIRTTNNDialect
)
