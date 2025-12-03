add_subdirectory(Allocation)

add_mlir_dialect_library(MLIRD2MAnalysis
  AssociatedDMAWaits.cpp
  GenericOpBufferAnalysis.cpp
  CBProducerConsumer.cpp
  GenericInterchangeAnalysis.cpp
  DstAnalysisBasic.cpp
  DstAnalysisGraphColoring.cpp
  DstCapacityAnalysis.cpp
  GraphColoringStrategy.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  DEPENDS
    MLIRD2MOpsIncGen
    MLIRTTCoreOpsIncGen

  LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRD2MAllocation
)
