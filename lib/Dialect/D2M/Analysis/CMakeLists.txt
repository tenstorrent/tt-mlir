add_subdirectory(Allocation)

add_mlir_dialect_library(MLIRD2MAnalysis
  AssociatedDMAWaits.cpp
  GenericOpBufferAnalysis.cpp
  CBProducerConsumer.cpp
  GenericInterchangeAnalysis.cpp
  ValidateLayouts.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  DEPENDS
    MLIRD2MOpsIncGen
    MLIRD2MAnalysisPassesIncGen
    MLIRTTCoreOpsIncGen

  LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRD2MAllocation
)
