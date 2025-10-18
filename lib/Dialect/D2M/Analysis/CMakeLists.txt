add_subdirectory(Allocation)

add_mlir_dialect_library(MLIRD2MAnalysis
  AssociatedDMAWaits.cpp
  CBProducerConsumer.cpp
  DestRegisterAnalysis.cpp
  GenericInterchangeAnalysis.cpp
  GenericOpBufferAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  DEPENDS
    MLIRD2MOpsIncGen
    MLIRTTCoreOpsIncGen

  LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRD2MAllocation
)
