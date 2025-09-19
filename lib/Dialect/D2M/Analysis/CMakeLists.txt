add_subdirectory(Allocation)

add_mlir_dialect_library(MLIRD2MAnalysis
  AssociatedDMAWaits.cpp
  BlockFactorAnalysis.cpp
  CBProducerConsumer.cpp
  GenericInterchangeAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  DEPENDS
    MLIRD2MOpsIncGen
    MLIRTTCoreOpsIncGen

  LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRD2MAllocation
)
