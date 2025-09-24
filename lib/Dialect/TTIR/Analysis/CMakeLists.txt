add_subdirectory(Allocation)

add_mlir_dialect_library(MLIRTTIRAnalysis
  AssociatedDMAWaits.cpp
  CBProducerConsumer.cpp
  GenericInterchangeAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ttmlir

  DEPENDS
    MLIRTTIROpsIncGen
    MLIRTTCoreOpsIncGen

  LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRTTIRAllocation
)
