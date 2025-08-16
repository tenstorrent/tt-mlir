add_mlir_dialect_library(MLIRTTIRAnalysis
  AllocationPlanner.cpp
  AllocationTools.cpp
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
  )
