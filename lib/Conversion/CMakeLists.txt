add_subdirectory(StableHLOToTTIR)
add_subdirectory(TosaToTTIR)
add_subdirectory(TTNNToEmitC)
add_subdirectory(TTIRToTTNN)

add_library(TTMLIRConversions INTERFACE)

target_link_libraries(TTMLIRConversions INTERFACE
  TTMLIRStableHLOToTTIR
  TTMLIRTosaToTTIR
  TTMLIRTTNNToEmitC
  TTMLIRTTIRToTTNN
)
