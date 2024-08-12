include_directories(${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo)
include_directories(${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo-build)

add_subdirectory(StableHLOToTTIR)
add_subdirectory(TosaToTTIR)
add_subdirectory(TTNNToEmitC)
add_subdirectory(TTIRToTTNN)

add_library(TTMLIRConversions INTERFACE)

add_dependencies(TTMLIRConversions stablehlo)

target_link_libraries(TTMLIRConversions INTERFACE
  TTMLIRStableHLOToTTIR
  TTMLIRTosaToTTIR
  TTMLIRTTNNToEmitC
  TTMLIRTTIRToTTNN
)
