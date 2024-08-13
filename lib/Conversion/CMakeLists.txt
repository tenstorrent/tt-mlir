add_library(TTMLIRConversions INTERFACE)

if (TTMLIR_ENABLE_STABLEHLO)
include_directories(${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo)
include_directories(${PROJECT_SOURCE_DIR}/third_party/stablehlo/src/stablehlo-build)
add_subdirectory(StableHLOToTTIR)
add_dependencies(TTMLIRConversions stablehlo)
endif()

add_subdirectory(TosaToTTIR)
add_subdirectory(TTNNToEmitC)
add_subdirectory(TTIRToTTNN)

set(link_libs 
TTMLIRTosaToTTIR;
TTMLIRTTNNToEmitC;
TTMLIRTTIRToTTNN
)

if (TTMLIR_ENABLE_STABLEHLO)
list(APPEND link_libs TTMLIRStableHLOToTTIR)
endif()

target_link_libraries(TTMLIRConversions INTERFACE
  ${link_libs}
)
