add_library(TTMLIRConversions INTERFACE)

add_subdirectory(TosaToTTIR)
add_subdirectory(TTNNToEmitC)
add_subdirectory(TTIRToTTNN)
if (TTMLIR_ENABLE_STABLEHLO)
add_subdirectory(StableHLOToTTIR)
endif()

include_directories(${TTMLIR_SOURCE_DIR}/include)

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
