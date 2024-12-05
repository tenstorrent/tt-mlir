add_library(TTMLIRConversions INTERFACE)

add_subdirectory(TosaToTTIR)
add_subdirectory(TTIRToTTIRDecomposition)
add_subdirectory(TTIRToTTIRFusion)
add_subdirectory(TTNNToEmitC)
add_subdirectory(TTIRToTTNN)
add_subdirectory(TTIRToTTMetal)
add_subdirectory(TTKernelToEmitC)

if (TTMLIR_ENABLE_STABLEHLO)
add_subdirectory(StableHLOToTTIR)
endif()

include_directories(${TTMLIR_SOURCE_DIR}/include)

set(link_libs
TTMLIRTosaToTTIR
TTMLIRTTIRToTTIRDecomposition
TTMLIRTTIRToTTIRFusion
TTMLIRTTNNToEmitC
TTMLIRTTIRToTTNN
TTMLIRTTIRToTTMetal
TTMLIRTTKernelToEmitC
)

if (TTMLIR_ENABLE_STABLEHLO)
list(APPEND link_libs TTMLIRStableHLOToTTIR)
endif()

target_link_libraries(TTMLIRConversions INTERFACE
${link_libs}
)
