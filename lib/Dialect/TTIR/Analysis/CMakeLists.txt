add_mlir_dialect_library(MLIRTTIRAnalysis
        LegalGridAnalysis.cpp
        OpConfigAnalysis.cpp
        ShardingAnalysis.cpp
        ShardChainConfig.cpp
        DFShardingPolicy.cpp
        ShardSolver.cpp
        TTNNOpInterface.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen

        LINK_LIBS
        MLIRScheduler
        )

target_link_libraries(MLIRTTIRAnalysis PRIVATE TTMETAL_LIBRARY TTNN_LIBRARY TTNN_MLIR_INTERFACE_LIBRARY)

# Get the public header directory of the MLIR interface library
get_target_property(MLIR_INTERFACE_PUBLIC_HEADER TTNN_MLIR_INTERFACE_LIBRARY PUBLIC_HEADER)
get_filename_component(MLIR_INTERFACE_PUBLIC_HEADER_INCLUDE ${MLIR_INTERFACE_PUBLIC_HEADER} DIRECTORY)

target_include_directories(MLIRTTIRAnalysis PRIVATE
    ${MLIR_INTERFACE_PUBLIC_HEADER_INCLUDE}
)
