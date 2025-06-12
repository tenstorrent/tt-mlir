add_mlir_dialect_library(MLIRTTIREraseInverseOps
        EraseInverseOps.cpp
        BroadcastCommutePatterns.cpp
        ElementwiseCommutePatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen
)
