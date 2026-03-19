add_mlir_dialect_library(MLIRTTNNEraseInverseOps
        EraseInverseOps.cpp
        ElementwiseCommutePatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTNNPassesIncGen
)
