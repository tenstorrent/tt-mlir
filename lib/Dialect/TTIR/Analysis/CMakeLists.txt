add_mlir_dialect_library(MLIRTTIRAnalysis
        LegalGridAnalysis.cpp
        OptimalTargetGridAnalysis.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen
        )
