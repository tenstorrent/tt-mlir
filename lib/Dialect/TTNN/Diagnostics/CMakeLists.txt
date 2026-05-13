add_mlir_dialect_library(MLIRTTNNDiagnostics
        DecisionTrace.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTTNNAnalysis
        )
