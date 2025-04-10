add_mlir_dialect_library(MLIRTTNNTransforms
        Optimizer.cpp
        Passes.cpp
        TTNNLayout.cpp
        TTNNDecomposeLayouts.cpp
        TTNNToCpp.cpp
        TTNNOptimizeConv2d.cpp
        Workarounds/Decomposition/ArgMaxOpRewritePattern.cpp
        Workarounds/Decomposition/CumSumOpRewritePattern.cpp
        Workarounds/Decomposition/EmbeddingOpSqueezeWeightRewritePattern.cpp
        Workarounds/Decomposition/ReduceOpsRewritePattern.cpp
        Workarounds/Decomposition/RepeatOpRewritePattern.cpp
        Workarounds/TTNNWorkaroundsPatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTNNPassesIncGen
        MLIRTTOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTTNNDialect
        MLIRTTNNAnalysis
        MLIRTTDialect
        )
