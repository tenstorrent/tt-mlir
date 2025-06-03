if (TTMLIR_ENABLE_OPMODEL)
        add_definitions(-DTTMLIR_ENABLE_OPMODEL)
endif()

add_mlir_dialect_library(MLIRTTNNTransforms
        Optimizer.cpp
        Passes.cpp
        TTNNDecomposeLayouts.cpp
        TTNNFusing.cpp
        TTNNLayout.cpp
        TTNNMergeLayoutsWithFuncArgsPass.cpp
        TTNNPrepareConv2dWeights.cpp
        TTNNToCpp.cpp
        Workarounds/Decomposition/ArgMaxOpRewritePattern.cpp
        Workarounds/Decomposition/CumSumOpDimRewritePattern.cpp
        Workarounds/Decomposition/CumSumOpRankRewritePattern.cpp
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
