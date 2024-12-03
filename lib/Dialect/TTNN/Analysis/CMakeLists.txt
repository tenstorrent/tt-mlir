add_mlir_dialect_library(MLIRTTNNAnalysis
        LegalLayoutAnalysis.cpp
        OpConfigAnalysis.cpp
        MemoryLayoutAnalysis.cpp
        L1ChainConfig.cpp
        DisjointL1ChainConfigsUnion.cpp
        DFShardingPolicy.cpp
        GreedyL1InterleavedPolicy.cpp
        BFInterleavedPolicy.cpp
        ShardSolver.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTNNPassesIncGen
        MLIRTTOpsIncGen

        LINK_LIBS PUBLIC
        MLIRScheduler
        )
