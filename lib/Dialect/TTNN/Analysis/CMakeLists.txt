add_mlir_dialect_library(MLIRTTNNAnalysis
        LegalLayoutAnalysis.cpp
        OpConfigAnalysis.cpp
        MemoryLayoutAnalysis.cpp
        L1ChainConfig.cpp
        DFShardingPolicy.cpp
        L1InterleavedPolicy.cpp
        ShardSolver.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTNNPassesIncGen
        MLIRTTOpsIncGen

        LINK_LIBS
        MLIRScheduler
        )
