add_mlir_dialect_library(MLIRTTNNAnalysis
        LegalGridAnalysis.cpp
        OpConfigAnalysis.cpp
        ShardingAnalysis.cpp
        ShardChainConfig.cpp
        DFShardingPolicy.cpp
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
