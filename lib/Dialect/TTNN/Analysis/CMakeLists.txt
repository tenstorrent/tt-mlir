add_mlir_dialect_library(MLIRTTNNAnalysis
        BFInterleavedPolicy.cpp
        Conv2dConfigSearchSpace.cpp
        DFShardingPolicy.cpp
        GreedyL1InterleavedPolicy.cpp
        L1ChainConfig.cpp
        LegalOpConfigAnalysis.cpp
        LegalOpLayoutAnalysis.cpp
        LegalTensorLayoutAnalysis.cpp
        MemoryLayoutAnalysis.cpp
        OpConfigAnalysis.cpp
        PostOptimizerValidationAnalysis.cpp
        ScalarDataTypeAnalysis.cpp
        ShardSolver.cpp
        TensorLayouts.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTNNPassesIncGen
        MLIRTTCoreOpsIncGen
        TTNNOpModelLib

        LINK_LIBS PUBLIC
        MLIRScheduler
        MLIRTTNNValidation
        TTMLIRTTNNUtils
        TTNNOpModelLib
        )
