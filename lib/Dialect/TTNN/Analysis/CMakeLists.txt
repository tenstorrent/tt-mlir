add_mlir_dialect_library(MLIRTTNNAnalysis
        BFInterleavedPolicy.cpp
        Conv2dConfigSearchSpace.cpp
        DFShardingPolicy.cpp
        MatmulProgramConfig.cpp
        GreedyL1InterleavedPolicy.cpp
        L1ChainConfig.cpp
        L1InterleavedFallbackAnalysis.cpp
        L1SpillManagement.cpp
        LayoutPropagation.cpp
        LegalOpConfigAnalysis.cpp
        LegalOpLayoutAnalysis.cpp
        LegalTensorLayoutAnalysis.cpp
        MemoryLayoutAnalysis.cpp
        OpConfigAnalysis.cpp
        OpModelStrategy.cpp
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
        TTMLIRTTNNUtils
        TTNNOpModelLib
        )
