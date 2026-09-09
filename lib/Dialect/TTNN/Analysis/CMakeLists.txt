add_mlir_dialect_library(MLIRTTNNAnalysis
        Conv2dConfigSearchSpace.cpp
        Conv3dConfigHeuristic.cpp
        MatmulProgramConfig.cpp
        L1SpillManagement.cpp
        MemoryLayoutPropagation.cpp
        LegalOpConfigAnalysis.cpp
        LegalOpLayoutAnalysis.cpp
        LegalTensorLayoutAnalysis.cpp
        OpConfigAnalysis.cpp
        OpModelStrategy.cpp
        ScalarDataTypeAnalysis.cpp
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

add_subdirectory(OpRules)
