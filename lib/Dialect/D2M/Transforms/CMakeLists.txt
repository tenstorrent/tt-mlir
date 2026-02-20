add_mlir_dialect_library(MLIRD2MTransforms
        AddScratchInputs.cpp
        Allocate.cpp
        DecomposeArange.cpp
        DecomposeMasking.cpp
        DecomposeComplexPermute.cpp
        GenericApplyInterchange.cpp
        GenericLinearizeMemref.cpp
        GenericRegionsToFuncs.cpp
        GenericReplaceGlobals.cpp
        GenericTileComputeLoops.cpp
        GridSelection.cpp
        InsertDstRegisterAccess.cpp
        InsertStreams.cpp
        LinalgToAffine.cpp
        LowerScratchAllocate.cpp
        LowerToLayout.cpp
        MaterializeViewReturns.cpp
        ElementwiseFusion.cpp
        OpScheduler.cpp
        SFPUTileLoopFission.cpp
        GlobalDataFormatConversion.cpp
        RankNormalization.cpp
        ScalarizeConstTensors.cpp
        GenerateOuterLoops.cpp
        LowerMulticastLoads.cpp
        SplitUnifiedThread.cpp
        PreallocateMcastSemaphores.cpp
        ScheduleDMA.cpp
        LowerLoadStoreOpsToDMA.cpp
        LowerLoadStoreOpsToExplicitCBForm.cpp
        ConvertLocalLoadStoreOpsToAliasedCBs.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRD2MOpsIncGen
        MLIRD2MPassesIncGen
        MLIRTTCoreOpsIncGen

        LINK_LIBS PUBLIC
        MLIRD2MAnalysis
        MLIRD2MUtils
        MLIRTTUtils
        )
