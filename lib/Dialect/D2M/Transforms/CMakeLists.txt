add_mlir_dialect_library(MLIRD2MTransforms
        Allocate.cpp
        GenericApplyInterchange.cpp
        GenericGenerateDatamovement.cpp
        GenericGenerateLoops.cpp
        GenericHWThreadSelection.cpp
        GenericLinearizeMemref.cpp
        GenericLowerDMAs.cpp
        GenericRegionsToFuncs.cpp
        GenericReplaceGlobals.cpp
        GenericTileComputeLoops.cpp
        GridSelection.cpp
        InsertDstRegisterAccess.cpp
        InsertStreams.cpp
        LowerToLayout.cpp
        ElementwiseFusion.cpp
        SFPUTileLoopFission.cpp
        GlobalDataFormatConversion.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRD2MOpsIncGen
        MLIRD2MPassesIncGen
        MLIRTTCoreOpsIncGen

        LINK_LIBS PUBLIC
        MLIRD2MAnalysis
        MLIRD2MUtils
        )
