add_subdirectory(EraseInverseOps)
add_mlir_dialect_library(MLIRTTIRTransforms
        Allocate.cpp
        Broadcast.cpp
        ExplicateTMs.cpp
        FlattenSlidingWindow.cpp
        GenericLinearizeMemref.cpp
        GenericGenerateDatamovement.cpp
        GenericGenerateLoops.cpp
        GenericHWThreadSelection.cpp
        GenericLowerDMAs.cpp
        GenericRegionsToFuncs.cpp
        InsertDstRegisterAccess.cpp
        GenericApplyInterchange.cpp
        GenericTileComputeLoops.cpp
        HoistCPUOps.cpp
        ElementTypeNormalization.cpp
        LowerToLayout.cpp
        Quantization.cpp
        QuantDequantConversion.cpp
        RemoveRedundantCastOps.cpp
        Transforms.cpp
        TTIRFusing.cpp
        MultiDeviceTensorAnnotation.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTTIREraseInverseOps
        MLIRTTIRAnalysis
        )
