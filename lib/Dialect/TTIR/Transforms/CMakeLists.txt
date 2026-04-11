add_subdirectory(EraseInverseOps)
add_subdirectory(HoistCPUOps)
add_mlir_dialect_library(MLIRTTIRTransforms
        Broadcast.cpp
        CPUBooleanNarrowing.cpp
        DecomposeComplexPermute.cpp
        DecomposeComplexReshape.cpp
        DecomposeComposites.cpp
        DecomposeMinReduction.cpp
        GlobalDataFormatConversion.cpp
        ExplicateTMs.cpp
        FlattenSlidingWindow.cpp
        FoldConstantReshapeBroadcast.cpp
        ElementTypeNormalization.cpp
        PredicateTypeAlignment.cpp
        InferKVCacheArgumentTypes.cpp
        PropagateWeightDtype.cpp
        MoveReshapeToConstant.cpp
        Quantization.cpp
        QuantDequantConversion.cpp
        ReductionForceKeepDim.cpp
        RankNormalization.cpp
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
        MLIRTTIRHoistCPUOps
        )
