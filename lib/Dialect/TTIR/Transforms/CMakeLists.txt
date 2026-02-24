add_subdirectory(EraseInverseOps)
add_mlir_dialect_library(MLIRTTIRTransforms
        Broadcast.cpp
        DecomposeComplexReshape.cpp
        ExplicateTMs.cpp
        FlattenSlidingWindow.cpp
        FoldConstantReshapeBroadcast.cpp
        HoistCPUOps.cpp
        ElementTypeNormalization.cpp
        InferKVCacheArgumentTypes.cpp
        MoveReshapeToConstant.cpp
        Quantization.cpp
        QuantDequantConversion.cpp
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
        )
