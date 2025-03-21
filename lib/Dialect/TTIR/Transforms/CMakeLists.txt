add_subdirectory(EraseInverseOps)
add_mlir_dialect_library(MLIRTTIRTransforms
        Allocate.cpp
        Broadcast.cpp
        Constant.cpp
        GeneralizeNamedOps.cpp
        EraseInverseOps/EraseInverseOps.cpp
        EraseInverseOps/BroadcastCommutePatterns.cpp
        EraseInverseOps/ElementwiseCommutePatterns.cpp
        # EraseInverseOps/EraseInverseOps.cpp
        # EraseInverseOps/BroadcastCommutePatterns.cpp
        # EraseInverseOps/ElementwiseCommutePatterns.cpp
        FlattenSlidingWindow.cpp
        GenericLinearizeMemref.cpp
        GenericGenerateDatamovement.cpp
        GenericHWThreadSelection.cpp
        AttachMetalLayout.cpp
        OptimizeTensorLayout.cpp
        HoistCPUOps.cpp
        Bufferization.cpp
        Layout.cpp
        Transforms.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTTIREraseInverseOps
        )
