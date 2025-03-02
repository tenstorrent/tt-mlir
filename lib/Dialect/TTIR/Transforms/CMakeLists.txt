add_mlir_dialect_library(MLIRTTIRTransforms
        Allocate.cpp
        Broadcast.cpp
        Constant.cpp
        GenericDatamovement.cpp
        HoistCPUOps.cpp
        Layout.cpp
        Transforms.cpp
        Utility.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen
        )
