add_mlir_dialect_library(MLIRTTIRTransforms
        PARTIAL_SOURCES_INTENDED
        Allocate.cpp
        Broadcast.cpp
        Constant.cpp
        Generic.cpp
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
