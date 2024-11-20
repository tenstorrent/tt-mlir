add_mlir_dialect_library(MLIRTTIRTransforms
        Allocate.cpp
        Constant.cpp
        Generic.cpp
        Layout.cpp
        Reshape.cpp
        Transforms.cpp
        Utility.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen
        )
