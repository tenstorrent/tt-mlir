add_mlir_dialect_library(MLIRTTMetalTransforms
        Passes.cpp
        KernelsToCpp.cpp
        SerializeToBinary.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTMetalOpsIncGen
        MLIRTTMetalPassesIncGen
        MLIRTTOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )

target_include_directories(MLIRTTMetalTransforms PUBLIC ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)
