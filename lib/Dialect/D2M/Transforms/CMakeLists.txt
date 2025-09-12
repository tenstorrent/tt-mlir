add_mlir_library(MLIRD2MTransforms
        Passes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRD2MPassesIncGen

        LINK_LIBS PUBLIC
        MLIRD2MDialect
        )
