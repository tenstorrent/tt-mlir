# Transform passes will be added here as needed
# For now, we'll comment out the transforms library until we have actual transform passes

# add_mlir_dialect_library(MLIRSFPUTransforms
#   # Transform source files will be added here
#   
#   ADDITIONAL_HEADER_DIRS
#   ${PROJECT_SOURCE_DIR}/include/ttmlir/Dialect/SFPU
# 
#   DEPENDS
#   MLIRSFPUPassIncGen
# 
#   LINK_LIBS PUBLIC
#   MLIRIR
#   MLIRPass
#   MLIRSFPUDialect
#   MLIRTransformUtils
# )