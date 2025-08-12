include(BuildFlatbuffers)

set(CUDA_FBS_GEN_SOURCES
  program.fbs
)

build_flatbuffers("cuda" "${CUDA_FBS_GEN_SOURCES}" CUDA_FBS)
