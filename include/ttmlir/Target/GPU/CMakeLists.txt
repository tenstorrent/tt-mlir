include(BuildFlatbuffers)

set(GPU_FBS_GEN_SOURCES
  program.fbs
)

build_flatbuffers("gpu" "${GPU_FBS_GEN_SOURCES}" GPU_FBS)
