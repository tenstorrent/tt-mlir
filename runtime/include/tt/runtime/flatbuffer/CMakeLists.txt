include(BuildFlatbuffers)

set(RUNTIME_FBS_GEN_SOURCES
  types.fbs
)

build_flatbuffers("runtime" "${RUNTIME_FBS_GEN_SOURCES}" RUNTIME_FBS FBS_GENERATION)
