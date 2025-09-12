include(BuildFlatbuffers)

set(DISTRIBUTED_FBS_GEN_SOURCES
  command.fbs
  response.fbs
)

build_flatbuffers("distributed" "${DISTRIBUTED_FBS_GEN_SOURCES}" DISTRIBUTED_FBS FBS_GENERATION)
