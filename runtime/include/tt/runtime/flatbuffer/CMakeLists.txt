include(BuildFlatbuffers)

set(RUNTIME_FBS_GEN_SOURCES
  types.fbs
)

build_flatbuffers("runtime" "${RUNTIME_FBS_GEN_SOURCES}" RUNTIME_FBS FBS_GENERATION)

add_custom_target(RUNTIME_FBS_GENERATION DEPENDS RUNTIME_FBS DISTRIBUTED_FBS)
