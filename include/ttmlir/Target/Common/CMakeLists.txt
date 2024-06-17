include(BuildFlatbuffers)

set(COMMON_FBS_GEN_SOURCES
  device_desc.fbs
  types.fbs
)

build_flatbuffers("${COMMON_FBS_GEN_SOURCES}" COMMON_FBS)
