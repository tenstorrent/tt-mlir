include(BuildFlatbuffers)

set(COMMON_FBS_GEN_SOURCES
  system_desc.fbs
  types.fbs
  version.fbs
  debug_info.fbs
)

build_flatbuffers("${COMMON_FBS_GEN_SOURCES}" COMMON_FBS)
