include(BuildFlatbuffers)

set(TTNN_FBS_GEN_SOURCES
  types.fbs
  binary.fbs
  program.fbs
)

build_flatbuffers("${TTNN_FBS_GEN_SOURCES}" TTNN_FBS)
