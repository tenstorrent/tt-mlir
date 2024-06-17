include(BuildFlatbuffers)

set(TTNN_FBS_GEN_SOURCES
  binary.fbs
)

build_flatbuffers("${TTNN_FBS_GEN_SOURCES}" TTNN_FBS)
