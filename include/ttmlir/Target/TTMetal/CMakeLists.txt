include(BuildFlatbuffers)

set(TTMETAL_FBS_GEN_SOURCES
  types.fbs
  command.fbs
  program.fbs
  binary.fbs
)

build_flatbuffers("${TTMETAL_FBS_GEN_SOURCES}" TTMETAL_FBS)
