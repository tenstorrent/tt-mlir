file(REMOVE_RECURSE
  "../../libTTMetalTargetFlatbuffer.a"
  "../../libTTMetalTargetFlatbuffer.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TTMetalTargetFlatbuffer.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
