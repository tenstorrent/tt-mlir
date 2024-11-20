file(REMOVE_RECURSE
  "../../libTTMLIRTTNNToEmitC.a"
  "../../libTTMLIRTTNNToEmitC.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TTMLIRTTNNToEmitC.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
