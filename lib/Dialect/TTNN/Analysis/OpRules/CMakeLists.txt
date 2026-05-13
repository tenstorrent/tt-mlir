set(OPRULES_SRCS
    ConvRules.cpp
    DataMovementRules.cpp
    EmbeddingRules.cpp
    MatmulRules.cpp
    NormalizationRules.cpp
    OpRuleBook.cpp
    TransformerRules.cpp
    TypecastRules.cpp
)

# Add sources to the parent library target.
foreach(src ${OPRULES_SRCS})
  target_sources(obj.MLIRTTNNAnalysis PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${src})
endforeach()
