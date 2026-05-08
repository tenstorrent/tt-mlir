// UNSUPPORTED: true
// Memory config mismatch, expected MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,...), got MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,...)
// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-permute-matmul-fusion=false" -o bert_1lyr_bs8_encoder_isl384_ttnn.mlir %models/single_blocks_and_layers/bert_1lyr_bs8_encoder_isl384.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn bert_1lyr_bs8_encoder_isl384_ttnn.mlir
