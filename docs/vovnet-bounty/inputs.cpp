#include "common.hpp"

::std::vector<::ttnn::Tensor> create_inputs_for_forward() {
  ttnn::distributed::MeshDevice *v1 = ttnn::DeviceGetter::getInstance();
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>>
      v2 = *v1;
  ::ttnn::Tensor v3 = ttnn::ones(
      ::ttnn::Shape({16, 3, 224, 224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, v2,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v9 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v11 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v12 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v13 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v14 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v15 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v16 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v17 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v18 = ttnn::ones(
      ::ttnn::Shape({1, 256, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v19 = ttnn::ones(
      ::ttnn::Shape({1, 256, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v20 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v21 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v22 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v23 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v24 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v25 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v26 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v27 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v28 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v29 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v30 = ttnn::ones(
      ::ttnn::Shape({1, 512, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v31 = ttnn::ones(
      ::ttnn::Shape({1, 512, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v32 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v33 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v34 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v35 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v36 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v37 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v38 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v39 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v40 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v41 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v42 = ttnn::ones(
      ::ttnn::Shape({1, 768, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v43 = ttnn::ones(
      ::ttnn::Shape({1, 768, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v44 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v45 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v46 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v47 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v48 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v49 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v50 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v51 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v52 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v53 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v54 = ttnn::ones(
      ::ttnn::Shape({1, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v55 = ttnn::ones(
      ::ttnn::Shape({1, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v56 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v57 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v58 = ttnn::ones(
      ::ttnn::Shape({64, 3, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v59 = ttnn::ones(
      ::ttnn::Shape({64, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v60 = ttnn::ones(
      ::ttnn::Shape({64, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v61 = ttnn::ones(
      ::ttnn::Shape({64, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v62 = ttnn::ones(
      ::ttnn::Shape({64, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v63 = ttnn::ones(
      ::ttnn::Shape({128, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v64 = ttnn::ones(
      ::ttnn::Shape({128, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v65 = ttnn::ones(
      ::ttnn::Shape({128, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v66 = ttnn::ones(
      ::ttnn::Shape({128, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v67 = ttnn::ones(
      ::ttnn::Shape({128, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v68 = ttnn::ones(
      ::ttnn::Shape({128, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v69 = ttnn::ones(
      ::ttnn::Shape({128, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v70 = ttnn::ones(
      ::ttnn::Shape({256, 448, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v71 = ttnn::ones(
      ::ttnn::Shape({256, 256, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v72 = ttnn::ones(
      ::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v73 = ttnn::ones(
      ::ttnn::Shape({160, 256, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v74 = ttnn::ones(
      ::ttnn::Shape({160, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v75 = ttnn::ones(
      ::ttnn::Shape({160, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v76 = ttnn::ones(
      ::ttnn::Shape({160, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v77 = ttnn::ones(
      ::ttnn::Shape({160, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v78 = ttnn::ones(
      ::ttnn::Shape({160, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v79 = ttnn::ones(
      ::ttnn::Shape({160, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v80 = ttnn::ones(
      ::ttnn::Shape({512, 736, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v81 = ttnn::ones(
      ::ttnn::Shape({512, 512, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v82 = ttnn::ones(
      ::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v83 = ttnn::ones(
      ::ttnn::Shape({192, 512, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v84 = ttnn::ones(
      ::ttnn::Shape({192, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v85 = ttnn::ones(
      ::ttnn::Shape({192, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v86 = ttnn::ones(
      ::ttnn::Shape({192, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v87 = ttnn::ones(
      ::ttnn::Shape({192, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v88 = ttnn::ones(
      ::ttnn::Shape({192, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v89 = ttnn::ones(
      ::ttnn::Shape({192, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v90 = ttnn::ones(
      ::ttnn::Shape({768, 1088, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v91 = ttnn::ones(
      ::ttnn::Shape({768, 768, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v92 = ttnn::ones(
      ::ttnn::Shape({1, 1, 1, 768}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v93 = ttnn::ones(
      ::ttnn::Shape({224, 768, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v94 = ttnn::ones(
      ::ttnn::Shape({224, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v95 = ttnn::ones(
      ::ttnn::Shape({224, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v96 = ttnn::ones(
      ::ttnn::Shape({224, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v97 = ttnn::ones(
      ::ttnn::Shape({224, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v98 = ttnn::ones(
      ::ttnn::Shape({224, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v99 = ttnn::ones(
      ::ttnn::Shape({224, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v100 = ttnn::ones(
      ::ttnn::Shape({1024, 1440, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v101 = ttnn::ones(
      ::ttnn::Shape({1024, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v102 = ttnn::ones(
      ::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>>
      v103 = *v1;
  ::ttnn::Tensor v104 = ttnn::ones(
      ::ttnn::Shape({1024, 1000}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::TILE, v103,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>>
      v105 = *v1;
  ::ttnn::Tensor v106 = ttnn::ones(
      ::ttnn::Shape({1000}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE,
      v105,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v107 = util_create_vec(
      v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
      v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33,
      v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48,
      v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63,
      v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77, v78,
      v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92, v93,
      v94, v95, v96, v97, v98, v99, v100, v101, v102, v104, v106);
  return v107;
}