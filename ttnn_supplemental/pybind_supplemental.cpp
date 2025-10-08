// pybind_ops.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "supplemental.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ttnn_supplemental, m) {
    m.doc() = "TTNN supplemental CCL operations";

    // Bind enums
    py::enum_<ttnn::supplemental::MeshShardDirection>(m, "MeshShardDirection")
        .value("FullToShard", ttnn::supplemental::MeshShardDirection::FullToShard)
        .value("ShardToFull", ttnn::supplemental::MeshShardDirection::ShardToFull)
        .export_values();

    py::enum_<ttnn::supplemental::MeshShardType>(m, "MeshShardType")
        .value("Identity", ttnn::supplemental::MeshShardType::Identity)
        .value("Replicate", ttnn::supplemental::MeshShardType::Replicate)
        .value("Maximal", ttnn::supplemental::MeshShardType::Maximal)
        .value("Devices", ttnn::supplemental::MeshShardType::Devices)
        .export_values();

    // Bind mesh_shard function
    m.def("mesh_shard",
          &ttnn::supplemental::mesh_shard,
          py::arg("input"),
          py::arg("mesh_device"),
          py::arg("shard_direction"),
          py::arg("shard_type"),
          py::arg("shard_shape"),
          py::arg("shard_dims"),
          R"doc(
          Shard or aggregate a tensor across a mesh device.

          Args:
              input (Tensor): The input tensor to shard or aggregate.
              mesh_device (MeshDevice): The mesh device to distribute across.
              shard_direction (MeshShardDirection): Direction of sharding (FullToShard or ShardToFull).
              shard_type (MeshShardType): Type of sharding (Identity, Replicate, Maximal, or Devices).
              shard_shape (List[int]): The shape of each shard.
              shard_dims (List[int]): The dimensions to shard over.

          Returns:
              Tensor: The sharded or aggregated tensor.
          )doc");

    // Bind all_gather function
    m.def("all_gather",
          &ttnn::supplemental::all_gather,
          py::arg("input"),
          py::arg("mesh_device"),
          py::arg("dim"),
          py::arg("cluster_axis"),
          py::arg("num_links"),
          py::arg("memory_config") = std::nullopt,
          R"doc(
          Perform all-gather operation on a tensor across mesh devices.

          Args:
              input (Tensor): The input device tensor.
              mesh_device (MeshDevice): The mesh device.
              dim (int): Dimension to gather along.
              cluster_axis (int): Cluster axis for the operation.
              num_links (int): Number of links to use.
              memory_config (Optional[MemoryConfig]): Output memory configuration.

          Returns:
              Tensor: The gathered tensor.
          )doc");

    // Bind reduce_scatter function
    m.def("reduce_scatter",
          &ttnn::supplemental::reduce_scatter,
          py::arg("input"),
          py::arg("mesh_device"),
          py::arg("scatter_dim"),
          py::arg("cluster_axis"),
          py::arg("num_links"),
          py::arg("memory_config") = std::nullopt,
          R"doc(
          Perform reduce-scatter operation on a tensor across mesh devices.

          Args:
              input (Tensor): The input device tensor.
              mesh_device (MeshDevice): The mesh device.
              scatter_dim (int): Dimension to scatter along.
              cluster_axis (int): Cluster axis for the operation.
              num_links (int): Number of links to use.
              memory_config (Optional[MemoryConfig]): Output memory configuration.

          Returns:
              Tensor: The reduced and scattered tensor.
          )doc");

    // Bind collective_permute function
    m.def("collective_permute",
          &ttnn::supplemental::collective_permute,
          py::arg("input"),
          py::arg("source_target_pairs"),
          R"doc(
          Perform collective permute operation to remap tensor shards across devices.

          Args:
              input (Tensor): The input device tensor.
              source_target_pairs (List[int]): Pairs of [source_id, target_id] for remapping.
                  Must have even length. Devices not in pairs will receive zeros.

          Returns:
              Tensor: The permuted tensor.
          )doc");

    // Bind point_to_point function
    m.def("point_to_point",
          &ttnn::supplemental::point_to_point,
          py::arg("input"),
          py::arg("send_coord"),
          py::arg("receive_coord"),
          py::arg("accum_tensor") = std::nullopt,
          R"doc(
          Perform point-to-point communication between devices.

          Args:
              input (Tensor): The input device tensor.
              send_coord (List[int]): Mesh coordinates of the sending device.
              receive_coord (List[int]): Mesh coordinates of the receiving device.
              accum_tensor (Optional[Tensor]): Optional accumulation tensor.

          Returns:
              Tensor: The output tensor with data transferred.
          )doc");
}