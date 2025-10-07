// pybind_ops.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ops.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ttnn_supplemental, m) {
    m.doc() = "TTNN supplemental operations";

    // Bind enums
    py::enum_<ttnn::distributed::MeshShardDirection>(m, "MeshShardDirection")
        .value("FullToShard", ttnn::distributed::MeshShardDirection::FullToShard)
        .value("ShardToFull", ttnn::distributed::MeshShardDirection::ShardToFull)
        .export_values();

    py::enum_<ttnn::distributed::MeshShardType>(m, "MeshShardType")
        .value("Identity", ttnn::distributed::MeshShardType::Identity)
        .value("Replicate", ttnn::distributed::MeshShardType::Replicate)
        .value("Maximal", ttnn::distributed::MeshShardType::Maximal)
        .value("Devices", ttnn::distributed::MeshShardType::Devices)
        .export_values();

    // Bind mesh_shard function
    m.def("mesh_shard",
          &ttnn::distributed::mesh_shard,
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
}