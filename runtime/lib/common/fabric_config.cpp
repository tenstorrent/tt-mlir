// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/fabric_config.h"
#include "tt-metalium/program_descriptors.hpp"
#include "tt/runtime/detail/common/logger.h"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::runtime::common {

template <typename ProgramOrDescriptor>
std::unordered_map<::tt::tt_metal::CoreCoord, std::vector<uint32_t>>
appendFabricConfigArgs(
    const ::tt::target::FabricConnectionConfig *fabricConnectionConfig,
    const target::metal::KernelConfig *kernelConfig,
    ProgramOrDescriptor &program, tt_metal::KernelHandle &handle,
    const tt_metal::distributed::MeshCoordinate deviceCoord,
    const tt_metal::distributed::MeshDevice *meshDevice,
    std::vector<uint32_t> rtArgsVec,
    const tt::tt_metal::CoreRangeSet &coreRangeSet) {
  LOG_ASSERT(fabricConnectionConfig != nullptr,
             "Fabric connection config must be available.");
  tt::target::Topology topology_type = fabricConnectionConfig->topology();
  uint32_t cluster_axis = fabricConnectionConfig->cluster_axis();
  uint32_t num_links = fabricConnectionConfig->num_links();

  std::unordered_map<tt::tt_metal::CoreCoord, std::vector<uint32_t>>
      fabricConfigArgs;

  tt::tt_fabric::FabricApiType api_type;

  // insert topology specific args (device specific)
  auto num_topology_arg_idx = rtArgsVec.size();
  rtArgsVec.push_back(1);
  LOG_ASSERT(meshDevice->shape().dims() == 2,
             "Only 2d mesh device is supported");
  if (topology_type == tt::target::Topology::Linear ||
      topology_type == tt::target::Topology::Ring) {
    // Add topology type (Line=0, Ring=1) and axis (for 1D)
    rtArgsVec.push_back(static_cast<uint32_t>(topology_type));
    LOG_ASSERT(cluster_axis < 2, "Invalid cluster axis, must be < 2");
    rtArgsVec.push_back(cluster_axis);

    // add mesh shape
    for (uint32_t dim = 0; dim < meshDevice->shape().dims(); dim++) {
      rtArgsVec.push_back(meshDevice->shape()[dim]);
    }

    // add forward and backward directions for each dim
    for (uint32_t dim = 0; dim < meshDevice->shape().dims(); dim++) {
      auto forwardCoord = deviceCoord;
      forwardCoord[dim] = (forwardCoord[dim] + 1) % meshDevice->shape()[dim];
      auto forward_direction = get_eth_forwarding_direction(
          meshDevice->get_fabric_node_id(deviceCoord),
          meshDevice->get_fabric_node_id(forwardCoord));
      LOG_ASSERT(
          forward_direction.has_value(),
          "Forward direction does not exist on mesh coordinate: ", deviceCoord);
      rtArgsVec.push_back(forward_direction.value());
      auto backwardCoord = deviceCoord;
      backwardCoord[dim] = (backwardCoord[dim] + meshDevice->shape()[dim] - 1) %
                           meshDevice->shape()[dim];
      auto backward_direction = get_eth_forwarding_direction(
          meshDevice->get_fabric_node_id(deviceCoord),
          meshDevice->get_fabric_node_id(backwardCoord));
      LOG_ASSERT(backward_direction.has_value(),
                 "Backward direction does not exist on mesh coordinate: ",
                 deviceCoord);
      rtArgsVec.push_back(backward_direction.value());
    }

    // Add mesh coordinate to device id mapping (in flattened mesh coordinate
    // order)
    auto coord_range =
        tt_metal::distributed::MeshCoordinateRange(meshDevice->shape());
    for (auto coord = coord_range.begin(); coord != coord_range.end();
         coord++) {
      rtArgsVec.push_back(meshDevice->get_fabric_node_id(*coord).chip_id);
    }

    api_type = tt::tt_fabric::FabricApiType::Linear;
    // update number of topology args
    rtArgsVec[num_topology_arg_idx] = (rtArgsVec.size() - num_topology_arg_idx);
  } else {
    LOG_ASSERT(false, EnumNameTopology(topology_type),
               " is not a supported topology");
  }

  // insert fabric connection args (device and core specific)
  std::vector<tt::tt_metal::CoreCoord> cores =
      tt::tt_metal::corerange_to_cores(coreRangeSet);
  LOG_ASSERT(cores.size() <= num_links, "Number of cores (", cores.size(),
             ") to connect to fabric routers exceeds number of routing "
             "planes available (",
             num_links, ")");
  for (uint32_t i = 0; i < cores.size(); i++) {
    std::vector<uint32_t> rtArgsVecPerCore = rtArgsVec;
    auto num_fabric_connection_arg_idx = rtArgsVecPerCore.size();
    // push arg placeholder to store number of fabric connection args
    rtArgsVecPerCore.push_back(1);

    tt::tt_fabric::FabricNodeId src_fabric_node_id(
        meshDevice->get_fabric_node_id(deviceCoord));

    std::vector<tt::tt_fabric::eth_chan_directions> all_routing_directions = {
        tt::tt_fabric::eth_chan_directions::EAST,
        tt::tt_fabric::eth_chan_directions::WEST,
        tt::tt_fabric::eth_chan_directions::NORTH,
        tt::tt_fabric::eth_chan_directions::SOUTH};
    // push arg placeholder to store number of connections
    rtArgsVecPerCore.push_back(0);
    uint32_t num_connections =
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
            src_fabric_node_id, all_routing_directions, {i}, program, handle,
            {cores[i]}, rtArgsVecPerCore, api_type);
    // update number of connections
    rtArgsVecPerCore[num_fabric_connection_arg_idx + 1] = num_connections;
    // update number of fabric connection args
    rtArgsVecPerCore[num_fabric_connection_arg_idx] =
        (rtArgsVecPerCore.size() - num_fabric_connection_arg_idx);
    fabricConfigArgs[cores[i]] = rtArgsVecPerCore;
  }
  return fabricConfigArgs;
}

template std::unordered_map<::tt::tt_metal::CoreCoord, std::vector<uint32_t>>
appendFabricConfigArgs<tt::tt_metal::Program>(
    const ::tt::target::FabricConnectionConfig *fabricConnectionConfig,
    const target::metal::KernelConfig *kernelConfig,
    tt::tt_metal::Program &program, tt_metal::KernelHandle &handle,
    const tt_metal::distributed::MeshCoordinate deviceCoord,
    const tt_metal::distributed::MeshDevice *meshDevice,
    std::vector<uint32_t> rtArgsVec,
    const tt::tt_metal::CoreRangeSet &coreRangeSet);

template std::unordered_map<::tt::tt_metal::CoreCoord, std::vector<uint32_t>>
appendFabricConfigArgs<tt::tt_metal::ProgramDescriptor>(
    const ::tt::target::FabricConnectionConfig *fabricConnectionConfig,
    const target::metal::KernelConfig *kernelConfig,
    tt::tt_metal::ProgramDescriptor &program, tt_metal::KernelHandle &handle,
    const tt_metal::distributed::MeshCoordinate deviceCoord,
    const tt_metal::distributed::MeshDevice *meshDevice,
    std::vector<uint32_t> rtArgsVec,
    const tt::tt_metal::CoreRangeSet &coreRangeSet);

} // namespace tt::runtime::common
