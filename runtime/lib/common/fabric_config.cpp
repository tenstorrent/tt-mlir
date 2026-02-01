// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/fabric_config.h"
#include "tt-metalium/program_descriptors.hpp"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/Target/Common/types_generated.h"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::runtime::common {

// only for ring/torus topologies
enum class RoutingMode {
  ShortestPath = 0,
  UnidirectionalRingTorus,
};

// only for ring/torus topologies and UnidirectionalRingTorus routing mode
enum class RoutingDirection {
  SouthEast,
  NorthWest,
};

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

  if (topology_type != tt::target::Topology::Linear &&
      topology_type != tt::target::Topology::Ring &&
      topology_type != tt::target::Topology::Mesh &&
      topology_type != tt::target::Topology::Torus) {
    LOG_ASSERT(false, EnumNameTopology(topology_type),
               " is not a supported topology");
  }

  // insert topology specific args (device specific)
  auto num_topology_arg_idx = rtArgsVec.size();
  rtArgsVec.push_back(1);
  LOG_ASSERT(meshDevice->shape().dims() == 2,
             "Only 2d mesh device is supported");
  RoutingMode routing_mode = RoutingMode::UnidirectionalRingTorus;
  std::vector<
      std::pair<tt_fabric::eth_chan_directions, tt_fabric::eth_chan_directions>>
      routing_directions;

  // Add topology type (Line=0, Ring=1) and axis (for 1D)
  rtArgsVec.push_back(static_cast<uint32_t>(topology_type));
  LOG_ASSERT(cluster_axis < 2, "Invalid cluster axis, must be < 2");
  rtArgsVec.push_back(cluster_axis);
  // Add routing mode
  rtArgsVec.push_back(static_cast<uint32_t>(routing_mode));
  // Add placeholder for routing direction (will be updated later if
  // UnidirectionalRingTorus routing mode)
  uint32_t routing_direction_idx = rtArgsVec.size();
  rtArgsVec.push_back(0);

  // add mesh shape
  for (uint32_t dim = 0; dim < meshDevice->shape().dims(); dim++) {
    rtArgsVec.push_back(meshDevice->shape()[dim]);
  }

  // add forward and backward directions for each dim
  // we don't have topology verification, that's something that we should add
  // and expose for users too! this is especially problematic for t3k folded
  // case
  for (uint32_t dim = 0; dim < meshDevice->shape().dims(); dim++) {
    routing_directions.push_back(
        std::make_pair(tt_fabric::eth_chan_directions::COUNT,
                       tt_fabric::eth_chan_directions::COUNT));

    // Forward direction
    // don't set forward for edge coords on line even if physical link exists
    if ((topology_type == tt::target::Topology::Linear ||
         topology_type == tt::target::Topology::Mesh) &&
        deviceCoord[dim] == meshDevice->shape()[dim] - 1) {
      rtArgsVec.push_back(-1);
    } else {
      auto forwardCoord = deviceCoord;
      forwardCoord[dim] = (forwardCoord[dim] + 1) % meshDevice->shape()[dim];
      // what i really want to know is whether the forward coord is a neighbour
      // (inferrable from topo?) and to get dir
      auto forward_directions = get_neighbour_eth_directions(
          meshDevice->get_fabric_node_id(deviceCoord),
          meshDevice->get_fabric_node_id(forwardCoord));

      // edge case: if dim size is 1, then neither forward nor backward exist
      if (meshDevice->shape()[dim] == 1) {
        rtArgsVec.push_back(-1);
      }
      // edge case: if dim size is 2, then check direction is not the same
      // (opposites)
      else if (meshDevice->shape()[dim] == 2) {
        // we can technically get both forward and backward directions here
        // since dim is 2 so assign SE for forward and NW for backward
        LOG_ASSERT(
            forward_directions.size() == 1 || forward_directions.size() == 2,
            "Number of forward directions is invalid on mesh coordinate: ",
            deviceCoord);
        if (std::find(forward_directions.begin(), forward_directions.end(),
                      tt_fabric::eth_chan_directions::SOUTH) !=
            forward_directions.end()) {
          rtArgsVec.push_back(tt_fabric::eth_chan_directions::SOUTH);
          routing_directions[dim].first = tt_fabric::eth_chan_directions::SOUTH;
        } else if (std::find(forward_directions.begin(),
                             forward_directions.end(),
                             tt_fabric::eth_chan_directions::EAST) !=
                   forward_directions.end()) {
          rtArgsVec.push_back(tt_fabric::eth_chan_directions::EAST);
          routing_directions[dim].first = tt_fabric::eth_chan_directions::EAST;
        } else {
          LOG_ASSERT(false, "Forward direction is missing on mesh coordinate: ",
                     deviceCoord);
        }
      } else {
        LOG_ASSERT(
            forward_directions.size() == 1,
            "Number of forward directions is invalid on mesh coordinate: ",
            deviceCoord);
        rtArgsVec.push_back(forward_directions[0]);
        routing_directions[dim].first = forward_directions[0];
      }
    }

    // Backward direction
    // don't set backward dir for edge coord on line/mesh even if physical link
    // exists
    if ((topology_type == tt::target::Topology::Linear ||
         topology_type == tt::target::Topology::Mesh) &&
        deviceCoord[dim] == 0) {
      rtArgsVec.push_back(
          -1); // this is technically wrong since its unsigned!!!!
    } else {
      auto backwardCoord = deviceCoord;
      backwardCoord[dim] = (backwardCoord[dim] + meshDevice->shape()[dim] - 1) %
                           meshDevice->shape()[dim];
      auto backward_directions = get_neighbour_eth_directions(
          meshDevice->get_fabric_node_id(deviceCoord),
          meshDevice->get_fabric_node_id(backwardCoord));
      ;

      // edge case: if dim size is 1, then neither forward nor backward exist
      if (meshDevice->shape()[dim] == 1) {
        rtArgsVec.push_back(-1);
      }
      // edge case: if dim size is 2, then check direction is not the same
      // (opposites)
      else if (meshDevice->shape()[dim] == 2) {
        // we can technically get both forward and backward directions here
        // since dim is 2 so assign SE for forward and NW for backward
        LOG_ASSERT(
            backward_directions.size() == 1 || backward_directions.size() == 2,
            "Number of backward directions is invalid on mesh coordinate: ",
            deviceCoord);
        if (std::find(backward_directions.begin(), backward_directions.end(),
                      tt_fabric::eth_chan_directions::NORTH) !=
            backward_directions.end()) {
          rtArgsVec.push_back(tt_fabric::eth_chan_directions::NORTH);
          routing_directions[dim].second =
              tt_fabric::eth_chan_directions::NORTH;
        } else if (std::find(backward_directions.begin(),
                             backward_directions.end(),
                             tt_fabric::eth_chan_directions::WEST) !=
                   backward_directions.end()) {
          rtArgsVec.push_back(tt_fabric::eth_chan_directions::WEST);
          routing_directions[dim].second = tt_fabric::eth_chan_directions::WEST;
        } else {
          LOG_ASSERT(false,
                     "Backward direction is missing on mesh coordinate: ",
                     deviceCoord);
        }
      } else {
        LOG_ASSERT(
            backward_directions.size() == 1,
            "Number of backward directions is invalid on mesh coordinate: ",
            deviceCoord);
        rtArgsVec.push_back(backward_directions[0]);
        routing_directions[dim].second = backward_directions[0];
      }
    }

    LOG_INFO("Forward direction: ", rtArgsVec[rtArgsVec.size() - 2]);
    LOG_INFO("Backward direction: ", rtArgsVec[rtArgsVec.size() - 1]);
  }

  // Add mesh coordinate to device id mapping (in flattened mesh coordinate
  // order)
  auto coord_range =
      tt_metal::distributed::MeshCoordinateRange(meshDevice->shape());
  for (auto coord = coord_range.begin(); coord != coord_range.end(); coord++) {
    rtArgsVec.push_back(meshDevice->get_fabric_node_id(*coord).chip_id);
  }

  // update number of topology args
  rtArgsVec[num_topology_arg_idx] = (rtArgsVec.size() - num_topology_arg_idx);

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
    // push arg placeholder to store number of connections
    rtArgsVecPerCore.push_back(0);
    std::vector<tt_fabric::eth_chan_directions> connection_directions;
    for (uint32_t dim = 0; dim < meshDevice->shape().dims(); dim++) {
      if (routing_mode == RoutingMode::ShortestPath ||
          topology_type == target::Topology::Linear ||
          topology_type == target::Topology::Mesh) {
        if (routing_directions[dim].first !=
            tt_fabric::eth_chan_directions::COUNT) {
          connection_directions.push_back(routing_directions[dim].first);
        }
        if (routing_directions[dim].second !=
            tt_fabric::eth_chan_directions::COUNT) {
          connection_directions.push_back(routing_directions[dim].second);
        }
      } else {
        // arbitrarily choose between the two directions for now
        if (i % 2 == 0 && routing_directions[dim].first !=
                              tt_fabric::eth_chan_directions::COUNT) {
          // set south east routing mode
          connection_directions.push_back(routing_directions[dim].first);
          rtArgsVecPerCore[routing_direction_idx] =
              static_cast<uint32_t>(RoutingDirection::SouthEast);
        } else if (i % 2 == 1 && routing_directions[dim].second !=
                                     tt_fabric::eth_chan_directions::COUNT) {
          // set north west routing mode
          connection_directions.push_back(routing_directions[dim].second);
          rtArgsVecPerCore[routing_direction_idx] =
              static_cast<uint32_t>(RoutingDirection::NorthWest);
        }
      }
    }
    uint32_t num_connections =
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
            src_fabric_node_id, connection_directions, {i}, program, handle,
            {cores[i]}, rtArgsVecPerCore, tt::tt_fabric::FabricApiType::Linear);
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
