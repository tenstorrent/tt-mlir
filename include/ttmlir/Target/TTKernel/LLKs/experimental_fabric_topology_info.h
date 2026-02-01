// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_FABRIC_TOPOLOGY_INFO_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_FABRIC_TOPOLOGY_INFO_H

/////////////////// Topology Info Struct and Functions ////////////////////
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include <array>

// hardcoded defaults for now
static constexpr uint32_t max_fabric_payload_size = 4352;
static constexpr size_t fabric_setup_args_start_idx = 0;
static constexpr size_t NUM_DIMS = 2;
static constexpr size_t MAX_SEND_DIR = 4;

struct TopologyInfo {
  enum class TopologyType { Ring = 0, Line, Mesh, Torus };

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

  TopologyType topology_type;
  RoutingMode routing_mode;
  RoutingDirection routing_direction;
  uint32_t axis; // dim to route on for 1d TopologyType's
  std::array<uint32_t, NUM_DIMS> mesh_shape;
  std::array<std::pair<eth_chan_directions, eth_chan_directions>, NUM_DIMS>
      routing_directions;
  static constexpr size_t MAX_MESH_SIZE = 32;
  uint32_t flattened_mesh_coordinate_to_device_id[MAX_MESH_SIZE];
  uint32_t device_id_to_flattened_mesh_coordinate[MAX_MESH_SIZE];

  uint32_t get_device_id(std::array<uint32_t, NUM_DIMS> logical_mesh_position) {
    // flatten logical_mesh_position first
    uint32_t flattened_mesh_coordinate = 0;
    for (uint32_t i = 0; i < NUM_DIMS; i++) {
      WAYPOINT("DA01");
      ASSERT(logical_mesh_position[i] < mesh_shape[i]);
      flattened_mesh_coordinate =
          flattened_mesh_coordinate * mesh_shape[i] + logical_mesh_position[i];
    }
    return flattened_mesh_coordinate_to_device_id[flattened_mesh_coordinate];
  }

  std::array<uint32_t, NUM_DIMS> get_logical_mesh_position(uint32_t device_id) {
    WAYPOINT("DA02");
    ASSERT(device_id < MAX_MESH_SIZE);
    uint32_t flattened_mesh_coordinate =
        device_id_to_flattened_mesh_coordinate[device_id];
    // unflatten mesh_coordinate
    std::array<uint32_t, NUM_DIMS> logical_mesh_position;
    for (int32_t i = NUM_DIMS - 1; i >= 0; i--) {
      logical_mesh_position[i] = flattened_mesh_coordinate % mesh_shape[i];
      flattened_mesh_coordinate /= mesh_shape[i];
      WAYPOINT("DA03");
      ASSERT(logical_mesh_position[i] < mesh_shape[i]);
    }
    return logical_mesh_position;
  }

  void build_from_args(size_t &rt_arg_idx) {
    // Read topology type (Line=0, Ring=1) and axis (for 1D)
    topology_type =
        static_cast<TopologyType>(get_arg_val<uint32_t>(rt_arg_idx++));
    axis = get_arg_val<uint32_t>(rt_arg_idx++);
    // Read routing mode
    routing_mode =
        static_cast<RoutingMode>(get_arg_val<uint32_t>(rt_arg_idx++));
    // Read routing direction
    routing_direction =
        static_cast<RoutingDirection>(get_arg_val<uint32_t>(rt_arg_idx++));

    // Read mesh shape
    uint32_t mesh_size = 1;
    for (uint32_t i = 0; i < NUM_DIMS; i++) {
      mesh_shape[i] = get_arg_val<uint32_t>(rt_arg_idx++);
      mesh_size = mesh_size * mesh_shape[i];
    }
    WAYPOINT("DA04");
    ASSERT(mesh_size <= MAX_MESH_SIZE);

    // Read directions
    for (uint32_t i = 0; i < NUM_DIMS; i++) {
      auto forward_dir =
          static_cast<eth_chan_directions>(get_arg_val<uint32_t>(rt_arg_idx++));
      auto backward_dir =
          static_cast<eth_chan_directions>(get_arg_val<uint32_t>(rt_arg_idx++));
      routing_directions[i] = {forward_dir, backward_dir};
    }

    // Read logical index to device id mapping and build reverse mapping
    // (device ids are provided in flattened mesh coordinate order)
    for (uint32_t i = 0; i < mesh_size; i++) {
      uint32_t device_id = get_arg_val<uint32_t>(rt_arg_idx++);
      flattened_mesh_coordinate_to_device_id[i] = device_id;
      WAYPOINT("DA05");
      ASSERT(device_id < MAX_MESH_SIZE);
      device_id_to_flattened_mesh_coordinate[device_id] = i;
    }
  }
};

FORCE_INLINE std::tuple<int32_t, int32_t, int32_t, int32_t>
get_ring_regions(int32_t my_idx, int32_t start_idx, int32_t end_idx,
                 int32_t size, bool is_forward) {
  WAYPOINT("DA06");
  ASSERT(my_idx != start_idx && my_idx != end_idx);
  bool in_between = ((my_idx - start_idx + size) % size) <
                    ((end_idx - start_idx + size) % size);
  if (!in_between) {
    // Forward: hit start_idx first, backward: hit end_idx first
    int32_t start_hop = is_forward ? (start_idx - my_idx + size) % size
                                   : (my_idx - end_idx + size) % size;
    uint32_t range = (end_idx - start_idx + size) % size + 1;
    WAYPOINT("DA07");
    ASSERT(start_hop + -1 < size);
    return {start_hop, range, 1, 0};
  } else {
    // send till end for forward
    int32_t start_1 = 1;
    int32_t range_1 = is_forward ? (end_idx - my_idx + size) % size
                                 : (my_idx - start_idx + size) % size;
    // hops from end device to start
    int32_t start_2_gap = (start_idx - end_idx + size) % size;
    // num devices in 2nd range: start device back to just before my device (for
    // forward)
    int32_t range_2 = is_forward ? (my_idx - start_idx + size) % size
                                 : (end_idx - my_idx + size) % size;
    WAYPOINT("DA08");
    ASSERT(start_1 + range_1 - 1 + start_2_gap + range_2 - 1 < size);
    return {start_1, range_1, start_2_gap, range_2};
  }
}

#endif
