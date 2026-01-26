// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_FABRIC_API_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_FABRIC_API_H

#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include <array>

namespace experimental {

// hardcoded defaults for now
static constexpr uint32_t max_fabric_payload_size = 4352;
static constexpr size_t fabric_setup_args_start_idx = 0;
static constexpr size_t NUM_DIMS = 2;

/////////////// Exposed APIs (for TTKernel Dialect Ops) ////////////////
//
// FabricConnectionManager:
//   FabricConnectionManager fcm;    // create_fabric_connection_manager
//   setup_fabric_connections(fcm)   // setup_fabric_connections
//   close_fabric_connections(fcm)   // close_fabric_connections
//
// Device/Topology Info:
//   uint16_t get_my_device_id()
//   std::array<uint32_t, NUM_DIMS> get_logical_mesh_position()
//   uint32_t get_device_id_from_logical_mesh_position(...)
//
// Fabric Write:
//   void fabric_fast_write_any_len(fcm, dst_mesh_id, dst_dev_id, ...)
//
////////////////////////////////////////////////////////////////////////

/////////////////// Device/Topology Info Functions ////////////////////

FORCE_INLINE uint16_t get_my_device_id() {
  auto *routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t *>(
      MEM_TENSIX_ROUTING_TABLE_BASE);
  return routing_table->my_device_id;
}

struct TopologyInfo {
  enum class TopologyType { Ring = 0, Line };

  TopologyType topology_type;
  uint32_t axis; // dim to route on for 1d TopologyType's
  std::array<uint32_t, NUM_DIMS> mesh_shape;
  std::array<std::pair<eth_chan_directions, eth_chan_directions>, NUM_DIMS> routing_directions;
  static constexpr size_t MAX_MESH_SIZE = 32;
  uint32_t flattened_mesh_coordinate_to_device_id[MAX_MESH_SIZE];
  uint32_t device_id_to_flattened_mesh_coordinate[MAX_MESH_SIZE];

  uint32_t get_device_id(std::array<uint32_t, NUM_DIMS> logical_mesh_position) {
    // flatten logical_mesh_position first
    uint32_t flattened_mesh_coordinate = 0;
    for (uint32_t i = 0; i < NUM_DIMS; i++) {
      ASSERT(logical_mesh_position[i] < mesh_shape[i]);
      flattened_mesh_coordinate = flattened_mesh_coordinate * mesh_shape[i] + logical_mesh_position[i]; 
    }
    return flattened_mesh_coordinate_to_device_id[flattened_mesh_coordinate];
  }

  std::array<uint32_t, NUM_DIMS> get_logical_mesh_position(uint32_t device_id) {
    ASSERT(device_id < MAX_MESH_SIZE);
    uint32_t flattened_mesh_coordinate = device_id_to_flattened_mesh_coordinate[device_id];
    // unflatten mesh_coordinate
    std::array<uint32_t, NUM_DIMS> logical_mesh_position;
    for (int32_t i = NUM_DIMS - 1; i >= 0; i--) {
      logical_mesh_position[i] = flattened_mesh_coordinate % mesh_shape[i];
      flattened_mesh_coordinate /= mesh_shape[i];
      ASSERT(logical_mesh_position[i] < mesh_shape[i]);
    }
    return logical_mesh_position;
  }

  void build_from_args(size_t &rt_arg_idx) {
    // Read topology type (Line=0, Ring=1) and axis (for 1D)
    topology_type =
        static_cast<TopologyType>(get_arg_val<uint32_t>(rt_arg_idx++));
    axis = get_arg_val<uint32_t>(rt_arg_idx++);

    // Read mesh shape
    uint32_t mesh_size = 1;
    for (uint32_t i = 0; i < NUM_DIMS; i++) {
      mesh_shape[i] = get_arg_val<uint32_t>(rt_arg_idx++);
      mesh_size = mesh_size * mesh_shape[i]; 
    }
    ASSERT(mesh_size <= MAX_MESH_SIZE);

    // Read directions
    for (uint32_t i = 0; i < NUM_DIMS; i++) {
      auto forward_dir = static_cast<eth_chan_directions>(get_arg_val<uint32_t>(rt_arg_idx++));
      auto backward_dir = static_cast<eth_chan_directions>(get_arg_val<uint32_t>(rt_arg_idx++));
      routing_directions[i] = {forward_dir, backward_dir};
    }

    // Read logical index to device id mapping and build reverse mapping
    // (device ids are provided in flattened mesh coordinate order)
    for (uint32_t i = 0; i < mesh_size; i++) {
      uint32_t device_id = get_arg_val<uint32_t>(rt_arg_idx++);
      flattened_mesh_coordinate_to_device_id[i] = device_id;
      ASSERT(device_id < MAX_MESH_SIZE);
      device_id_to_flattened_mesh_coordinate[device_id] = i;
    }
  }
};

////////// FabricConnectionManager and Setup/Teardown Functions //////////

struct FabricConnectionManager {
  TopologyInfo topology_info;
  tt::tt_fabric::RoutingPlaneConnectionManager fabric_connections;
  int route_id = -1;
  bool initialized = false;

  FORCE_INLINE TopologyInfo &get_topology() {
    ASSERT(initialized);
    return topology_info;
  }

  FORCE_INLINE RoutingPlaneConnectionManager &get_fabric_connections() {
    ASSERT(initialized);
    return fabric_connections;
  }

  FORCE_INLINE volatile tt_l1_ptr PACKET_HEADER_TYPE *get_header(uint32_t dir) {
    ASSERT(initialized);
    return PacketHeaderPool::header_table[route_id].first + dir;
  }
};

FORCE_INLINE std::array<uint32_t, NUM_DIMS> get_logical_mesh_position(FabricConnectionManager &fabric_connection_manager, uint32_t device_id) {
  return fabric_connection_manager.get_topology().get_logical_mesh_position(get_my_device_id());
}

FORCE_INLINE uint32_t
get_device_id_from_logical_mesh_position(FabricConnectionManager &fabric_connection_manager, std::array<uint32_t, NUM_DIMS> logical_mesh_position) {
  return fabric_connection_manager.get_topology().get_device_id(logical_mesh_position);
}

FORCE_INLINE void
setup_fabric_connections(FabricConnectionManager &fabric_connection_manager) {
  uint32_t num_topology_args =
      get_arg_val<uint32_t>(fabric_setup_args_start_idx);
  uint32_t num_fabric_connection_args =
      get_arg_val<uint32_t>(fabric_setup_args_start_idx + num_topology_args);
  uint32_t num_send_dir = get_arg_val<uint32_t>(fabric_setup_args_start_idx +
                                                num_topology_args + 1);

  if (!fabric_connection_manager.initialized) {
    // set up topology
    size_t topology_arg_idx = fabric_setup_args_start_idx + 1;
    fabric_connection_manager.topology_info.build_from_args(
        topology_arg_idx);

    // set up routing plane connection manager
    size_t fabric_connection_arg_idx =
        fabric_setup_args_start_idx + num_topology_args + 2;
    fabric_connection_manager.fabric_connections =
        tt::tt_fabric::RoutingPlaneConnectionManager::template build_from_args<
            tt::tt_fabric::RoutingPlaneConnectionManager::
                BUILD_AND_OPEN_CONNECTION>(fabric_connection_arg_idx,
                                           num_send_dir);

    // set up packet header pool
    fabric_connection_manager.route_id =
        PacketHeaderPool::allocate_header_n(num_send_dir);
    ASSERT(fabric_connection_manager.route_id != -1);
  }
  fabric_connection_manager.initialized = true;
}

// teardown fabric connections (packet header pool and topology don't need
// teardown)
FORCE_INLINE void
close_fabric_connections(FabricConnectionManager &fabric_connection_manager) {
  if (fabric_connection_manager.initialized) {
    fabric_connection_manager.fabric_connections.close();
  }
}

////////////////// Fabric Write APIs (and helpers) /////////////////////

#ifdef FABRIC_2D

FORCE_INLINE std::pair<uint32_t, uint32_t> calculate_initial_direction_and_hops(
    TopologyInfo &topology_info, uint16_t dst_device_id, uint16_t my_device_id) {
  auto *routing_info =
      reinterpret_cast<tt_l1_ptr intra_mesh_routing_path_t<2, true> *>(
          ROUTING_PATH_BASE_2D);

  uint32_t initial_dir = static_cast<uint32_t>(eth_chan_directions::EAST);

  const auto &compressed_route = routing_info->paths[dst_device_id];
  uint8_t ns_hops = compressed_route.get_ns_hops();
  uint8_t ew_hops = compressed_route.get_ew_hops();

  if (ns_hops > 0) {
    if (compressed_route.get_ns_direction()) {
      initial_dir = static_cast<uint32_t>(eth_chan_directions::SOUTH);
    } else {
      initial_dir = static_cast<uint32_t>(eth_chan_directions::NORTH);
    }
  } else if (ew_hops > 0) {
    if (compressed_route.get_ew_direction()) {
      initial_dir = static_cast<uint32_t>(eth_chan_directions::EAST);
    } else {
      initial_dir = static_cast<uint32_t>(eth_chan_directions::WEST);
    }
  } else {
    ASSERT(false);
  }

  return {initial_dir, ns_hops + ew_hops};
}

#else // 1D Fabric (only supports line/ring topologies)

FORCE_INLINE std::pair<uint32_t, uint32_t> calculate_initial_direction_and_hops(
    TopologyInfo &topology, uint16_t dst_device_id, uint16_t my_device_id) {
#if defined(API_TYPE_Linear)
  // TODO: check that my_device_id and dst_device_id are in same line/ring
  uint32_t my_logical_index = topology.get_logical_mesh_position(my_device_id)[topology.axis];
  uint32_t dest_logical_index = topology.get_logical_mesh_position(dst_device_id)[topology.axis];
  if (topology.topology_type == TopologyInfo::TopologyType::Line) {
    if (my_logical_index < dest_logical_index) {
      return {static_cast<uint32_t>(topology.routing_directions[topology.axis].first),
              dest_logical_index - my_logical_index};
    } else {
      return {static_cast<uint32_t>(topology.routing_directions[topology.axis].second),
              my_logical_index - dest_logical_index};
    }
  } else { // TopologyInfo::TopologyType::Ring; select shortest route from
           // the two directions
    uint32_t ring_size = topology.mesh_shape[topology.axis];
    if ((dest_logical_index - my_logical_index) % ring_size <
        (my_logical_index - dest_logical_index) % ring_size) {
      return {static_cast<uint32_t>(topology.routing_directions[topology.axis].first),
              (dest_logical_index - my_logical_index) % ring_size};
    } else {
      return {static_cast<uint32_t>(topology.routing_directions[topology.axis].second),
              (dest_logical_index - my_logical_index) % ring_size};
    }
  }
#else
#error                                                                         \
    "Only API_TYPE_Linear (i.e. line and ring topologies) supported for 1D Fabric Config"
#endif
}

#endif

FORCE_INLINE int
get_connection_index_by_tag(RoutingPlaneConnectionManager &fabric_connections,
                            uint32_t tag) {
  for (uint32_t i = 0; i < fabric_connections.active_count(); ++i) {
    if (fabric_connections.get(i).tag == tag) {
      return i;
    }
  }

  ASSERT(false);
  return -1;
}

FORCE_INLINE void
fabric_fast_write(WorkerToFabricEdmSender &connection,
                  volatile tt_l1_ptr PACKET_HEADER_TYPE *packet_header,
                  uint32_t src_addr, uint64_t dest_addr, uint32_t len_bytes,
                  bool multicast = false, uint32_t num_dests = 1) {
  if (multicast) {
    // TODO: Set up multicast header with proper routing
    ASSERT(false);
    while (1) {
    }
  } else {
    packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_addr},
                                        len_bytes);
  }

  connection.wait_for_empty_write_slot();
  connection.send_payload_without_header_non_blocking_from_address(src_addr,
                                                                   len_bytes);
  connection.send_payload_flush_non_blocking_from_address(
      reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));

  noc_async_writes_flushed();
}

// Note: in ring and torus fabric, there are multiple routes but we are using
// shortest path route
FORCE_INLINE void
fabric_fast_write_any_len(FabricConnectionManager &fabric_connection_manager,
                          uint16_t dst_mesh_id, uint16_t dst_dev_id,
                          uint64_t dest_addr, uint32_t src_addr,
                          uint32_t len_bytes) {
  tt_l1_ptr routing_l1_info_t *routing_table =
      reinterpret_cast<tt_l1_ptr routing_l1_info_t *>(
          MEM_TENSIX_ROUTING_TABLE_BASE);
  uint16_t my_device_id = routing_table->my_device_id;
  uint16_t my_mesh_id = routing_table->my_mesh_id;
  ASSERT(my_mesh_id == dst_mesh_id); // we dont support inter-mesh routing yet

  auto [initial_dir, num_hops] = calculate_initial_direction_and_hops(
      fabric_connection_manager.get_topology(), dst_dev_id, my_device_id);
  auto connection_index = get_connection_index_by_tag(
      fabric_connection_manager.get_fabric_connections(), initial_dir);
  ASSERT(connection_index != -1);
  auto &connection = fabric_connection_manager.get_fabric_connections()
                         .get(connection_index)
                         .sender;
  volatile tt_l1_ptr PACKET_HEADER_TYPE *packet_header =
      fabric_connection_manager.get_header(connection_index);
#ifdef FABRIC_2D
  bool result = fabric_set_unicast_route(
      static_cast<volatile tt_l1_ptr HybridMeshPacketHeader *>(packet_header),
      dst_dev_id, dst_mesh_id);
#else // 1D fabric
  static_cast<volatile tt_l1_ptr LowLatencyPacketHeader *>(packet_header)
      ->to_chip_unicast(num_hops);
#endif

  while (len_bytes > max_fabric_payload_size) {
    fabric_fast_write(connection, packet_header, src_addr, dest_addr,
                      max_fabric_payload_size);

    src_addr += max_fabric_payload_size;
    dest_addr += max_fabric_payload_size;
    len_bytes -= max_fabric_payload_size;
  }
  fabric_fast_write(connection, packet_header, src_addr, dest_addr, len_bytes);
}

} // namespace experimental

#endif
