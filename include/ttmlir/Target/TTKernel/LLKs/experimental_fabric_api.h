// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_FABRIC_API_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_FABRIC_API_H

// #include "experimental_fabric_topology_info.h"

namespace experimental {

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
//   void fabric_mcast_fast_write_any_len(fcm, dst_mesh_id, ...)
//
////////////////////////////////////////////////////////////////////////

////////// FabricConnectionManager and Setup/Teardown Functions //////////

FORCE_INLINE uint16_t get_my_device_id() {
  auto *routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t *>(
      MEM_TENSIX_ROUTING_TABLE_BASE);
  return routing_table->my_device_id;
}

// TODO: remove this helper and use a array map
FORCE_INLINE int
get_connection_index_by_tag(RoutingPlaneConnectionManager &fabric_connections,
                            uint32_t tag) {
  for (uint32_t i = 0; i < fabric_connections.active_count(); ++i) {
    if (fabric_connections.get(i).tag == tag) {
      return i;
    }
  }

  WAYPOINT("DA13");
  ASSERT(false);
  return -1;
}

struct FabricConnectionManager {
  TopologyInfo topology_info;
  tt::tt_fabric::RoutingPlaneConnectionManager fabric_connections;
  int route_id = -1;
  bool initialized = false;

  FORCE_INLINE TopologyInfo &get_topology() {
    WAYPOINT("DA14");
    ASSERT(initialized);
    return topology_info;
  }

  FORCE_INLINE std::pair<WorkerToFabricEdmSender &,
                         volatile tt_l1_ptr PACKET_HEADER_TYPE *>
  get_connection_and_packet_header(uint32_t dir) {
    WAYPOINT("DA15");
    ASSERT(initialized);
    auto connection_index =
        get_connection_index_by_tag(fabric_connections, dir);
    WAYPOINT("DA16");
    ASSERT(connection_index != -1);
    volatile tt_l1_ptr PACKET_HEADER_TYPE *packet_header =
        PacketHeaderPool::header_table[route_id].first + connection_index;
    auto &connection = fabric_connections.get(connection_index).sender;
    return {connection, packet_header};
  }

  FORCE_INLINE RoutingPlaneConnectionManager &get_fabric_connections() {
    WAYPOINT("DA17");
    ASSERT(initialized);
    return fabric_connections;
  }

  FORCE_INLINE volatile tt_l1_ptr PACKET_HEADER_TYPE *get_header(uint32_t dir) {
    WAYPOINT("DA18");
    ASSERT(initialized);
    return PacketHeaderPool::header_table[route_id].first + dir;
  }
};

FORCE_INLINE std::array<uint32_t, NUM_DIMS>
get_logical_mesh_position(FabricConnectionManager &fabric_connection_manager,
                          uint32_t device_id) {
  return fabric_connection_manager.get_topology().get_logical_mesh_position(
      get_my_device_id());
}

FORCE_INLINE uint32_t get_device_id_from_logical_mesh_position(
    FabricConnectionManager &fabric_connection_manager,
    std::array<uint32_t, NUM_DIMS> logical_mesh_position) {
  return fabric_connection_manager.get_topology().get_device_id(
      logical_mesh_position);
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
    fabric_connection_manager.topology_info.build_from_args(topology_arg_idx);

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
    WAYPOINT("DA19");
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

////////////////// Routing Helper Functions (emitted separately)
////////////////////////

////////////////// Fabric Write APIs (and helpers) /////////////////////

FORCE_INLINE void
fabric_fast_write(WorkerToFabricEdmSender &connection,
                  volatile tt_l1_ptr PACKET_HEADER_TYPE *packet_header,
                  uint32_t src_addr, uint64_t dest_addr, uint32_t len_bytes) {
  packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_addr},
                                      len_bytes);

  connection.wait_for_empty_write_slot();
  connection.send_payload_without_header_non_blocking_from_address(src_addr,
                                                                   len_bytes);
  connection.send_payload_flush_non_blocking_from_address(
      reinterpret_cast<uint32_t>(packet_header), sizeof(PACKET_HEADER_TYPE));

  noc_async_writes_flushed(); // TODO: remove this???
}

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
  WAYPOINT("DA21");
  ASSERT(my_mesh_id == dst_mesh_id); // we dont support inter-mesh routing yet

  auto unicast_params = get_unicast_params(
      fabric_connection_manager.get_topology(), my_device_id, dst_dev_id);
  auto [connection, packet_header] =
      fabric_connection_manager.get_connection_and_packet_header(
          unicast_params.outgoing_direction);
#ifdef FABRIC_2D
  fabric_set_unicast_route_custom(
      static_cast<volatile tt_l1_ptr HybridMeshPacketHeader *>(packet_header),
      dst_dev_id, dst_mesh_id, unicast_params.ns_hops, unicast_params.ew_hops,
      unicast_params.ns_dir, unicast_params.ew_dir);
#else // 1D fabric
  static_cast<volatile tt_l1_ptr LowLatencyPacketHeader *>(packet_header)
      ->to_chip_unicast(unicast_params.num_hops);
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

// Conditions:
// - dest_start_logical_index <= dest_end_logical_index in all dimensions
// - there is at least one destination (not including sender);
FORCE_INLINE void fabric_mcast_fast_write_any_len(
    FabricConnectionManager &fabric_connection_manager, uint16_t dst_mesh_id,
    uint16_t dst_dev_id_start, uint16_t dst_dev_id_end, uint64_t dest_addr,
    uint32_t src_addr, uint32_t len_bytes) {
  tt_l1_ptr routing_l1_info_t *routing_table =
      reinterpret_cast<tt_l1_ptr routing_l1_info_t *>(
          MEM_TENSIX_ROUTING_TABLE_BASE);
  uint16_t my_device_id = routing_table->my_device_id;
  uint16_t my_mesh_id = routing_table->my_mesh_id;
  WAYPOINT("DA22");
  ASSERT(my_mesh_id == dst_mesh_id); // we dont support inter-mesh routing yet

  // Get routing info and set up headers for each directions
  auto mcast_params =
      get_mcast_params(fabric_connection_manager.get_topology(), my_device_id,
                       dst_dev_id_start, dst_dev_id_end);
  for (uint32_t i = 0; i < MAX_SEND_DIR; i++) {
    if (mcast_params.params_per_direction[i].active) {
      auto [connection, packet_header] =
          fabric_connection_manager.get_connection_and_packet_header(i);
#ifdef FABRIC_2D
      fabric_set_mcast_route(
          static_cast<volatile tt_l1_ptr HybridMeshPacketHeader *>(
              packet_header),
          my_device_id, // TODO: what should this even be? (only relevant for
                        // inter-mesh)
          my_mesh_id,   // TODO: what should this even be? (only relevant for
                        // inter-mesh)
          mcast_params.params_per_direction[i].e_num_hops,
          mcast_params.params_per_direction[i].w_num_hops,
          mcast_params.params_per_direction[i].n_num_hops,
          mcast_params.params_per_direction[i].s_num_hops);
#else // 1D fabric
      static_cast<volatile tt_l1_ptr LowLatencyPacketHeader *>(packet_header)
          ->to_chip_multicast(
              mcast_params.params_per_direction[i].mcast_command_header);
#endif
    }
  }

  while (len_bytes > max_fabric_payload_size) {
    for (uint32_t i = 0; i < MAX_SEND_DIR; i++) {
      if (mcast_params.params_per_direction[i].active) {
        auto [connection, packet_header] =
            fabric_connection_manager.get_connection_and_packet_header(i);
        fabric_fast_write(connection, packet_header, src_addr, dest_addr,
                          max_fabric_payload_size);
      }
    }

    src_addr += max_fabric_payload_size;
    dest_addr += max_fabric_payload_size;
    len_bytes -= max_fabric_payload_size;
  }

  for (uint32_t i = 0; i < MAX_SEND_DIR; i++) {
    if (mcast_params.params_per_direction[i].active) {
      auto [connection, packet_header] =
          fabric_connection_manager.get_connection_and_packet_header(i);
      fabric_fast_write(connection, packet_header, src_addr, dest_addr,
                        len_bytes);
    }
  }
}

} // namespace experimental

#endif
