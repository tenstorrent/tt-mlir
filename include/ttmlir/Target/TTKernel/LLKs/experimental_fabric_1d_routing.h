// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef EXPERIMENTAL_FABRIC_1D_ROUTING_H
#define EXPERIMENTAL_FABRIC_1D_ROUTING_H

#ifndef FABRIC_2D // 1D Fabric

struct UnicastRoutingParams {
  uint32_t outgoing_direction;
  uint32_t num_hops;
};

struct McastRoutingParams {
  struct McastRoutingParamsPerDirection {
    bool active = false;
    tt::tt_fabric::MulticastRoutingCommandHeader mcast_command_header;
  };

  std::array<McastRoutingParamsPerDirection, MAX_SEND_DIR> params_per_direction;
};

// Forward declarations
FORCE_INLINE UnicastRoutingParams get_unicast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_device_id);
FORCE_INLINE UnicastRoutingParams get_unicast_params_line(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_device_id);
FORCE_INLINE McastRoutingParams get_mcast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_start_device_id,
    uint16_t dst_end_device_id);
FORCE_INLINE McastRoutingParams
get_mcast_params_line(TopologyInfo &topology, uint16_t my_device_id,
                      uint16_t dst_start_device_id, uint16_t dst_end_device_id);

FORCE_INLINE UnicastRoutingParams get_unicast_params(TopologyInfo &topology,
                                                     uint16_t my_device_id,
                                                     uint16_t dst_device_id) {
  if (topology.topology_type == TopologyInfo::TopologyType::Ring) {
    // Assert: Only UnidirRingTorus routing mode supported for ring topology.
    WAYPOINT("DA23");
    ASSERT(topology.routing_mode == TopologyInfo::RoutingMode::UnidirRingTorus);
    return get_unicast_params_unidir_ring(topology, my_device_id,
                                          dst_device_id);
  } else if (topology.topology_type == TopologyInfo::TopologyType::Line) {
    // Assert: Only BidirLineMesh routing mode supported for line topology.
    WAYPOINT("DA24");
    ASSERT(topology.routing_mode == TopologyInfo::RoutingMode::BidirLineMesh);
    return get_unicast_params_line(topology, my_device_id, dst_device_id);
  } else {
    // Assert: Unsupported topology type.
    WAYPOINT("DA25");
    ASSERT(false);
    return UnicastRoutingParams();
  }
}

FORCE_INLINE UnicastRoutingParams get_unicast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_device_id) {
  int32_t my_idx =
      topology.get_logical_mesh_position(my_device_id)[topology.axis];
  int32_t dest_idx =
      topology.get_logical_mesh_position(dst_device_id)[topology.axis];
  int32_t size = topology.mesh_shape[topology.axis];
  WAYPOINT("DA26");
  ASSERT(my_idx != dest_idx);

  bool is_forward =
      (topology.routing_direction == TopologyInfo::RoutingDirection::Forward);
  UnicastRoutingParams result;
  result.outgoing_direction =
      is_forward ? static_cast<uint32_t>(
                       topology.routing_directions[topology.axis].first)
                 : static_cast<uint32_t>(
                       topology.routing_directions[topology.axis].second);
  ASSERT(result.outgoing_direction != eth_chan_directions::COUNT);

  result.num_hops =
      is_forward ? static_cast<uint8_t>((dest_idx - my_idx + size) % size)
                 : static_cast<uint8_t>((my_idx - dest_idx + size) % size);

  return result;
}

FORCE_INLINE UnicastRoutingParams get_unicast_params_line(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_device_id) {
  // TODO: check that my_device_id and dst_device_id are in same line/ring
  int32_t my_idx =
      topology.get_logical_mesh_position(my_device_id)[topology.axis];
  int32_t dest_idx =
      topology.get_logical_mesh_position(dst_device_id)[topology.axis];
  WAYPOINT("DA27");
  ASSERT(my_idx != dest_idx);

  UnicastRoutingParams result;
  if (my_idx < dest_idx) {
    result.outgoing_direction =
        static_cast<uint32_t>(topology.routing_directions[topology.axis].first);
    result.num_hops = dest_idx - my_idx;
  } else {
    result.outgoing_direction = static_cast<uint32_t>(
        topology.routing_directions[topology.axis].second);
    result.num_hops = my_idx - dest_idx;
  }
  ASSERT(result.outgoing_direction != eth_chan_directions::COUNT);

  return result;
}

FORCE_INLINE McastRoutingParams get_mcast_params(TopologyInfo &topology,
                                                 uint16_t my_device_id,
                                                 uint16_t dst_start_device_id,
                                                 uint16_t dst_end_device_id) {
  if (topology.topology_type == TopologyInfo::TopologyType::Ring) {
    // Assert: Only UnidirRingTorus routing mode supported for ring topology.
    WAYPOINT("DA28");
    ASSERT(topology.routing_mode == TopologyInfo::RoutingMode::UnidirRingTorus);
    return get_mcast_params_unidir_ring(topology, my_device_id,
                                        dst_start_device_id, dst_end_device_id);
  } else if (topology.topology_type == TopologyInfo::TopologyType::Line) {
    // Assert: Only BidirLineMesh routing mode supported for line topology.
    WAYPOINT("DA29");
    ASSERT(topology.routing_mode == TopologyInfo::RoutingMode::BidirLineMesh);
    return get_mcast_params_line(topology, my_device_id, dst_start_device_id,
                                 dst_end_device_id);
  } else {
    // Assert: Unsupported topology type.
    WAYPOINT("DA30");
    ASSERT(false);
    return McastRoutingParams();
  }
}

FORCE_INLINE McastRoutingParams get_mcast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_start_device_id,
    uint16_t dst_end_device_id) {
  int32_t my_idx =
      topology.get_logical_mesh_position(my_device_id)[topology.axis];
  int32_t start_idx =
      topology.get_logical_mesh_position(dst_start_device_id)[topology.axis];
  int32_t end_idx =
      topology.get_logical_mesh_position(dst_end_device_id)[topology.axis];
  int32_t size = topology.mesh_shape[topology.axis];

  //  Assert: at least one destination (not including sender)
  WAYPOINT("DA31");
  ASSERT(!(my_idx == start_idx && start_idx == end_idx));

  McastRoutingParams result;
  bool is_forward =
      (topology.routing_direction == TopologyInfo::RoutingDirection::Forward);
  uint32_t dir = is_forward
                     ? static_cast<uint32_t>(
                           topology.routing_directions[topology.axis].first)
                     : static_cast<uint32_t>(
                           topology.routing_directions[topology.axis].second);
  ASSERT(dir != eth_chan_directions::COUNT);

  auto [start_1, range_1, start_2_gap, range_2] =
      get_ring_regions(my_idx, start_idx, end_idx, size, is_forward);
  // ASSERT: gap not supported
  WAYPOINT("DA32");
  ASSERT(start_2_gap == 1);

  result.params_per_direction[dir].active = true;
  result.params_per_direction[dir].mcast_command_header = {
      static_cast<uint8_t>(start_1), static_cast<uint8_t>(range_1 + range_2)};

  return result;
}

// Note: in line don't use wrap with sender inside; in ring, if sender inside,
// only broadcast to full ring to assure this
FORCE_INLINE McastRoutingParams get_mcast_params_line(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_start_device_id,
    uint16_t dst_end_device_id) {
  // TODO: check that my_device_id and dst_device_id are in same line/ring
  int32_t my_idx =
      topology.get_logical_mesh_position(my_device_id)[topology.axis];
  int32_t start_idx =
      topology.get_logical_mesh_position(dst_start_device_id)[topology.axis];
  int32_t end_idx =
      topology.get_logical_mesh_position(dst_end_device_id)[topology.axis];
  int32_t size = topology.mesh_shape[topology.axis];

  // Assert: at least one destination (not including sender)
  WAYPOINT("DA33");
  ASSERT(!(my_idx == start_idx && start_idx == end_idx));

  auto [start_fwd, range_fwd, start_bwd, range_bwd] =
      get_line_regions(my_idx, start_idx, end_idx, size);

  McastRoutingParams result;
  uint32_t fwd_dir =
      static_cast<uint32_t>(topology.routing_directions[topology.axis].first);
  uint32_t bwd_dir =
      static_cast<uint32_t>(topology.routing_directions[topology.axis].second);
  ASSERT(fwd_dir != eth_chan_directions::COUNT);
  ASSERT(bwd_dir != eth_chan_directions::COUNT);

  if (range_fwd != 0) {
    result.params_per_direction[fwd_dir].active = true;
    result.params_per_direction[fwd_dir].mcast_command_header = {
        static_cast<uint8_t>(start_fwd), static_cast<uint8_t>(range_fwd)};
  }
  if (range_bwd != 0) {
    result.params_per_direction[bwd_dir].active = true;
    result.params_per_direction[bwd_dir].mcast_command_header = {
        static_cast<uint8_t>(start_bwd), static_cast<uint8_t>(range_bwd)};
  }

  return result;
}

#endif // !FABRIC_2D

#endif // EXPERIMENTAL_FABRIC_1D_ROUTING_H
