// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef EXPERIMENTAL_FABRIC_1D_ROUTING_H
#define EXPERIMENTAL_FABRIC_1D_ROUTING_H

#ifndef FABRIC_2D             // 1D Fabric
#include "api/debug/dprint.h" // TODO: remove this

struct UnicastParams {
  uint32_t outgoing_direction;
  uint32_t num_hops;
};

struct McastParams {
  struct McastParamsPerDirection {
    bool active = false;
    tt::tt_fabric::MulticastRoutingCommandHeader mcast_command_header;
  };

  std::array<McastParamsPerDirection, MAX_SEND_DIR> params_per_direction;
};

// Forward declarations
FORCE_INLINE UnicastParams get_unicast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_device_id);
FORCE_INLINE UnicastParams get_unicast_params_line(TopologyInfo &topology,
                                                   uint16_t my_device_id,
                                                   uint16_t dst_device_id);
FORCE_INLINE McastParams get_mcast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_start_device_id,
    uint16_t dst_end_device_id);
FORCE_INLINE McastParams get_mcast_params_line(TopologyInfo &topology,
                                               uint16_t my_device_id,
                                               uint16_t dst_start_device_id,
                                               uint16_t dst_end_device_id);

FORCE_INLINE UnicastParams get_unicast_params(TopologyInfo &topology,
                                              uint16_t my_device_id,
                                              uint16_t dst_device_id) {
  if (topology.topology_type == TopologyInfo::TopologyType::Ring) {
    // Assert: Shortest path not supported for ring topology.
    WAYPOINT("DA19");
    ASSERT(topology.routing_mode ==
           TopologyInfo::RoutingMode::UnidirectionalRingTorus);
    return get_unicast_params_unidir_ring(topology, my_device_id,
                                          dst_device_id);
  } else if (topology.topology_type == TopologyInfo::TopologyType::Line) {
    // Assert: Unidirectional routing mode not supported for line topology.
    WAYPOINT("DA20");
    ASSERT(topology.routing_mode !=
           TopologyInfo::RoutingMode::UnidirectionalRingTorus);
    return get_unicast_params_line(topology, my_device_id, dst_device_id);
  } else {
    // Assert: Unsupported topology type.
    WAYPOINT("DA21");
    ASSERT(false);
    return UnicastParams();
  }
}

FORCE_INLINE UnicastParams get_unicast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_device_id) {
  int32_t my_idx =
      topology.get_logical_mesh_position(my_device_id)[topology.axis];
  int32_t dest_idx =
      topology.get_logical_mesh_position(dst_device_id)[topology.axis];
  int32_t size = topology.mesh_shape[topology.axis];
  WAYPOINT("DA22");
  ASSERT(my_idx != dest_idx);

  bool is_forward =
      (topology.routing_direction == TopologyInfo::RoutingDirection::SouthEast);
  UnicastParams result;
  result.outgoing_direction =
      is_forward ? static_cast<uint32_t>(
                       topology.routing_directions[topology.axis].first)
                 : static_cast<uint32_t>(
                       topology.routing_directions[topology.axis].second);
  result.num_hops =
      is_forward ? static_cast<uint8_t>((dest_idx - my_idx + size) % size)
                 : static_cast<uint8_t>((my_idx - dest_idx + size) % size);

  return result;
}

FORCE_INLINE UnicastParams get_unicast_params_line(TopologyInfo &topology,
                                                   uint16_t my_device_id,
                                                   uint16_t dst_device_id) {
  // TODO: check that my_device_id and dst_device_id are in same line/ring
  int32_t my_idx =
      topology.get_logical_mesh_position(my_device_id)[topology.axis];
  int32_t dest_idx =
      topology.get_logical_mesh_position(dst_device_id)[topology.axis];
  WAYPOINT("DA23");
  ASSERT(my_idx != dest_idx);

  UnicastParams result;
  if (my_idx < dest_idx) {
    result.outgoing_direction =
        static_cast<uint32_t>(topology.routing_directions[topology.axis].first);
    result.num_hops = dest_idx - my_idx;
  } else {
    result.outgoing_direction = static_cast<uint32_t>(
        topology.routing_directions[topology.axis].second);
    result.num_hops = my_idx - dest_idx;
  }

  return result;
}

FORCE_INLINE McastParams get_mcast_params(TopologyInfo &topology,
                                          uint16_t my_device_id,
                                          uint16_t dst_start_device_id,
                                          uint16_t dst_end_device_id) {
  if (topology.topology_type == TopologyInfo::TopologyType::Ring) {
    // Assert: Shortest path not supported for ring topology.
    WAYPOINT("DA24");
    ASSERT(topology.routing_mode ==
           TopologyInfo::RoutingMode::UnidirectionalRingTorus);
    return get_mcast_params_unidir_ring(topology, my_device_id,
                                        dst_start_device_id, dst_end_device_id);
    // return get_mcast_params_shortest_path(topology, my_device_id,
    // dst_start_device_id, dst_end_device_id);
  } else if (topology.topology_type == TopologyInfo::TopologyType::Line) {
    // Assert: Unidirectional routing mode not supported for line topology.
    // WAYPOINT("DA25"); ASSERT(topology.routing_mode !=
    // TopologyInfo::RoutingMode::UnidirectionalRingTorus);
    return get_mcast_params_line(topology, my_device_id, dst_start_device_id,
                                 dst_end_device_id);
  } else {
    // Assert: Unsupported topology type.
    WAYPOINT("DA26");
    ASSERT(false);
    return McastParams();
  }
}

FORCE_INLINE McastParams get_mcast_params_unidir_ring(
    TopologyInfo &topology, uint16_t my_device_id, uint16_t dst_start_device_id,
    uint16_t dst_end_device_id) {
  int32_t my_idx =
      topology.get_logical_mesh_position(my_device_id)[topology.axis];
  int32_t start_idx =
      topology.get_logical_mesh_position(dst_start_device_id)[topology.axis];
  int32_t end_idx =
      topology.get_logical_mesh_position(dst_end_device_id)[topology.axis];
  int32_t size = topology.mesh_shape[topology.axis];

  // DPRINT << "my_idx: " << my_idx << ", start_idx: " << start_idx << ",
  // end_idx: " << end_idx << "\n";
  //  Assert: at least one destination (not including sender)
  WAYPOINT("DA27");
  ASSERT(!(my_idx == start_idx && start_idx == end_idx));

  McastParams result;
  bool is_forward =
      (topology.routing_direction == TopologyInfo::RoutingDirection::SouthEast);
  uint32_t dir = is_forward
                     ? static_cast<uint32_t>(
                           topology.routing_directions[topology.axis].first)
                     : static_cast<uint32_t>(
                           topology.routing_directions[topology.axis].second);

  auto [start_1, range_1, start_2_gap, range_2] =
      get_ring_regions(my_idx, start_idx, end_idx, size, is_forward);
  // ASSERT: gap not supported
  WAYPOINT("DA28");
  ASSERT(start_2_gap == 1);
  // DPRINT << "start_hop: " << (uint32_t)start_hop << ", range: " <<
  // (uint32_t)range << "\n";

  result.params_per_direction[dir].active = true;
  result.params_per_direction[dir].mcast_command_header = {
      static_cast<uint8_t>(start_1), static_cast<uint8_t>(range_1 + range_2)};

  return result;
}

// Note: in line don't use wrap with sender inside; in ring, if sender inside,
// only broadcast to full ring to assure this
FORCE_INLINE McastParams get_mcast_params_line(TopologyInfo &topology,
                                               uint16_t my_device_id,
                                               uint16_t dst_start_device_id,
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
  WAYPOINT("DA29");
  ASSERT(!(my_idx == start_idx && start_idx == end_idx));

  // if (topology.topology_type == TopologyInfo::TopologyType::Ring) {
  //   // to actually implement this, we will translate the ring to start from
  //   the farthest position from my_idx
  //   // that is closer in the backward direction (than the forward),
  //   // for even sized ring, we break the tie by preferring the forward
  //   direction as closer int farthest_backward_idx = (my_idx - (size - 1) / 2)
  //   % size; my_idx = my_idx - farthest_backward_idx % size; start_idx =
  //   start_idx - farthest_backward_idx % size; end_idx = end_idx -
  //   farthest_backward_idx % size;
  // }

  auto [start_fwd, range_fwd, start_bwd, range_bwd] =
      get_line_regions(my_idx, start_idx, end_idx, size);

  McastParams result;
  uint32_t fwd_dir =
      static_cast<uint32_t>(topology.routing_directions[topology.axis].first);
  uint32_t bwd_dir =
      static_cast<uint32_t>(topology.routing_directions[topology.axis].second);

  if (start_fwd != -1) {
    result.params_per_direction[fwd_dir].active = true;
    result.params_per_direction[fwd_dir].mcast_command_header = {
        static_cast<uint8_t>(start_fwd), static_cast<uint8_t>(range_fwd)};
  }
  if (start_bwd != -1) {
    result.params_per_direction[bwd_dir].active = true;
    result.params_per_direction[bwd_dir].mcast_command_header = {
        static_cast<uint8_t>(start_bwd), static_cast<uint8_t>(range_bwd)};
  }

  return result;
}

#endif // !FABRIC_2D

#endif // EXPERIMENTAL_FABRIC_1D_ROUTING_H
