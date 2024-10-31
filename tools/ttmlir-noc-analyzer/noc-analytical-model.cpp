#include "noc-analytical-model.hpp"

namespace analyzer::model {
// Returns (total_issue_latency (no overhead), round_trip_latency, flit_latency)
std::tuple<float, float, float>
get_read_latency(std::map<std::string, int> noc_params, int buffer_size) {
  int transfer_size =
      buffer_size <= NOC_MAX_BURST_SIZE ? buffer_size : NOC_MAX_BURST_SIZE;
  //   float num_flits_per_transfer = transfer_size /
  //   noc_params["noc_data_width"];
  float num_transfer = buffer_size / transfer_size;
  int round_trip_latency = noc_params["round_trip_latency"];
  float issue_latency =
      noc_params["niu_programming"] + noc_params["non_niu_programming"];

  float total_issue_latency = issue_latency * num_transfer;
  float total_flit_latency =
      noc_params["head_flit_latency"] +
      noc_params["flit_latency"] * buffer_size / noc_params["noc_data_width"];

  //   float barrier_latency;
  //   if (num_flits_per_transfer >= issue_latency) {
  //     barrier_latency =
  //         noc_params["pre_issue_overhead"] + issue_latency +
  //         round_trip_latency + noc_params["head_flit_latency"] +
  //         noc_params["flit_latency"] * num_flits_per_transfer * num_transfer
  //         - total_issue_latency;
  //   } else {
  //     barrier_latency = round_trip_latency + noc_params["head_flit_latency"]
  //     +
  //                       noc_params["flit_latency"] * num_flits_per_transfer;
  //   }

  return std::tuple<float, float, float>{
      total_issue_latency, round_trip_latency, total_flit_latency};
}

std::tuple<float, float, float>
get_write_latency(std::map<std::string, int> noc_params, int buffer_size) {
  auto transfer_size =
      buffer_size <= NOC_MAX_BURST_SIZE ? buffer_size : NOC_MAX_BURST_SIZE;
  float num_flits_per_transfer = transfer_size / noc_params["noc_data_width"];
  float num_transfer = buffer_size / transfer_size;
  int round_trip_latency = noc_params["round_trip_latency"];

  float issue_latency =
      noc_params["niu_programming"] + noc_params["non_niu_programming"];

  float total_flit_latency =
      noc_params["flit_latency"] * buffer_size / noc_params["noc_data_width"];

  float total_issue_latency;
  float barrier_latency;

  if (num_flits_per_transfer >= issue_latency) {
    if (num_transfer < 3) {
      total_issue_latency = issue_latency * num_transfer;
    } else {
      total_issue_latency = issue_latency +
                            noc_params["flit_latency"] *
                                num_flits_per_transfer * (num_transfer - 2) +
                            15;
    }
    barrier_latency =
        issue_latency + round_trip_latency + noc_params["head_flit_latency"] +
        noc_params["flit_latency"] * num_flits_per_transfer * num_transfer -
        total_issue_latency;
  } else {
    total_issue_latency =
        noc_params["pre_issue_overhead"] + issue_latency * num_transfer;
    barrier_latency = noc_params["round_trip_latency"] +
                      noc_params["head_flit_latency"] +
                      noc_params["flit_latency"] * num_flits_per_transfer;
  }

  return std::tuple<float, float, float>{total_issue_latency, barrier_latency,
                                         total_flit_latency};
}
} // namespace analyzer::model
