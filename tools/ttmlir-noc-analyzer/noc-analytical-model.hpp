#include <string>
#include <map>

#define NOC_MAX_BURST_SIZE 8192

namespace analyzer::model {

    const std::map<std::string, int> noc_params = {
        {"pre_issue_overhead", 17}, 
        {"niu_programming", 6}, 
        {"non_niu_programming", 43}, 
        {"round_trip_latency", 96}, 
        {"head_flit_latency", 1},
        {"flit_latency", 1},
        {"noc_data_width", 32}};

    std::tuple<float, float, float> get_read_latency(std::map<std::string, int> noc_params, int buffer_size);
    std::tuple<float, float, float> get_write_latency(std::map<std::string, int> noc_params, int buffer_size);
}