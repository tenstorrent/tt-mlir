// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

namespace tt::runtime::profiler {

class ProfilerManager {
public: 
    static ProfilerManager& instance();

    std::string getOutputDirectory();
    std::string getAddress();
    int getPort();
    std::string getCaptureCommand();
    std::string getCSVExportTimesCommand();
    std::string getCSVExportDataCommand();
    std::vector<std::string> getProfilerEnvVars();

    std::string setOutputDirectory(const std::string& outputDirectory);
    std::string setAddress(const std::string& address);
    int setPort(int port);

    std::string tracyOutputFileName = "output.tracy";
    std::string tracyOpsTimeFileName = "tracy_ops_times.csv";
    std::string tracyOpsDataFileName = "tracy_ops_data.csv";

private:
    ProfilerManager() = default;

    ProfilerManager(const ProfilerManager&) = delete;
    ProfilerManager& operator=(const ProfilerManager&) = delete;

    std::string m_outputDirectory;
    std::string m_address;
    int m_port;
};

class ProcessManager {
public:
    static ProcessManager& instance();

    void setEnvFlags(const std::vector<std::string>& envVars);
    void start(const std::string& command);
    void stop();
    void execute(const std::string& command, const std::string& output_file);
    pid_t pid() const;

private:
    ProcessManager() = default;

    ProcessManager(const ProcessManager&) = delete;
    ProcessManager& operator=(const ProcessManager&) = delete;

    pid_t m_pid = -1;
};

void start_profiler(std::string outputDirectory, std::string address, int port);
void stop_profiler();

} // namespace tt::runtime::profiler
