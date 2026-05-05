#include "common.hpp"

int32_t main() {
  TRACY_ZONE("main");

  ttnn::distributed::MeshDevice* device = ttnn::DeviceGetter::getInstance();
  device->enable_program_cache();

  // Build inputs once; first element is the image, the rest are raw weights
  // owned by the model.
  std::vector<ttnn::Tensor> inputs = create_inputs_for_forward();
  ttnn::Tensor image = inputs[0];
  VoVNet model(device, std::move(inputs));

  // Pass 1: weight prep + kernel compilation.
  std::cout << "Pass 1: weight prep + kernel compilation..." << std::endl;
  {
    TRACY_ZONE("Pass 1 (compile + prep)");
    auto t0 = std::chrono::high_resolution_clock::now();
    ttnn::Tensor out = model(image);
    ttnn::Tensor host = ttnn::from_device(out);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  Init: "
              << std::chrono::duration<double>(t1 - t0).count() << "s"
              << std::endl;
  }

  // Pass 2: warmup (program cache hit). The metal trace below requires the
  // program cache to be fully populated.
  std::cout << "Pass 2: warmup..." << std::endl;
  {
    TRACY_ZONE("Pass 2 (warmup)");
    ttnn::Tensor out = model(image);
    ttnn::Tensor host = ttnn::from_device(out);
  }

  std::cout << "Program cache entries: "
            << device->num_program_cache_entries() << std::endl;

  // Capture metal trace: records the dispatch sequence of one forward() pass.
  // Subsequent replays reissue it without going through host op dispatch.
  std::cout << "Capturing metal trace..." << std::endl;
  {
    TRACY_ZONE("trace capture");
    auto t0 = std::chrono::high_resolution_clock::now();
    model.capture_trace();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  Trace capture: "
              << std::chrono::duration<double>(t1 - t0).count() << "s"
              << std::endl;
  }

  // Benchmark loop: replay-only path. Per-iter timer wraps replay + from_device
  // (the latter blocks until the device finishes, so the FPS is honest).
  constexpr int kNumIters = 10;
  constexpr int kBatchSize = 16;
  std::cout << "\n=== Benchmark (metal trace replay) ===" << std::endl;
  for (int i = 0; i < kNumIters; ++i) {
    TRACY_ZONE("benchmark iter");
    auto t0 = std::chrono::high_resolution_clock::now();
    ttnn::Tensor out = model.replay();
    ttnn::Tensor host = ttnn::from_device(out);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Iter " << i << ": " << secs << "s, "
              << (kBatchSize / secs) << " FPS" << std::endl;
  }

  return 0;
}
