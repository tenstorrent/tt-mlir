// VoVNet-v2 (ese_vovnet19b_dw.ra_in1k) TT-NN Implementation
// Bounty: tenstorrent/tt-mlir#4349
// Target: ~1400 FPS on Wormhole N150

#include <iostream>
#include <chrono>
#include <vector>
#include "ttnn/operations/ccl/all_gather.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/conv2d/conv2d.hpp"
#include "ttnn/operations/normalization/layer_normalize.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/reduce/reduce_mode.hpp"

// VoVNet Configuration for ese_vovnet19b_dw variant
namespace vovnet_config {
    constexpr int IMAGE_HEIGHT = 224;
    constexpr int IMAGE_WIDTH = 224;
    constexpr int IN_CHANNELS = 3;
    constexpr int NUM_CLASSES = 1000;
    
    // Stage configurations for ese_vovnet19b_dw
    const std::vector<std::vector<int>> STAGE_OUT_CHANNELS = {
        {64, 64, 128],   // Stage 1
        [256, 256, 256, 256],  // Stage 2  
        [512, 512, 512, 512],  // Stage 3
        [1024, 1024, 1024]     // Stage 4
    };
    
    const std::vector<int> STAGE_STRIDES = {2, 2, 2, 2};
}

// Helper to create convolution with optimized parameters
ttnn::Tensor create_conv2d(const ttnn::Tensor& input,
                           int out_channels,
                           int kernel_size,
                           int stride = 1,
                           int padding = 0,
                           int groups = 1) {
    auto device = ttnn::get_device(0);
    
    ttnn::Conv2dConfig config;
    config.dtype = ttnn::bfloat16;
    config.weights_dtype = ttnn::bfloat16;
    config.activation = "";
    
    return ttnn::conv2d(input, out_channels, kernel_size, config, stride, padding, groups);
}

// Squeeze-and-Excitation block (ESE attention)
ttnn::Tensor ese_block(const ttnn::Tensor& input, int channels) {
    // Global average pooling
    auto gap = ttnn::reduce操作(input, ttnn::ReduceMode::Mean, {2, 3});
    
    // FC -> ReLU -> FC -> Sigmoid
    auto fc1 = create_conv2d(gap, channels / 4, 1);
    auto relu = ttnn::relu(fc1);
    auto fc2 = create_conv2d(relu, channels, 1);
    auto sigmoid = ttnn::sigmoid(fc2);
    
    // Scale
    return ttnn::multiply(input, sigmoid);
}

// VoVNet Residual Block with Depthwise Convolution
ttnn::Tensor vovnet_residual_block(const ttnn::Tensor& input,
                                   int in_channels,
                                   int out_channels,
                                   int stride = 1) {
    // Depthwise convolution
    auto dw_conv = create_conv2d(input, in_channels, 3, stride, 1, in_channels);
    
    // Pointwise convolution 1
    auto pw1 = create_conv2d(dw_conv, out_channels / 2, 1);
    
    // Activation
    auto act1 = ttnn::relu(pw1);
    
    // Pointwise convolution 2
    auto pw2 = create_conv2d(act1, out_channels, 1);
    
    // ESE attention
    auto ese = ese_block(pw2, out_channels);
    
    // Residual connection (if dimensions match)
    ttnn::Tensor residual = input;
    if (stride != 1 || in_channels != out_channels) {
        // Adjust residual path
        if (in_channels != out_channels) {
            auto proj_conv = create_conv2d(input, out_channels, 1, stride);
            residual = proj_conv;
        }
    }
    
    // Add with SE output
    auto add_out = ttnn::add(residual, ese);
    return ttnn::relu(add_out);
}

// VoVNet Stage (multiple residual blocks)
ttnn::Tensor vovnet_stage(const ttnn::Tensor& input,
                          const std::vector<int>& out_channels,
                          int stride,
                          int stage_idx) {
    // First block with stride and channel change
    auto x = vovnet_residual_block(input, input.get_shape()[1], out_channels[0], stride);
    
    // Remaining blocks
    for (size_t i = 1; i < out_channels.size(); ++i) {
        x = vovnet_residual_block(x, out_channels[i-1], out_channels[i], 1);
    }
    
    return x;
}

// VoVNet-v2 Stem (Initial convolution layers)
ttnn::Tensor vovnet_stem(const ttnn::Tensor& input) {
    // Stem: 3x3 conv -> bn -> relu -> maxpool
    auto conv1 = create_conv2d(input, 64, 3, 2, 1);
    auto bn1 = ttnn::batch_norm(conv1);
    auto relu1 = ttnn::relu(bn1);
    auto pool1 = ttnn::max_pool2d(relu1, 3, 2, 1);
    
    return pool1;
}

// Full VoVNet-v2 Forward Pass
ttnn::Tensor forward_vovnet(const ttnn::Tensor& input) {
    // Input shape: [batch, 3, 224, 224]
    
    // Stem
    auto x = vovnet_stem(input);
    
    // Stage 1: 64 -> 128 channels
    x = vovnet_stage(x, vovnet_config::STAGE_OUT_CHANNELS[0], 2, 0);
    
    // Stage 2: 128 -> 256 channels
    x = vovnet_stage(x, vovnet_config::STAGE_OUT_CHANNELS[1], 2, 1);
    
    // Stage 3: 256 -> 512 channels
    x = vovnet_stage(x, vovnet_config::STAGE_OUT_CHANNELS[2], 2, 2);
    
    // Stage 4: 512 -> 1024 channels
    x = vovnet_stage(x, vovnet_config::STAGE_OUT_CHANNELS[3], 2, 3);
    
    // Global Average Pooling
    x = ttnn::reduce(x, ttnn::ReduceMode::Mean, {2, 3});
    
    // Classifier
    auto classifier = create_conv2d(x, vovnet_config::NUM_CLASSES, 1);
    
    return classifier;
}

// Benchmark function
void benchmark_vovnet(int batch_size, int num_iterations) {
    auto device = ttnn::get_device(0);
    
    // Create input tensor with fake data
    std::vector<float> input_data(batch_size * vovnet_config::IN_CHANNELS * 
                                   vovnet_config::IMAGE_HEIGHT * vovnet_config::IMAGE_WIDTH, 0.1f);
    auto input_shape = ttnn::Shape{batch_size, vovnet_config::IN_CHANNELS,
                                   vovnet_config::IMAGE_HEIGHT, vovnet_config::IMAGE_WIDTH};
    
    auto input = ttnn::from_vector(input_data, input_shape, ttnn::bfloat16, device);
    
    std::cout << "Running VoVNet benchmark..." << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Iterations: " << num_iterations << std::endl;
    
    // Warmup (first iteration may be slower due to kernel compilation)
    {
        auto output = forward_vovnet(input);
        ttnn::synchronize();
    }
    
    // Timed iterations
    std::vector<double> times;
    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto output = forward_vovnet(input);
        ttnn::from_device(output);  // Block until tensor is read from device
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        times.push_back(duration.count());
        
        std::cout << "Batch " << (i + 1) << " time: " << duration.count() << "s" << std::endl;
    }
    
    // Calculate statistics
    double total_time = 0;
    for (double t : times) total_time += t;
    double avg_time = total_time / times.size();
    double throughput = batch_size / avg_time;
    
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << "Average time per batch: " << avg_time << "s" << std::endl;
    std::cout << "Throughput: " << throughput << " samples/sec" << std::endl;
    std::cout << "Target: ~1400 samples/sec" << std::endl;
}

int main(int argc, char** argv) {
    int batch_size = 8;
    int iterations = 10;
    
    if (argc > 1) batch_size = std::atoi(argv[1]);
    if (argc > 2) iterations = std::atoi(argv[2]);
    
    benchmark_vovnet(batch_size, iterations);
    return 0;
}
