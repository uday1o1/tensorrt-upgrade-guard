#include "kernels/residual_rmsnorm_launch.hpp"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

bool cudaOk(cudaError_t result, char const* operation)
{
    if (result == cudaSuccess)
    {
        return true;
    }
    std::cerr << operation << " failed: " << cudaGetErrorString(result) << '\n';
    return false;
}

struct Buffers
{
    float* x{nullptr};
    float* residual{nullptr};
    float* gamma{nullptr};
    float* output{nullptr};

    ~Buffers()
    {
        cudaFree(output);
        cudaFree(gamma);
        cudaFree(residual);
        cudaFree(x);
    }
};

bool allocate(Buffers& buffers, std::int32_t rows, std::int32_t hidden)
{
    std::size_t const bytes
        = static_cast<std::size_t>(rows) * static_cast<std::size_t>(hidden) * sizeof(float);
    std::vector<float> gamma(static_cast<std::size_t>(hidden), 1.0F);
    return cudaOk(cudaMalloc(&buffers.x, bytes), "cudaMalloc(x)")
        && cudaOk(cudaMalloc(&buffers.residual, bytes), "cudaMalloc(residual)")
        && cudaOk(cudaMalloc(&buffers.gamma, gamma.size() * sizeof(float)), "cudaMalloc(gamma)")
        && cudaOk(cudaMalloc(&buffers.output, bytes), "cudaMalloc(output)")
        && cudaOk(cudaMemset(buffers.x, 1, bytes), "cudaMemset(x)")
        && cudaOk(cudaMemset(buffers.residual, 2, bytes), "cudaMemset(residual)")
        && cudaOk(cudaMemcpy(buffers.gamma, gamma.data(), gamma.size() * sizeof(float),
                      cudaMemcpyHostToDevice),
            "cudaMemcpy(gamma)");
}

cudaError_t launch(bool optimized, Buffers const& buffers, std::int32_t rows, std::int32_t hidden)
{
    if (optimized)
    {
        return upgrade_guard::launchResidualRmsNormOptimized(nvinfer1::DataType::kFLOAT,
            buffers.x, buffers.residual, buffers.gamma, buffers.output, rows, hidden, 1e-5F,
            nullptr);
    }
    return upgrade_guard::launchResidualRmsNormScalar(nvinfer1::DataType::kFLOAT, buffers.x,
        buffers.residual, buffers.gamma, buffers.output, rows, hidden, 1e-5F, nullptr);
}

float benchmark(bool optimized, Buffers const& buffers, std::int32_t rows, std::int32_t hidden)
{
    constexpr std::int32_t warmups{20};
    constexpr std::int32_t repetitions{300};
    for (std::int32_t index = 0; index < warmups; ++index)
    {
        if (launch(optimized, buffers, rows, hidden) != cudaSuccess)
        {
            return NAN;
        }
    }
    cudaEvent_t start{};
    cudaEvent_t stop{};
    if (!cudaOk(cudaEventCreate(&start), "cudaEventCreate(start)")
        || !cudaOk(cudaEventCreate(&stop), "cudaEventCreate(stop)")
        || !cudaOk(cudaEventRecord(start), "cudaEventRecord(start)"))
    {
        return NAN;
    }
    for (std::int32_t index = 0; index < repetitions; ++index)
    {
        if (launch(optimized, buffers, rows, hidden) != cudaSuccess)
        {
            return NAN;
        }
    }
    float milliseconds{NAN};
    bool const ok = cudaOk(cudaEventRecord(stop), "cudaEventRecord(stop)")
        && cudaOk(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)")
        && cudaOk(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ok ? milliseconds / static_cast<float>(repetitions) : NAN;
}

bool profileOnly()
{
    constexpr std::int32_t rows{4096};
    constexpr std::int32_t hidden{256};
    Buffers buffers;
    if (!allocate(buffers, rows, hidden))
    {
        return false;
    }
    nvtxRangePushA("upgrade_guard/residual_rmsnorm_optimized");
    for (std::int32_t index = 0; index < 20; ++index)
    {
        if (launch(true, buffers, rows, hidden) != cudaSuccess)
        {
            nvtxRangePop();
            return false;
        }
    }
    bool const ok = cudaOk(cudaDeviceSynchronize(), "profile synchronization");
    nvtxRangePop();
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--profile-only") == 0)
    {
        return profileOnly() ? 0 : 1;
    }
    constexpr std::int32_t rows{4096};
    std::vector<std::int32_t> const hiddenSizes{256, 259};
    bool measuredBenefit{false};
    bool noRegression{true};
    std::cout << std::fixed << std::setprecision(9) << "{\"schema_version\":"
              << "\"upgradeguard.dev/cuda-benchmark/v1\",\"profiled\":false,\"cases\":[";
    for (std::size_t index = 0; index < hiddenSizes.size(); ++index)
    {
        std::int32_t const hidden = hiddenSizes[index];
        Buffers buffers;
        if (!allocate(buffers, rows, hidden))
        {
            return 1;
        }
        float const scalar = benchmark(false, buffers, rows, hidden);
        float const optimized = benchmark(true, buffers, rows, hidden);
        if (!std::isfinite(scalar) || !std::isfinite(optimized) || scalar <= 0.0F)
        {
            return 1;
        }
        float const ratio = optimized / scalar;
        measuredBenefit = measuredBenefit || ratio < 0.98F;
        noRegression = noRegression && ratio <= 1.05F;
        if (index != 0)
        {
            std::cout << ',';
        }
        std::cout << "{\"hidden\":" << hidden << ",\"rows\":" << rows
                  << ",\"scalar_ms\":" << scalar << ",\"optimized_ms\":" << optimized
                  << ",\"ratio\":" << ratio << '}';
    }
    bool const passed = measuredBenefit && noRegression;
    std::cout << "],\"status\":\"" << (passed ? "passed" : "failed")
              << "\",\"measured_benefit\":" << (measuredBenefit ? "true" : "false")
              << ",\"required_shapes_within_policy\":" << (noRegression ? "true" : "false")
              << "}\n";
    return passed ? 0 : 1;
}
