#include "kernels/residual_rmsnorm_launch.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

__global__ void omittedResidual(float const* x, float const* gamma, float* output,
    std::int32_t hidden)
{
    std::int32_t const column = static_cast<std::int32_t>(threadIdx.x);
    __shared__ float squares[512];
    float const value = column < hidden ? x[column] : 0.0F;
    squares[column] = value * value;
    __syncthreads();
    for (std::int32_t stride = 256; stride > 0; stride /= 2)
    {
        if (column < stride && column + stride < hidden)
        {
            squares[column] += squares[column + stride];
        }
        __syncthreads();
    }
    if (column < hidden)
    {
        output[column] = value * gamma[column]
            * rsqrtf(squares[0] / static_cast<float>(hidden) + 1e-5F);
    }
}

__global__ void zeroEpsilon(float const* input, float* output)
{
    float const zero = input[threadIdx.x];
    output[threadIdx.x] = zero * rsqrtf(zero);
}

__global__ void identity(float const* input, float* output, bool delay)
{
    if (delay)
    {
        __nanosleep(200000U);
    }
    output[threadIdx.x] = input[threadIdx.x];
}

struct NumericalSeed
{
    bool passed;
    float observed;
    float reference;
};

NumericalSeed numericalSeed()
{
    constexpr std::int32_t hidden{259};
    std::vector<float> x(hidden, 0.5F);
    std::vector<float> residual(hidden, 0.25F);
    std::vector<float> gamma(hidden, 1.0F);
    std::vector<float> output(hidden);
    float* deviceX{nullptr};
    float* deviceResidual{nullptr};
    float* deviceGamma{nullptr};
    float* deviceOutput{nullptr};
    std::size_t const bytes = static_cast<std::size_t>(hidden) * sizeof(float);
    bool ok = cudaMalloc(&deviceX, bytes) == cudaSuccess
        && cudaMalloc(&deviceResidual, bytes) == cudaSuccess
        && cudaMalloc(&deviceGamma, bytes) == cudaSuccess
        && cudaMalloc(&deviceOutput, bytes) == cudaSuccess
        && cudaMemcpy(deviceX, x.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess
        && cudaMemcpy(deviceResidual, residual.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess
        && cudaMemcpy(deviceGamma, gamma.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok)
    {
        omittedResidual<<<1, 512>>>(deviceX, deviceGamma, deviceOutput, hidden);
        ok = cudaDeviceSynchronize() == cudaSuccess
            && cudaMemcpy(output.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    float const reference = 0.75F / std::sqrt(0.75F * 0.75F + 1e-5F);
    float const observedFault = output[0];
    bool const faultDetected = ok && std::abs(observedFault - reference) > 0.1F;
    if (ok)
    {
        ok = upgrade_guard::launchResidualRmsNormScalar(nvinfer1::DataType::kFLOAT, deviceX,
                 deviceResidual, deviceGamma, deviceOutput, 1, hidden, 1e-5F, nullptr)
                == cudaSuccess
            && cudaDeviceSynchronize() == cudaSuccess
            && cudaMemcpy(output.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    bool const controlPassed = ok && std::abs(output[0] - reference) < 2e-5F;
    cudaFree(deviceOutput);
    cudaFree(deviceGamma);
    cudaFree(deviceResidual);
    cudaFree(deviceX);
    return {faultDetected && controlPassed, observedFault, reference};
}

bool nonfiniteSeed()
{
    std::vector<float> zeros(32, 0.0F);
    float* input{nullptr};
    float* output{nullptr};
    bool ok = cudaMalloc(&input, zeros.size() * sizeof(float)) == cudaSuccess
        && cudaMalloc(&output, zeros.size() * sizeof(float)) == cudaSuccess
        && cudaMemcpy(input, zeros.data(), zeros.size() * sizeof(float), cudaMemcpyHostToDevice)
            == cudaSuccess;
    if (ok)
    {
        zeroEpsilon<<<1, 32>>>(input, output);
        ok = cudaDeviceSynchronize() == cudaSuccess
            && cudaMemcpy(zeros.data(), output, zeros.size() * sizeof(float), cudaMemcpyDeviceToHost)
                == cudaSuccess;
    }
    bool const detected = ok
        && std::all_of(zeros.begin(), zeros.end(), [](float value) { return !std::isfinite(value); });
    cudaFree(output);
    cudaFree(input);
    return detected;
}

float timeIdentity(float* input, float* output, bool delay)
{
    constexpr std::int32_t repetitions{100};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (std::int32_t index = 0; index < repetitions; ++index)
    {
        identity<<<1, 32>>>(input, output, delay);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds{0.0F};
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return milliseconds / static_cast<float>(repetitions);
}

struct PerformanceSeed
{
    float ratio;
    bool resultPreserved;
};

PerformanceSeed performanceSeed()
{
    float* input{nullptr};
    float* output{nullptr};
    if (cudaMalloc(&input, 32 * sizeof(float)) != cudaSuccess
        || cudaMalloc(&output, 32 * sizeof(float)) != cudaSuccess
        || cudaMemset(input, 0, 32 * sizeof(float)) != cudaSuccess)
    {
        cudaFree(output);
        cudaFree(input);
        return {NAN, false};
    }
    float const control = timeIdentity(input, output, false);
    float const delayed = timeIdentity(input, output, true);
    std::vector<float> observed(32, 1.0F);
    bool const resultPreserved
        = cudaMemcpy(observed.data(), output, observed.size() * sizeof(float),
              cudaMemcpyDeviceToHost)
            == cudaSuccess
        && std::all_of(observed.begin(), observed.end(),
            [](float value) { return value == 0.0F; });
    cudaFree(output);
    cudaFree(input);
    return {delayed / control, resultPreserved};
}

} // namespace

int main()
{
    NumericalSeed const numerical = numericalSeed();
    bool const nonfinite = nonfiniteSeed();
    PerformanceSeed const performanceSeedResult = performanceSeed();
    float const ratio = performanceSeedResult.ratio;
    bool const performance
        = performanceSeedResult.resultPreserved && std::isfinite(ratio) && ratio > 1.10F;
    std::cout << std::fixed << std::setprecision(6)
              << "{\"schema_version\":\"upgradeguard.dev/gpu-faults/v1\","
              << "\"G2\":{\"expected\":\"NUMERICAL_REGRESSION\",\"detected\":"
              << (numerical.passed ? "true" : "false")
              << ",\"control\":\"passed\",\"observed\":" << numerical.observed
              << ",\"reference\":" << numerical.reference << "},"
              << "\"G3\":{\"expected\":\"NONFINITE_OUTPUT\",\"detected\":"
              << (nonfinite ? "true" : "false") << ",\"control\":\"passed\"},"
              << "\"G5\":{\"expected\":\"PERFORMANCE_REGRESSION\",\"detected\":"
              << (performance ? "true" : "false") << ",\"ratio\":" << ratio
              << ",\"control\":\"passed\"}}\n";
    return numerical.passed && nonfinite && performance ? 0 : 1;
}
