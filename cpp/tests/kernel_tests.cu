#include "kernels/residual_rmsnorm_launch.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
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

template <typename T>
float asFloat(T value)
{
    return static_cast<float>(value);
}

template <>
float asFloat<half>(half value)
{
    return __half2float(value);
}

template <typename T>
T fromFloat(float value)
{
    return static_cast<T>(value);
}

template <>
half fromFloat<half>(float value)
{
    return __float2half_rn(value);
}

template <typename T>
bool runCase(std::int32_t rows, std::int32_t hidden, float scale)
{
    std::size_t const elements = static_cast<std::size_t>(rows) * static_cast<std::size_t>(hidden);
    std::mt19937 generator{20260813U + static_cast<std::uint32_t>(hidden)};
    std::uniform_real_distribution<float> distribution{-scale, scale};
    std::vector<T> x(elements);
    std::vector<T> residual(elements);
    std::vector<float> gamma(static_cast<std::size_t>(hidden));
    for (std::size_t index = 0; index < elements; ++index)
    {
        x[index] = fromFloat<T>(distribution(generator));
        residual[index] = fromFloat<T>(distribution(generator));
    }
    for (std::int32_t index = 0; index < hidden; ++index)
    {
        gamma[static_cast<std::size_t>(index)] = 0.5F + static_cast<float>(index % 17) / 17.0F;
    }
    T* deviceX{nullptr};
    T* deviceResidual{nullptr};
    float* deviceGamma{nullptr};
    T* deviceScalar{nullptr};
    T* deviceOptimized{nullptr};
    bool ok = cudaOk(cudaMalloc(&deviceX, elements * sizeof(T)), "cudaMalloc(x)")
        && cudaOk(cudaMalloc(&deviceResidual, elements * sizeof(T)), "cudaMalloc(residual)")
        && cudaOk(cudaMalloc(&deviceGamma, gamma.size() * sizeof(float)), "cudaMalloc(gamma)")
        && cudaOk(cudaMalloc(&deviceScalar, elements * sizeof(T)), "cudaMalloc(scalar)")
        && cudaOk(cudaMalloc(&deviceOptimized, elements * sizeof(T)), "cudaMalloc(optimized)");
    if (ok)
    {
        ok = cudaOk(cudaMemcpy(deviceX, x.data(), elements * sizeof(T), cudaMemcpyHostToDevice), "copy x")
            && cudaOk(cudaMemcpy(deviceResidual, residual.data(), elements * sizeof(T), cudaMemcpyHostToDevice),
                "copy residual")
            && cudaOk(cudaMemcpy(deviceGamma, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice),
                "copy gamma");
    }
    nvinfer1::DataType const type
        = std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
    if (ok)
    {
        ok = cudaOk(upgrade_guard::launchResidualRmsNormScalar(type, deviceX, deviceResidual,
                            deviceGamma, deviceScalar, rows, hidden, 1e-5F, nullptr),
                 "scalar launch")
            && cudaOk(upgrade_guard::launchResidualRmsNormOptimized(type, deviceX, deviceResidual,
                          deviceGamma, deviceOptimized, rows, hidden, 1e-5F, nullptr),
                "optimized launch")
            && cudaOk(cudaDeviceSynchronize(), "kernel synchronize");
    }
    std::vector<T> scalar(elements);
    std::vector<T> optimized(elements);
    if (ok)
    {
        ok = cudaOk(cudaMemcpy(scalar.data(), deviceScalar, elements * sizeof(T), cudaMemcpyDeviceToHost),
                 "copy scalar")
            && cudaOk(cudaMemcpy(optimized.data(), deviceOptimized, elements * sizeof(T), cudaMemcpyDeviceToHost),
                "copy optimized");
    }
    float const tolerance = std::is_same_v<T, float> ? 2e-5F : 3e-3F;
    for (std::int32_t row = 0; ok && row < rows; ++row)
    {
        float sum{0.0F};
        for (std::int32_t column = 0; column < hidden; ++column)
        {
            std::size_t const index = static_cast<std::size_t>(row * hidden + column);
            float const combined = asFloat(x[index]) + asFloat(residual[index]);
            sum += combined * combined;
        }
        float const inverseRms = 1.0F / std::sqrt(sum / static_cast<float>(hidden) + 1e-5F);
        for (std::int32_t column = 0; column < hidden; ++column)
        {
            std::size_t const index = static_cast<std::size_t>(row * hidden + column);
            float const expected
                = (asFloat(x[index]) + asFloat(residual[index])) * gamma[column] * inverseRms;
            if (std::abs(asFloat(scalar[index]) - expected) > tolerance
                || std::abs(asFloat(optimized[index]) - expected) > tolerance
                || !std::isfinite(asFloat(scalar[index])) || !std::isfinite(asFloat(optimized[index])))
            {
                std::cerr << "mismatch at row=" << row << " column=" << column << '\n';
                ok = false;
                break;
            }
        }
    }
    cudaFree(deviceOptimized);
    cudaFree(deviceScalar);
    cudaFree(deviceGamma);
    cudaFree(deviceResidual);
    cudaFree(deviceX);
    return ok;
}

} // namespace

int main()
{
    bool ok{true};
    for (std::int32_t hidden : {7, 256, 259})
    {
        ok = runCase<float>(3, hidden, 1.0F) && ok;
        ok = runCase<half>(3, hidden, 1.0F) && ok;
    }
    ok = runCase<float>(2, 256, 1000.0F) && ok;
    ok = runCase<half>(2, 256, 0.0F) && ok;
    return ok ? 0 : 1;
}
