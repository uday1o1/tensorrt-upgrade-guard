#include "kernels/residual_rmsnorm_launch.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace upgrade_guard
{
namespace
{

__device__ float warpSum(float value)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

template <typename T>
__device__ float scalarLoad(T const* value, std::int64_t index);

template <>
__device__ float scalarLoad<float>(float const* value, std::int64_t index)
{
    return value[index];
}

template <>
__device__ float scalarLoad<half>(half const* value, std::int64_t index)
{
    return __half2float(value[index]);
}

template <typename T>
__device__ T scalarStore(float value);

template <>
__device__ float scalarStore<float>(float value)
{
    return value;
}

template <>
__device__ half scalarStore<half>(float value)
{
    return __float2half_rn(value);
}

template <typename T>
__global__ void residualRmsNormWarp(T const* x, T const* residual, float const* gamma, T* output,
    std::int64_t rows, std::int32_t hidden, float epsilon)
{
    std::int64_t const row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows)
    {
        return;
    }
    std::int64_t const rowOffset = row * hidden;
    float sum{0.0F};
    for (std::int32_t column = static_cast<std::int32_t>(threadIdx.x); column < hidden;
         column += static_cast<std::int32_t>(blockDim.x))
    {
        float const value = scalarLoad(x, rowOffset + column) + scalarLoad(residual, rowOffset + column);
        sum += value * value;
    }
    sum = warpSum(sum);
    __shared__ float warpSums[8];
    std::int32_t const lane = static_cast<std::int32_t>(threadIdx.x) & 31;
    std::int32_t const warp = static_cast<std::int32_t>(threadIdx.x) >> 5;
    if (lane == 0)
    {
        warpSums[warp] = sum;
    }
    __syncthreads();
    float blockSum = static_cast<std::int32_t>(threadIdx.x) < 8 ? warpSums[lane] : 0.0F;
    if (warp == 0)
    {
        blockSum = warpSum(blockSum);
    }
    if (threadIdx.x == 0)
    {
        warpSums[0] = rsqrtf(blockSum / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();
    float const inverseRms = warpSums[0];
    for (std::int32_t column = static_cast<std::int32_t>(threadIdx.x); column < hidden;
         column += static_cast<std::int32_t>(blockDim.x))
    {
        float const value = scalarLoad(x, rowOffset + column) + scalarLoad(residual, rowOffset + column);
        output[rowOffset + column] = scalarStore<T>(value * gamma[column] * inverseRms);
    }
}

__global__ void residualRmsNormFloat4(float const* x, float const* residual, float const* gamma,
    float* output, std::int64_t rows, std::int32_t hidden, float epsilon)
{
    std::int64_t const row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows)
    {
        return;
    }
    std::int64_t const rowOffset = row * hidden;
    auto const* x4 = reinterpret_cast<float4 const*>(x + rowOffset);
    auto const* residual4 = reinterpret_cast<float4 const*>(residual + rowOffset);
    auto* output4 = reinterpret_cast<float4*>(output + rowOffset);
    std::int32_t const vectors = hidden / 4;
    float sum{0.0F};
    for (std::int32_t index = static_cast<std::int32_t>(threadIdx.x); index < vectors;
         index += static_cast<std::int32_t>(blockDim.x))
    {
        float4 const a = x4[index];
        float4 const b = residual4[index];
        float4 const z{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
        sum += z.x * z.x + z.y * z.y + z.z * z.z + z.w * z.w;
    }
    sum = warpSum(sum);
    __shared__ float warpSums[8];
    int const lane = static_cast<int>(threadIdx.x) & 31;
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0)
    {
        warpSums[warp] = sum;
    }
    __syncthreads();
    float blockSum = static_cast<int>(threadIdx.x) < 8 ? warpSums[lane] : 0.0F;
    if (warp == 0)
    {
        blockSum = warpSum(blockSum);
    }
    if (threadIdx.x == 0)
    {
        warpSums[0] = rsqrtf(blockSum / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();
    float const inverseRms = warpSums[0];
    for (std::int32_t index = static_cast<std::int32_t>(threadIdx.x); index < vectors;
         index += static_cast<std::int32_t>(blockDim.x))
    {
        float4 const a = x4[index];
        float4 const b = residual4[index];
        std::int32_t const column = index * 4;
        output4[index] = float4{(a.x + b.x) * gamma[column] * inverseRms,
            (a.y + b.y) * gamma[column + 1] * inverseRms,
            (a.z + b.z) * gamma[column + 2] * inverseRms,
            (a.w + b.w) * gamma[column + 3] * inverseRms};
    }
}

__global__ void residualRmsNormHalf2(half const* x, half const* residual, float const* gamma,
    half* output, std::int64_t rows, std::int32_t hidden, float epsilon)
{
    std::int64_t const row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows)
    {
        return;
    }
    std::int64_t const rowOffset = row * hidden;
    auto const* x2 = reinterpret_cast<half2 const*>(x + rowOffset);
    auto const* residual2 = reinterpret_cast<half2 const*>(residual + rowOffset);
    auto* output2 = reinterpret_cast<half2*>(output + rowOffset);
    std::int32_t const vectors = hidden / 2;
    float sum{0.0F};
    for (std::int32_t index = static_cast<std::int32_t>(threadIdx.x); index < vectors;
         index += static_cast<std::int32_t>(blockDim.x))
    {
        float2 const a = __half22float2(x2[index]);
        float2 const b = __half22float2(residual2[index]);
        float const first = a.x + b.x;
        float const second = a.y + b.y;
        sum += first * first + second * second;
    }
    sum = warpSum(sum);
    __shared__ float warpSums[8];
    int const lane = static_cast<int>(threadIdx.x) & 31;
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0)
    {
        warpSums[warp] = sum;
    }
    __syncthreads();
    float blockSum = static_cast<int>(threadIdx.x) < 8 ? warpSums[lane] : 0.0F;
    if (warp == 0)
    {
        blockSum = warpSum(blockSum);
    }
    if (threadIdx.x == 0)
    {
        warpSums[0] = rsqrtf(blockSum / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();
    float const inverseRms = warpSums[0];
    for (std::int32_t index = static_cast<std::int32_t>(threadIdx.x); index < vectors;
         index += static_cast<std::int32_t>(blockDim.x))
    {
        float2 const a = __half22float2(x2[index]);
        float2 const b = __half22float2(residual2[index]);
        std::int32_t const column = index * 2;
        output2[index] = __floats2half2_rn(
            (a.x + b.x) * gamma[column] * inverseRms,
            (a.y + b.y) * gamma[column + 1] * inverseRms);
    }
}

template <typename T>
cudaError_t launchWarp(T const* x, T const* residual, float const* gamma, T* output, std::int64_t rows,
    std::int32_t hidden, float epsilon, cudaStream_t stream)
{
    constexpr std::int32_t threads{256};
    residualRmsNormWarp<<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
        x, residual, gamma, output, rows, hidden, epsilon);
    return cudaPeekAtLastError();
}

bool aligned(void const* pointer, std::uintptr_t alignment)
{
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

} // namespace

cudaError_t launchResidualRmsNormOptimized(nvinfer1::DataType type, void const* x, void const* residual,
    float const* gamma, void* output, std::int64_t rows, std::int32_t hidden, float epsilon,
    cudaStream_t stream) noexcept
{
    constexpr std::int32_t threads{256};
    if (type == nvinfer1::DataType::kFLOAT && hidden % 4 == 0 && aligned(x, alignof(float4))
        && aligned(residual, alignof(float4)) && aligned(output, alignof(float4)))
    {
        residualRmsNormFloat4<<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
            static_cast<float const*>(x), static_cast<float const*>(residual), gamma,
            static_cast<float*>(output), rows, hidden, epsilon);
        return cudaPeekAtLastError();
    }
    if (type == nvinfer1::DataType::kFLOAT)
    {
        return launchWarp(static_cast<float const*>(x), static_cast<float const*>(residual), gamma,
            static_cast<float*>(output), rows, hidden, epsilon, stream);
    }
    if (type == nvinfer1::DataType::kHALF && hidden % 2 == 0 && aligned(x, alignof(half2))
        && aligned(residual, alignof(half2)) && aligned(output, alignof(half2)))
    {
        residualRmsNormHalf2<<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
            static_cast<half const*>(x), static_cast<half const*>(residual), gamma,
            static_cast<half*>(output), rows, hidden, epsilon);
        return cudaPeekAtLastError();
    }
    if (type == nvinfer1::DataType::kHALF)
    {
        return launchWarp(static_cast<half const*>(x), static_cast<half const*>(residual), gamma,
            static_cast<half*>(output), rows, hidden, epsilon, stream);
    }
    return cudaErrorInvalidValue;
}

} // namespace upgrade_guard
