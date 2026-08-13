#include "kernels/residual_rmsnorm_launch.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace upgrade_guard
{
namespace
{

template <typename T>
__device__ float load(T const* value, std::int64_t index)
{
    return static_cast<float>(value[index]);
}

template <>
__device__ float load<half>(half const* value, std::int64_t index)
{
    return __half2float(value[index]);
}

template <typename T>
__device__ T store(float value);

template <>
__device__ float store<float>(float value)
{
    return value;
}

template <>
__device__ half store<half>(float value)
{
    return __float2half_rn(value);
}

template <typename T>
__global__ void residualRmsNormScalar(T const* x, T const* residual, float const* gamma, T* output,
    std::int64_t rows, std::int32_t hidden, float epsilon)
{
    std::int64_t const row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows)
    {
        return;
    }
    std::int64_t const offset = row * hidden;
    float sum{0.0F};
    for (std::int32_t column = static_cast<std::int32_t>(threadIdx.x); column < hidden;
         column += static_cast<std::int32_t>(blockDim.x))
    {
        float const value = load(x, offset + column) + load(residual, offset + column);
        sum += value * value;
    }
    extern __shared__ float shared[];
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (std::int32_t stride = static_cast<std::int32_t>(blockDim.x) / 2; stride > 0; stride /= 2)
    {
        if (static_cast<std::int32_t>(threadIdx.x) < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float const inverseRms = rsqrtf(shared[0] / static_cast<float>(hidden) + epsilon);
    for (std::int32_t column = static_cast<std::int32_t>(threadIdx.x); column < hidden;
         column += static_cast<std::int32_t>(blockDim.x))
    {
        float const value = load(x, offset + column) + load(residual, offset + column);
        output[offset + column] = store<T>(value * gamma[column] * inverseRms);
    }
}

template <typename T>
cudaError_t launch(T const* x, T const* residual, float const* gamma, T* output, std::int64_t rows,
    std::int32_t hidden, float epsilon, cudaStream_t stream) noexcept
{
    constexpr std::int32_t threads{256};
    residualRmsNormScalar<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
        x, residual, gamma, output, rows, hidden, epsilon);
    return cudaPeekAtLastError();
}

} // namespace

cudaError_t launchResidualRmsNormScalar(nvinfer1::DataType type, void const* x, void const* residual,
    float const* gamma, void* output, std::int64_t rows, std::int32_t hidden, float epsilon,
    cudaStream_t stream) noexcept
{
    if (type == nvinfer1::DataType::kFLOAT)
    {
        return launch(static_cast<float const*>(x), static_cast<float const*>(residual), gamma,
            static_cast<float*>(output), rows, hidden, epsilon, stream);
    }
    if (type == nvinfer1::DataType::kHALF)
    {
        return launch(static_cast<half const*>(x), static_cast<half const*>(residual), gamma,
            static_cast<half*>(output), rows, hidden, epsilon, stream);
    }
    return cudaErrorInvalidValue;
}

} // namespace upgrade_guard
