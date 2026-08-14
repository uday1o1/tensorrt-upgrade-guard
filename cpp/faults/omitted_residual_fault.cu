#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace upgrade_guard
{
namespace
{

template <typename T>
__device__ float loadFault(T const* value, std::int64_t index)
{
    return static_cast<float>(value[index]);
}

template <>
__device__ float loadFault<half>(half const* value, std::int64_t index)
{
    return __half2float(value[index]);
}

template <typename T>
__device__ T storeFault(float value);

template <>
__device__ float storeFault<float>(float value)
{
    return value;
}

template <>
__device__ half storeFault<half>(float value)
{
    return __float2half_rn(value);
}

template <typename T>
__global__ void omittedResidualFault(T const* x, float const* gamma, T* output,
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
        float const value = loadFault(x, offset + column);
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
        float const value = loadFault(x, offset + column);
        output[offset + column] = storeFault<T>(value * gamma[column] * inverseRms);
    }
}

template <typename T>
cudaError_t launchFault(T const* x, float const* gamma, T* output, std::int64_t rows,
    std::int32_t hidden, float epsilon, cudaStream_t stream) noexcept
{
    constexpr std::int32_t threads{256};
    omittedResidualFault<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
        x, gamma, output, rows, hidden, epsilon);
    return cudaPeekAtLastError();
}

} // namespace

cudaError_t launchResidualRmsNormOmitResidual(nvinfer1::DataType type, void const* x,
    float const* gamma, void* output, std::int64_t rows, std::int32_t hidden, float epsilon,
    cudaStream_t stream) noexcept
{
    if (type == nvinfer1::DataType::kFLOAT)
    {
        return launchFault(static_cast<float const*>(x), gamma, static_cast<float*>(output), rows,
            hidden, epsilon, stream);
    }
    if (type == nvinfer1::DataType::kHALF)
    {
        return launchFault(static_cast<half const*>(x), gamma, static_cast<half*>(output), rows,
            hidden, epsilon, stream);
    }
    return cudaErrorInvalidValue;
}

} // namespace upgrade_guard
