#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

__global__ void quarantinedTailOutOfBounds(float* output, std::int32_t count)
{
    std::int32_t const index = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index <= count)
    {
        output[index] = 1.0F;
    }
}

int main()
{
    constexpr std::int32_t count{257};
    float* output{nullptr};
    if (cudaMalloc(&output, count * sizeof(float)) != cudaSuccess)
    {
        return 2;
    }
    quarantinedTailOutOfBounds<<<2, 256>>>(output, count);
    cudaError_t const result = cudaDeviceSynchronize();
    cudaFree(output);
    if (result != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(result) << '\n';
        return 1;
    }
    return 0;
}
