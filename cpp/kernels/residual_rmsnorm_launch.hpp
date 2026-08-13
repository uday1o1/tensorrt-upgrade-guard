#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace upgrade_guard
{

enum class ResidualRmsNormTactic : std::int32_t
{
    kScalarReference = 1,
    kVectorizedWarp = 2,
};

cudaError_t launchResidualRmsNormScalar(nvinfer1::DataType type, void const* x, void const* residual,
    float const* gamma, void* output, std::int64_t rows, std::int32_t hidden, float epsilon,
    cudaStream_t stream) noexcept;

cudaError_t launchResidualRmsNormOptimized(nvinfer1::DataType type, void const* x, void const* residual,
    float const* gamma, void* output, std::int64_t rows, std::int32_t hidden, float epsilon,
    cudaStream_t stream) noexcept;

} // namespace upgrade_guard
