#include "kernels/residual_rmsnorm_launch.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
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

__global__ void identity(float const* input, float* output, std::uint32_t delayNanoseconds)
{
    if (delayNanoseconds > 0U)
    {
        __nanosleep(delayNanoseconds);
    }
    output[threadIdx.x] = input[threadIdx.x];
}

struct NumericalSeed
{
    bool detected;
    bool controlPassed;
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
    return {faultDetected, controlPassed, observedFault, reference};
}

struct NonfiniteSeed
{
    bool detected;
    bool controlPassed;
};

NonfiniteSeed nonfiniteSeed()
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
    std::vector<float> gamma(zeros.size(), 1.0F);
    float* residual{nullptr};
    float* deviceGamma{nullptr};
    bool controlOk = cudaMalloc(&residual, zeros.size() * sizeof(float)) == cudaSuccess
        && cudaMalloc(&deviceGamma, gamma.size() * sizeof(float)) == cudaSuccess
        && cudaMemset(residual, 0, zeros.size() * sizeof(float)) == cudaSuccess
        && cudaMemcpy(deviceGamma, gamma.data(), gamma.size() * sizeof(float),
               cudaMemcpyHostToDevice)
            == cudaSuccess;
    if (controlOk)
    {
        controlOk = upgrade_guard::launchResidualRmsNormScalar(nvinfer1::DataType::kFLOAT,
                        input, residual, deviceGamma, output, 1,
                        static_cast<std::int32_t>(zeros.size()), 1e-5F, nullptr)
                == cudaSuccess
            && cudaDeviceSynchronize() == cudaSuccess
            && cudaMemcpy(zeros.data(), output, zeros.size() * sizeof(float),
                   cudaMemcpyDeviceToHost)
                == cudaSuccess;
    }
    bool const controlPassed = controlOk
        && std::all_of(zeros.begin(), zeros.end(),
            [](float value) { return std::isfinite(value) && value == 0.0F; });
    cudaFree(deviceGamma);
    cudaFree(residual);
    cudaFree(output);
    cudaFree(input);
    return {detected, controlPassed};
}

float timeIdentity(float* input, float* output, std::uint32_t delayNanoseconds)
{
    constexpr std::int32_t repetitions{2000};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (std::int32_t index = 0; index < repetitions; ++index)
    {
        identity<<<1, 32>>>(input, output, delayNanoseconds);
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
    std::int32_t pairIndex;
    bool baselineFirst;
    float baselineMilliseconds;
    float candidateMilliseconds;
    float ratio;
    std::uint32_t delayNanoseconds;
    bool calibrated;
    bool resultPreserved;
};

PerformanceSeed performanceSeed(std::int32_t pairIndex)
{
    constexpr float targetRatio{1.10F};
    constexpr std::array<bool, 20> baselineFirstSchedule{true, false, false, true, true,
        false, true, false, false, true, false, true, true, false, false, true, true, false,
        true, false};
    float* input{nullptr};
    float* output{nullptr};
    if (cudaMalloc(&input, 32 * sizeof(float)) != cudaSuccess
        || cudaMalloc(&output, 32 * sizeof(float)) != cudaSuccess
        || cudaMemset(input, 0, 32 * sizeof(float)) != cudaSuccess)
    {
        cudaFree(output);
        cudaFree(input);
        return {pairIndex, true, NAN, NAN, NAN, 0U, false, false};
    }
    std::uint32_t low{1U};
    std::uint32_t high{200000U};
    std::uint32_t selectedDelay{0U};
    float selectedDistance{INFINITY};
    float calibrationRatio{NAN};
    for (std::int32_t iteration = 0; iteration < 18; ++iteration)
    {
        std::uint32_t const delay = low + (high - low) / 2U;
        float const baseline = timeIdentity(input, output, 0U);
        float const candidate = timeIdentity(input, output, delay);
        float const ratio = candidate / baseline;
        float const distance = std::abs(ratio - targetRatio);
        if (std::isfinite(ratio) && distance < selectedDistance)
        {
            selectedDelay = delay;
            selectedDistance = distance;
            calibrationRatio = ratio;
        }
        if (!std::isfinite(ratio) || ratio >= targetRatio)
        {
            high = delay > 1U ? delay - 1U : 1U;
        }
        else
        {
            low = delay + 1U;
        }
        if (low > high)
        {
            break;
        }
    }
    bool const baselineFirst = baselineFirstSchedule[static_cast<std::size_t>(pairIndex)];
    float baseline{NAN};
    float candidate{NAN};
    if (baselineFirst)
    {
        baseline = timeIdentity(input, output, 0U);
        candidate = timeIdentity(input, output, selectedDelay);
    }
    else
    {
        candidate = timeIdentity(input, output, selectedDelay);
        baseline = timeIdentity(input, output, 0U);
    }
    std::vector<float> observed(32, 1.0F);
    bool const resultPreserved
        = cudaMemcpy(observed.data(), output, observed.size() * sizeof(float),
              cudaMemcpyDeviceToHost)
            == cudaSuccess
        && std::all_of(observed.begin(), observed.end(),
            [](float value) { return value == 0.0F; });
    cudaFree(output);
    cudaFree(input);
    bool const calibrated = selectedDelay > 0U && std::isfinite(calibrationRatio)
        && calibrationRatio >= 1.06F && calibrationRatio <= 1.20F;
    return {pairIndex, baselineFirst, baseline, candidate, candidate / baseline, selectedDelay,
        calibrated, resultPreserved};
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3 || std::string(argv[1]) != "--pair-index")
    {
        std::cerr << "usage: upgrade_guard_gpu_faults --pair-index INDEX\n";
        return 2;
    }
    char* end{nullptr};
    long const parsedPairIndex = std::strtol(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || parsedPairIndex < 0 || parsedPairIndex >= 20)
    {
        std::cerr << "pair index must be in [0, 19]\n";
        return 2;
    }
    std::int32_t const pairIndex = static_cast<std::int32_t>(parsedPairIndex);
    NumericalSeed const numerical = numericalSeed();
    NonfiniteSeed const nonfinite = nonfiniteSeed();
    PerformanceSeed const performanceSeedResult = performanceSeed(pairIndex);
    float const ratio = performanceSeedResult.ratio;
    bool const performance = performanceSeedResult.resultPreserved
        && performanceSeedResult.calibrated && std::isfinite(ratio) && ratio > 1.03F;
    std::cout << std::fixed << std::setprecision(6)
              << "{\"schema_version\":\"upgradeguard.dev/gpu-faults/v1\","
              << "\"G2\":{\"expected\":\"NUMERICAL_REGRESSION\",\"detected\":"
              << (numerical.detected ? "true" : "false") << ",\"control\":\""
              << (numerical.controlPassed ? "passed" : "failed")
              << "\",\"observed\":" << numerical.observed
              << ",\"reference\":" << numerical.reference << "},"
              << "\"G3\":{\"expected\":\"NONFINITE_OUTPUT\",\"detected\":"
              << (nonfinite.detected ? "true" : "false") << ",\"control\":\""
              << (nonfinite.controlPassed ? "passed" : "failed") << "\"},"
              << "\"G5\":{\"expected\":\"PERFORMANCE_REGRESSION\",\"detected\":"
              << (performance ? "true" : "false") << ",\"control\":\""
              << (performanceSeedResult.resultPreserved ? "passed" : "failed")
              << "\",\"pair_index\":" << performanceSeedResult.pairIndex
              << ",\"order\":\""
              << (performanceSeedResult.baselineFirst ? "baseline_then_candidate"
                                                      : "candidate_then_baseline")
              << "\",\"order_seed\":20260813,\"target_ratio\":1.100000,"
              << "\"baseline_ms\":" << performanceSeedResult.baselineMilliseconds
              << ",\"candidate_ms\":" << performanceSeedResult.candidateMilliseconds
              << ",\"ratio\":" << ratio << ",\"delay_nanoseconds\":"
              << performanceSeedResult.delayNanoseconds << ",\"calibrated\":"
              << (performanceSeedResult.calibrated ? "true" : "false") << "}}\n";
    return numerical.detected && numerical.controlPassed && nonfinite.detected
            && nonfinite.controlPassed && performanceSeedResult.resultPreserved
            && performanceSeedResult.calibrated
            && std::isfinite(performanceSeedResult.baselineMilliseconds)
            && performanceSeedResult.baselineMilliseconds > 0.0F
            && std::isfinite(performanceSeedResult.candidateMilliseconds)
            && performanceSeedResult.candidateMilliseconds > 0.0F
        ? 0
        : 1;
}
