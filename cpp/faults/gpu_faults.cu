#include "plugin/residual_rmsnorm_plugin.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

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

NumericalSeed numericalSeed(std::int32_t rows, std::int32_t hidden, float xValue,
    float residualValue, float gammaValue)
{
    std::size_t const elements
        = static_cast<std::size_t>(rows) * static_cast<std::size_t>(hidden);
    std::vector<float> x(elements, xValue);
    std::vector<float> residual(elements, residualValue);
    std::vector<float> gamma(static_cast<std::size_t>(hidden), gammaValue);
    std::vector<float> output(elements);
    float* deviceX{nullptr};
    float* deviceResidual{nullptr};
    float* deviceGamma{nullptr};
    float* deviceOutput{nullptr};
    std::size_t const bytes = elements * sizeof(float);
    std::size_t const gammaBytes = static_cast<std::size_t>(hidden) * sizeof(float);
    bool ok = cudaMalloc(&deviceX, bytes) == cudaSuccess
        && cudaMalloc(&deviceResidual, bytes) == cudaSuccess
        && cudaMalloc(&deviceGamma, gammaBytes) == cudaSuccess
        && cudaMalloc(&deviceOutput, bytes) == cudaSuccess
        && cudaMemcpy(deviceX, x.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess
        && cudaMemcpy(deviceResidual, residual.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess
        && cudaMemcpy(deviceGamma, gamma.data(), gammaBytes, cudaMemcpyHostToDevice)
            == cudaSuccess;
    if (ok)
    {
        upgrade_guard::ResidualRmsNormPlugin fault{1e-5F};
        nvinfer1::PluginTensorDesc inputDesc[3]{};
        nvinfer1::PluginTensorDesc outputDesc[1]{};
        inputDesc[0].dims.nbDims = 3;
        inputDesc[0].dims.d[0] = rows;
        inputDesc[0].dims.d[1] = 1;
        inputDesc[0].dims.d[2] = hidden;
        inputDesc[0].type = nvinfer1::DataType::kFLOAT;
        inputDesc[0].format = nvinfer1::TensorFormat::kLINEAR;
        inputDesc[1] = inputDesc[0];
        inputDesc[2].dims.nbDims = 1;
        inputDesc[2].dims.d[0] = hidden;
        inputDesc[2].type = nvinfer1::DataType::kFLOAT;
        inputDesc[2].format = nvinfer1::TensorFormat::kLINEAR;
        outputDesc[0] = inputDesc[0];
        void const* pluginInputs[3]{deviceX, deviceResidual, deviceGamma};
        void* pluginOutputs[1]{deviceOutput};
        ok = fault.setTactic(
                 static_cast<std::int32_t>(upgrade_guard::ResidualRmsNormTactic::kScalarReference))
                == 0
            && fault.onShapeChange(inputDesc, 3, outputDesc, 1) == 0
            && fault.enqueue(inputDesc, outputDesc, pluginInputs, pluginOutputs, nullptr, nullptr) == 0
            && cudaDeviceSynchronize() == cudaSuccess
            && cudaMemcpy(output.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    float const combined = xValue + residualValue;
    float const reference
        = combined * gammaValue / std::sqrt(combined * combined + 1e-5F);
    float const observedFault = output[0];
    bool const faultDetected = ok && std::abs(observedFault - reference) > 0.1F;
    if (ok)
    {
        ok = upgrade_guard::launchResidualRmsNormScalar(nvinfer1::DataType::kFLOAT, deviceX,
                 deviceResidual, deviceGamma, deviceOutput, rows, hidden, 1e-5F, nullptr)
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

constexpr std::array<bool, 24> PERFORMANCE_ORDER_SCHEDULE{true, false, false, true, true,
    false, true, false, false, true, false, true, true, false, false, true, true, false, true,
    false, true, false, true, false};

PerformanceSeed performanceSeed(std::int32_t pairIndex)
{
    constexpr float targetRatio{1.10F};
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
    bool const baselineFirst = PERFORMANCE_ORDER_SCHEDULE[static_cast<std::size_t>(pairIndex)];
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
    std::int32_t pairIndex{-1};
    std::int32_t rows{1};
    std::int32_t hidden{259};
    float xValue{0.5F};
    float residualValue{0.25F};
    float gammaValue{1.0F};
    bool onlyG2{false};
    auto parseInteger = [](char const* text, std::int32_t& value) {
        char* end{nullptr};
        long const parsed = std::strtol(text, &end, 10);
        if (end == text || *end != '\0' || parsed < 0
            || parsed > static_cast<long>(std::numeric_limits<std::int32_t>::max()))
        {
            return false;
        }
        value = static_cast<std::int32_t>(parsed);
        return true;
    };
    auto parseFloat = [](char const* text, float& value) {
        char* end{nullptr};
        float const parsed = std::strtof(text, &end);
        if (end == text || *end != '\0' || !std::isfinite(parsed))
        {
            return false;
        }
        value = parsed;
        return true;
    };
    for (int index = 1; index < argc; ++index)
    {
        std::string const option{argv[index]};
        if (option == "--only-g2")
        {
            onlyG2 = true;
            continue;
        }
        if (index + 1 >= argc)
        {
            std::cerr << "missing value for " << option << "\n";
            return 2;
        }
        bool valid{false};
        if (option == "--pair-index")
        {
            valid = parseInteger(argv[++index], pairIndex);
        }
        else if (option == "--rows")
        {
            valid = parseInteger(argv[++index], rows);
        }
        else if (option == "--hidden")
        {
            valid = parseInteger(argv[++index], hidden);
        }
        else if (option == "--x-value")
        {
            valid = parseFloat(argv[++index], xValue);
        }
        else if (option == "--residual-value")
        {
            valid = parseFloat(argv[++index], residualValue);
        }
        else if (option == "--gamma-value")
        {
            valid = parseFloat(argv[++index], gammaValue);
        }
        else
        {
            std::cerr << "unknown option: " << option << "\n";
            return 2;
        }
        if (!valid)
        {
            std::cerr << "invalid value for " << option << "\n";
            return 2;
        }
    }
    if (pairIndex < 0
        || static_cast<std::size_t>(pairIndex) >= PERFORMANCE_ORDER_SCHEDULE.size() || rows < 1
        || rows > 4096 || hidden < 1
        || hidden > 65536)
    {
        std::cerr << "pair index, rows, or hidden size is outside the bounded range\n";
        return 2;
    }
    NumericalSeed const numerical
        = numericalSeed(rows, hidden, xValue, residualValue, gammaValue);
    if (onlyG2)
    {
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"schema_version\":\"upgradeguard.dev/gpu-faults/v1\","
                  << "\"G2\":{\"mechanism\":\"plugin_omits_residual_at_hidden_259\","
                  << "\"detected\":" << (numerical.detected ? "true" : "false")
                  << ",\"control\":\""
                  << (numerical.controlPassed ? "passed" : "failed")
                  << "\",\"observed\":" << numerical.observed
                  << ",\"reference\":" << numerical.reference << ",\"rows\":" << rows
                  << ",\"hidden\":" << hidden << ",\"x_value\":" << xValue
                  << ",\"residual_value\":" << residualValue << ",\"gamma_value\":"
                  << gammaValue << "}}\n";
        return numerical.detected && numerical.controlPassed ? 0 : 1;
    }
    NonfiniteSeed const nonfinite = nonfiniteSeed();
    PerformanceSeed const performanceSeedResult = performanceSeed(pairIndex);
    float const ratio = performanceSeedResult.ratio;
    bool const performance = performanceSeedResult.resultPreserved
        && performanceSeedResult.calibrated && std::isfinite(ratio) && ratio > 1.03F;
    std::cout << std::fixed << std::setprecision(6)
              << "{\"schema_version\":\"upgradeguard.dev/gpu-faults/v1\","
              << "\"G2\":{\"mechanism\":\"plugin_omits_residual_at_hidden_259\",\"detected\":"
              << (numerical.detected ? "true" : "false") << ",\"control\":\""
              << (numerical.controlPassed ? "passed" : "failed")
              << "\",\"observed\":" << numerical.observed
              << ",\"reference\":" << numerical.reference
              << ",\"observed_facts\":{\"numerical_valid\":"
              << (numerical.detected ? "false" : "true") << "}},"
              << "\"G3\":{\"mechanism\":\"zero_epsilon_zero_input\",\"detected\":"
              << (nonfinite.detected ? "true" : "false") << ",\"control\":\""
              << (nonfinite.controlPassed ? "passed" : "failed")
              << "\",\"observed_facts\":{\"finite\":"
              << (nonfinite.detected ? "false" : "true") << "}},"
              << "\"G5\":{\"mechanism\":\"controlled_device_delay\",\"detected\":"
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
