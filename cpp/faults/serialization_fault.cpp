#include "plugin/residual_rmsnorm_plugin.hpp"

#include <NvInfer.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace
{

float serializedEpsilon(nvinfer1::IPluginV3& plugin)
{
    auto* runtime = static_cast<nvinfer1::IPluginV3OneRuntime*>(
        plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
    if (runtime == nullptr)
    {
        return 0.0F;
    }
    auto const* fields = runtime->getFieldsToSerialize();
    for (std::int32_t index = 0; fields != nullptr && index < fields->nbFields; ++index)
    {
        auto const& field = fields->fields[index];
        if (field.name != nullptr && std::string_view(field.name) == "epsilon"
            && field.type == nvinfer1::PluginFieldType::kFLOAT32 && field.length == 1)
        {
            return *static_cast<float const*>(field.data);
        }
    }
    return 0.0F;
}

std::unique_ptr<nvinfer1::IPluginV3> create(float epsilon)
{
    upgrade_guard::ResidualRmsNormPluginCreator creator;
    std::int32_t version{upgrade_guard::kSerializationVersion};
    std::int64_t workspace{0};
    std::array<nvinfer1::PluginField, 3> fields{
        nvinfer1::PluginField{"serialization_version", &version,
            nvinfer1::PluginFieldType::kINT32, 1},
        nvinfer1::PluginField{"epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1},
        nvinfer1::PluginField{
            "extra_workspace_bytes", &workspace, nvinfer1::PluginFieldType::kINT64, 1},
    };
    nvinfer1::PluginFieldCollection collection{
        static_cast<std::int32_t>(fields.size()), fields.data()};
    return std::unique_ptr<nvinfer1::IPluginV3>{
        creator.createPlugin("serialization", &collection, nvinfer1::TensorRTPhase::kBUILD)};
}

bool containsEpsilon(nvinfer1::PluginFieldCollection const* fields)
{
    for (std::int32_t index = 0; fields != nullptr && index < fields->nbFields; ++index)
    {
        auto const& field = fields->fields[index];
        if (field.name != nullptr && std::string_view(field.name) == "epsilon")
        {
            return true;
        }
    }
    return false;
}

struct RestoreResult
{
    std::unique_ptr<nvinfer1::IPluginV3> plugin;
    bool epsilonPresent;
};

RestoreResult restore(
    nvinfer1::PluginFieldCollection const* serialized, bool omitEpsilon)
{
    if (serialized == nullptr)
    {
        return {nullptr, false};
    }
    std::vector<nvinfer1::PluginField> fields;
    for (std::int32_t index = 0; index < serialized->nbFields; ++index)
    {
        auto const& field = serialized->fields[index];
        if (omitEpsilon && field.name != nullptr && std::string_view(field.name) == "epsilon")
        {
            continue;
        }
        fields.push_back(field);
    }
    nvinfer1::PluginFieldCollection collection{
        static_cast<std::int32_t>(fields.size()), fields.data()};
    upgrade_guard::ResidualRmsNormPluginCreator creator;
    return {
        std::unique_ptr<nvinfer1::IPluginV3>{
            creator.createPlugin("restored", &collection, nvinfer1::TensorRTPhase::kRUNTIME)},
        containsEpsilon(&collection),
    };
}

} // namespace

int main()
{
    constexpr float expected{0.0125F};
    auto original = create(expected);
    auto* runtime = original == nullptr
        ? nullptr
        : static_cast<nvinfer1::IPluginV3OneRuntime*>(
              original->getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
    auto const* serialized = runtime == nullptr ? nullptr : runtime->getFieldsToSerialize();
    auto control = restore(serialized, false);
    auto fault = restore(serialized, true);
    float const controlEpsilon
        = control.plugin == nullptr ? 0.0F : serializedEpsilon(*control.plugin);
    float const faultEpsilon = fault.plugin == nullptr ? 0.0F : serializedEpsilon(*fault.plugin);
    bool const controlPassed = std::abs(controlEpsilon - expected) < 1e-8F;
    bool const faultDetected
        = fault.plugin != nullptr && std::abs(faultEpsilon - expected) > 1e-3F;
    bool const serializedEpsilonPresent = containsEpsilon(serialized);
    std::cout << "{\"schema_version\":\"upgradeguard.dev/serialization-fault/v1\","
              << "\"mechanism\":\"creator_omits_serialized_epsilon\",\"detected\":"
              << (faultDetected ? "true" : "false") << ",\"control\":\""
              << (controlPassed ? "passed" : "failed") << "\",\"expected_epsilon\":"
              << expected << ",\"control_epsilon\":" << controlEpsilon
              << ",\"fault_epsilon\":" << faultEpsilon
              << ",\"serialized_epsilon_present\":"
              << (serializedEpsilonPresent ? "true" : "false")
              << ",\"control_restore_epsilon_present\":"
              << (control.epsilonPresent ? "true" : "false")
              << ",\"fault_restore_epsilon_present\":"
              << (fault.epsilonPresent ? "true" : "false")
              << ",\"observed_facts\":{\"numerical_valid\":"
              << (faultDetected ? "false" : "true") << "}}\n";
    return serializedEpsilonPresent && control.epsilonPresent && !fault.epsilonPresent
            && controlPassed && faultDetected
        ? 0
        : 1;
}
