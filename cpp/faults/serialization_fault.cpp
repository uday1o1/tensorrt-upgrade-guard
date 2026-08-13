#include "plugin/residual_rmsnorm_plugin.hpp"

#include <NvInfer.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>

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

} // namespace

int main()
{
    constexpr float expected{0.0125F};
    auto control = create(expected);
    auto fault = create(1e-5F);
    bool const controlPassed
        = control != nullptr && std::abs(serializedEpsilon(*control) - expected) < 1e-8F;
    bool const faultDetected
        = fault != nullptr && std::abs(serializedEpsilon(*fault) - expected) > 1e-3F;
    std::cout << "{\"schema_version\":\"upgradeguard.dev/serialization-fault/v1\","
              << "\"expected\":\"NUMERICAL_REGRESSION\",\"detected\":"
              << (faultDetected ? "true" : "false") << ",\"control\":\""
              << (controlPassed ? "passed" : "failed") << "\"}\n";
    return controlPassed && faultDetected ? 0 : 1;
}
