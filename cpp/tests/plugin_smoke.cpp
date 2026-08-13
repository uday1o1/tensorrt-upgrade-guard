#include "plugin/residual_rmsnorm_plugin.hpp"

#include <NvInfer.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>

namespace
{

bool require(bool condition, char const* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
    }
    return condition;
}

nvinfer1::Dims dimensions(std::initializer_list<std::int64_t> values)
{
    nvinfer1::Dims result{};
    result.nbDims = static_cast<std::int32_t>(values.size());
    std::int32_t index{0};
    for (std::int64_t value : values)
    {
        result.d[index++] = value;
    }
    return result;
}

nvinfer1::PluginTensorDesc runtimeDescriptor(
    nvinfer1::DataType type, nvinfer1::Dims const& shape)
{
    nvinfer1::PluginTensorDesc result{};
    result.dims = shape;
    result.type = type;
    result.format = nvinfer1::TensorFormat::kLINEAR;
    return result;
}

nvinfer1::DynamicPluginTensorDesc buildDescriptor(
    nvinfer1::DataType type, nvinfer1::Dims const& shape)
{
    nvinfer1::DynamicPluginTensorDesc result{};
    result.desc = runtimeDescriptor(type, shape);
    result.min = shape;
    result.opt = shape;
    result.max = shape;
    return result;
}

} // namespace

int main()
{
    upgrade_guard::ResidualRmsNormPluginCreator creator;
    float epsilon{1e-5F};
    std::int32_t serializationVersion{upgrade_guard::kSerializationVersion};
    std::int64_t workspace{0};
    std::array<nvinfer1::PluginField, 3> fields{
        nvinfer1::PluginField{"serialization_version", &serializationVersion,
            nvinfer1::PluginFieldType::kINT32, 1},
        nvinfer1::PluginField{"epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1},
        nvinfer1::PluginField{
            "extra_workspace_bytes", &workspace, nvinfer1::PluginFieldType::kINT64, 1},
    };
    nvinfer1::PluginFieldCollection collection{
        static_cast<std::int32_t>(fields.size()), fields.data()};
    std::unique_ptr<nvinfer1::IPluginV3> plugin{
        creator.createPlugin("test", &collection, nvinfer1::TensorRTPhase::kBUILD)};
    bool ok = require(plugin != nullptr, "valid creator fields were rejected");
    if (!ok)
    {
        return 1;
    }
    auto* build = static_cast<nvinfer1::IPluginV3OneBuild*>(
        plugin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD));
    auto* runtime = static_cast<nvinfer1::IPluginV3OneRuntime*>(
        plugin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
    ok = require(build != nullptr && runtime != nullptr, "capability interface is missing") && ok;
    std::array<nvinfer1::DataType, 3> inputTypes{nvinfer1::DataType::kHALF,
        nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType outputType{nvinfer1::DataType::kFLOAT};
    ok = require(build->getOutputDataTypes(&outputType, 1, inputTypes.data(), 3) == 0,
             "valid output type negotiation failed")
        && ok;
    ok = require(outputType == nvinfer1::DataType::kHALF, "output type did not follow activation") && ok;
    std::array<std::int32_t, 2> tactics{};
    ok = require(build->getNbTactics() == 2, "plugin did not advertise two tactics") && ok;
    ok = require(build->getValidTactics(tactics.data(), 2) == 0 && tactics[0] != tactics[1]
            && tactics[0] != 0 && tactics[1] != 0,
             "plugin tactics are invalid")
        && ok;
    ok = require(runtime->setTactic(tactics[0]) == 0 && runtime->setTactic(tactics[1]) == 0,
             "valid tactic selection failed")
        && ok;
    ok = require(runtime->setTactic(999) != 0, "unknown tactic was accepted") && ok;
    nvinfer1::Dims const activationShape = dimensions({2, 17, 256});
    nvinfer1::Dims const gammaShape = dimensions({256});
    std::array<nvinfer1::DynamicPluginTensorDesc, 3> buildInputs{
        buildDescriptor(nvinfer1::DataType::kHALF, activationShape),
        buildDescriptor(nvinfer1::DataType::kHALF, activationShape),
        buildDescriptor(nvinfer1::DataType::kFLOAT, gammaShape),
    };
    std::array<nvinfer1::DynamicPluginTensorDesc, 1> buildOutputs{
        buildDescriptor(nvinfer1::DataType::kHALF, activationShape)};
    ok = require(build->configurePlugin(buildInputs.data(), 3, buildOutputs.data(), 1) == 0,
             "valid dynamic profile was rejected")
        && ok;
    auto invalidGamma = buildInputs;
    invalidGamma[2].max = dimensions({255});
    ok = require(build->configurePlugin(invalidGamma.data(), 3, buildOutputs.data(), 1) != 0,
             "mismatched gamma profile was accepted")
        && ok;
    auto invalidRank = buildInputs;
    invalidRank[0].min = dimensions({256});
    ok = require(build->configurePlugin(invalidRank.data(), 3, buildOutputs.data(), 1) != 0,
             "invalid activation rank was accepted")
        && ok;
    std::array<nvinfer1::DataType, 3> invalidTypes{nvinfer1::DataType::kINT32,
        nvinfer1::DataType::kINT32, nvinfer1::DataType::kFLOAT};
    ok = require(build->getOutputDataTypes(&outputType, 1, invalidTypes.data(), 3) != 0,
             "invalid activation dtype was accepted")
        && ok;
    std::array<nvinfer1::PluginTensorDesc, 3> runtimeInputs{
        runtimeDescriptor(nvinfer1::DataType::kHALF, activationShape),
        runtimeDescriptor(nvinfer1::DataType::kHALF, activationShape),
        runtimeDescriptor(nvinfer1::DataType::kFLOAT, gammaShape),
    };
    std::array<nvinfer1::PluginTensorDesc, 1> runtimeOutputs{
        runtimeDescriptor(nvinfer1::DataType::kHALF, activationShape)};
    ok = require(runtime->onShapeChange(runtimeInputs.data(), 3, runtimeOutputs.data(), 1) == 0,
             "valid runtime shape was rejected")
        && ok;
    runtimeInputs[1].dims = dimensions({2, 18, 256});
    ok = require(runtime->onShapeChange(runtimeInputs.data(), 3, runtimeOutputs.data(), 1) != 0,
             "invalid runtime profile transition was accepted")
        && ok;
    std::unique_ptr<nvinfer1::IPluginV3> clone{plugin->clone()};
    ok = require(clone != nullptr, "context-safe clone failed") && ok;
    auto* serialized = runtime->getFieldsToSerialize();
    ok = require(serialized != nullptr && serialized->nbFields == 3,
             "serialization fields are incomplete")
        && ok;
    epsilon = 0.0F;
    std::unique_ptr<nvinfer1::IPluginV3> invalid{
        creator.createPlugin("invalid", &collection, nvinfer1::TensorRTPhase::kBUILD)};
    ok = require(invalid == nullptr, "nonpositive epsilon was accepted") && ok;
    return ok ? 0 : 1;
}
