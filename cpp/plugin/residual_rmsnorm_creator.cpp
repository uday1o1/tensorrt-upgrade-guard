#include "plugin/residual_rmsnorm_plugin.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string_view>

namespace upgrade_guard
{

ResidualRmsNormPluginCreator::ResidualRmsNormPluginCreator()
{
    mFields.emplace_back("serialization_version", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFields.emplace_back("epsilon", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
    mFields.emplace_back("extra_workspace_bytes", nullptr, nvinfer1::PluginFieldType::kINT64, 1);
    mCollection.nbFields = static_cast<std::int32_t>(mFields.size());
    mCollection.fields = mFields.data();
}

char const* ResidualRmsNormPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* ResidualRmsNormPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* ResidualRmsNormPluginCreator::getPluginNamespace() const noexcept
{
    return kPluginNamespace;
}

nvinfer1::PluginFieldCollection const* ResidualRmsNormPluginCreator::getFieldNames() noexcept
{
    return &mCollection;
}

nvinfer1::IPluginV3* ResidualRmsNormPluginCreator::createPlugin(char const* name,
    nvinfer1::PluginFieldCollection const* fields, nvinfer1::TensorRTPhase phase) noexcept
{
    static_cast<void>(name);
    static_cast<void>(phase);
    if (fields == nullptr)
    {
        return nullptr;
    }
    float epsilon{1e-5F};
    std::int64_t extraWorkspaceBytes{0};
    std::int32_t serializationVersion{kSerializationVersion};
    for (std::int32_t index = 0; index < fields->nbFields; ++index)
    {
        auto const& field = fields->fields[index];
        std::string_view const fieldName{field.name == nullptr ? "" : field.name};
        if (field.length != 1 || field.data == nullptr)
        {
            return nullptr;
        }
        if (fieldName == "epsilon")
        {
            if (field.type != nvinfer1::PluginFieldType::kFLOAT32)
            {
                return nullptr;
            }
            epsilon = *static_cast<float const*>(field.data);
        }
        else if (fieldName == "extra_workspace_bytes")
        {
            if (field.type != nvinfer1::PluginFieldType::kINT64)
            {
                return nullptr;
            }
            extraWorkspaceBytes = *static_cast<std::int64_t const*>(field.data);
        }
        else if (fieldName == "serialization_version")
        {
            if (field.type != nvinfer1::PluginFieldType::kINT32)
            {
                return nullptr;
            }
            serializationVersion = *static_cast<std::int32_t const*>(field.data);
        }
        else
        {
            return nullptr;
        }
    }
    if (serializationVersion != kSerializationVersion || !std::isfinite(epsilon) || epsilon <= 0.0F
        || extraWorkspaceBytes < 0)
    {
        return nullptr;
    }
    return new (std::nothrow) ResidualRmsNormPlugin(epsilon, extraWorkspaceBytes);
}

} // namespace upgrade_guard
