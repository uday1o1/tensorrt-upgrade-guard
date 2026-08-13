#include "plugin/residual_rmsnorm_plugin.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>

namespace upgrade_guard
{
namespace
{

bool dimensionsEqual(nvinfer1::Dims const& first, nvinfer1::Dims const& second) noexcept
{
    if (first.nbDims != second.nbDims)
    {
        return false;
    }
    for (std::int32_t index = 0; index < first.nbDims; ++index)
    {
        if (first.d[index] != second.d[index])
        {
            return false;
        }
    }
    return true;
}

bool validActivationDimensions(nvinfer1::Dims const& dimensions) noexcept
{
    return (dimensions.nbDims == 2 || dimensions.nbDims == 3)
        && dimensions.d[dimensions.nbDims - 1] > 0
        && dimensions.d[dimensions.nbDims - 1] <= std::numeric_limits<std::int32_t>::max();
}

bool validBuildDimensions(nvinfer1::DynamicPluginTensorDesc const* inputs, std::int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, std::int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 3 || nbOutputs != 1)
    {
        return false;
    }
    for (auto selector : {&nvinfer1::DynamicPluginTensorDesc::min,
             &nvinfer1::DynamicPluginTensorDesc::max})
    {
        auto const& x = inputs[0].*selector;
        auto const& residual = inputs[1].*selector;
        auto const& gamma = inputs[2].*selector;
        auto const& output = outputs[0].*selector;
        if (!validActivationDimensions(x) || !dimensionsEqual(x, residual) || !dimensionsEqual(x, output)
            || gamma.nbDims != 1 || gamma.d[0] != x.d[x.nbDims - 1])
        {
            return false;
        }
    }
    return true;
}

} // namespace

ResidualRmsNormPlugin::ResidualRmsNormPlugin(float epsilon, std::int64_t extraWorkspaceBytes)
    : mEpsilon(epsilon)
    , mExtraWorkspaceBytes(extraWorkspaceBytes)
{
    char identity[128]{};
    std::snprintf(identity, sizeof(identity), "v=%d;epsilon=%.9g;workspace=%lld",
        kSerializationVersion, static_cast<double>(mEpsilon),
        static_cast<long long>(mExtraWorkspaceBytes));
    mTimingCacheId = identity;
    mMetadata = identity;
    initializeSerializationFields();
}

ResidualRmsNormPlugin::ResidualRmsNormPlugin(ResidualRmsNormPlugin const& other)
    : mEpsilon(other.mEpsilon)
    , mExtraWorkspaceBytes(other.mExtraWorkspaceBytes)
    , mSerializationVersion(other.mSerializationVersion)
    , mTactic(other.mTactic)
    , mRows(other.mRows)
    , mHidden(other.mHidden)
    , mTimingCacheId(other.mTimingCacheId)
    , mMetadata(other.mMetadata)
{
    initializeSerializationFields();
}

nvinfer1::IPluginCapability* ResidualRmsNormPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept
{
    if (type == nvinfer1::PluginCapabilityType::kBUILD)
    {
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME)
    {
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kCORE)
    {
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    }
    return nullptr;
}

nvinfer1::IPluginV3* ResidualRmsNormPlugin::clone() noexcept
{
    return new (std::nothrow) ResidualRmsNormPlugin(*this);
}

char const* ResidualRmsNormPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* ResidualRmsNormPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* ResidualRmsNormPlugin::getPluginNamespace() const noexcept
{
    return kPluginNamespace;
}

std::int32_t ResidualRmsNormPlugin::getNbOutputs() const noexcept
{
    return 1;
}

std::int32_t ResidualRmsNormPlugin::getOutputDataTypes(nvinfer1::DataType* outputTypes,
    std::int32_t nbOutputs, nvinfer1::DataType const* inputTypes, std::int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != 3 || nbOutputs != 1
        || inputTypes[0] != inputTypes[1]
        || (inputTypes[0] != nvinfer1::DataType::kFLOAT
            && inputTypes[0] != nvinfer1::DataType::kHALF)
        || inputTypes[2] != nvinfer1::DataType::kFLOAT)
    {
        return -1;
    }
    outputTypes[0] = inputTypes[0];
    return 0;
}

std::int32_t ResidualRmsNormPlugin::getOutputShapes(nvinfer1::DimsExprs const* inputs,
    std::int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs, std::int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs, std::int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    static_cast<void>(shapeInputs);
    static_cast<void>(exprBuilder);
    if (inputs == nullptr || outputs == nullptr || nbInputs != 3 || nbShapeInputs != 0
        || nbOutputs != 1 || (inputs[0].nbDims != 2 && inputs[0].nbDims != 3)
        || inputs[2].nbDims != 1)
    {
        return -1;
    }
    outputs[0] = inputs[0];
    return 0;
}

bool ResidualRmsNormPlugin::supportsFormatCombination(std::int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut, std::int32_t nbInputs,
    std::int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != 3 || nbOutputs != 1 || pos < 0 || pos >= 4)
    {
        return false;
    }
    auto const& current = inOut[pos].desc;
    if (current.format != nvinfer1::TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 0)
    {
        return current.type == nvinfer1::DataType::kFLOAT
            || current.type == nvinfer1::DataType::kHALF;
    }
    if (pos == 1 || pos == 3)
    {
        return current.type == inOut[0].desc.type;
    }
    return current.type == nvinfer1::DataType::kFLOAT;
}

std::int32_t ResidualRmsNormPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs, std::int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, std::int32_t nbOutputs) noexcept
{
    if (!std::isfinite(mEpsilon) || mEpsilon <= 0.0F || mExtraWorkspaceBytes < 0)
    {
        return -1;
    }
    return validBuildDimensions(inputs, nbInputs, outputs, nbOutputs) ? 0 : -1;
}

std::size_t ResidualRmsNormPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs, std::int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, std::int32_t nbOutputs) const noexcept
{
    if (!validBuildDimensions(inputs, nbInputs, outputs, nbOutputs) || mExtraWorkspaceBytes < 0
        || static_cast<std::uint64_t>(mExtraWorkspaceBytes)
            > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        return 0;
    }
    return static_cast<std::size_t>(mExtraWorkspaceBytes);
}

std::int32_t ResidualRmsNormPlugin::getNbTactics() noexcept
{
    return 2;
}

std::int32_t ResidualRmsNormPlugin::getValidTactics(
    std::int32_t* tactics, std::int32_t nbTactics) noexcept
{
    if (tactics == nullptr || nbTactics != 2)
    {
        return -1;
    }
    tactics[0] = static_cast<std::int32_t>(ResidualRmsNormTactic::kScalarReference);
    tactics[1] = static_cast<std::int32_t>(ResidualRmsNormTactic::kVectorizedWarp);
    return 0;
}

char const* ResidualRmsNormPlugin::getTimingCacheID() noexcept
{
    return mTimingCacheId.c_str();
}

char const* ResidualRmsNormPlugin::getMetadataString() noexcept
{
    return mMetadata.c_str();
}

std::int32_t ResidualRmsNormPlugin::setTactic(std::int32_t tactic) noexcept
{
    if (tactic != static_cast<std::int32_t>(ResidualRmsNormTactic::kScalarReference)
        && tactic != static_cast<std::int32_t>(ResidualRmsNormTactic::kVectorizedWarp))
    {
        return -1;
    }
    mTactic = tactic;
    return 0;
}

std::int32_t ResidualRmsNormPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
    std::int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
    std::int32_t nbOutputs) noexcept
{
    return validateRuntimeShapes(inputs, nbInputs, outputs, nbOutputs) ? 0 : -1;
}

std::int32_t ResidualRmsNormPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    static_cast<void>(workspace);
    if (inputs == nullptr || outputs == nullptr
        || !validateRuntimeShapes(inputDesc, 3, outputDesc, 1))
    {
        return -1;
    }
    cudaError_t result{cudaErrorInvalidValue};
    if (mTactic == static_cast<std::int32_t>(ResidualRmsNormTactic::kScalarReference))
    {
        result = launchResidualRmsNormScalar(inputDesc[0].type, inputs[0], inputs[1],
            static_cast<float const*>(inputs[2]), outputs[0], mRows, mHidden, mEpsilon, stream);
    }
    else if (mTactic == static_cast<std::int32_t>(ResidualRmsNormTactic::kVectorizedWarp))
    {
        result = launchResidualRmsNormOptimized(inputDesc[0].type, inputs[0], inputs[1],
            static_cast<float const*>(inputs[2]), outputs[0], mRows, mHidden, mEpsilon, stream);
    }
    return result == cudaSuccess ? 0 : -1;
}

nvinfer1::IPluginV3* ResidualRmsNormPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* context) noexcept
{
    static_cast<void>(context);
    return clone();
}

nvinfer1::PluginFieldCollection const* ResidualRmsNormPlugin::getFieldsToSerialize() noexcept
{
    initializeSerializationFields();
    return &mSerializationCollection;
}

bool ResidualRmsNormPlugin::validateRuntimeShapes(nvinfer1::PluginTensorDesc const* inputs,
    std::int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
    std::int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 3 || nbOutputs != 1
        || !validActivationDimensions(inputs[0].dims)
        || !dimensionsEqual(inputs[0].dims, inputs[1].dims)
        || !dimensionsEqual(inputs[0].dims, outputs[0].dims) || inputs[2].dims.nbDims != 1
        || inputs[2].dims.d[0] != inputs[0].dims.d[inputs[0].dims.nbDims - 1]
        || inputs[0].type != inputs[1].type || inputs[0].type != outputs[0].type
        || (inputs[0].type != nvinfer1::DataType::kFLOAT
            && inputs[0].type != nvinfer1::DataType::kHALF)
        || inputs[2].type != nvinfer1::DataType::kFLOAT)
    {
        return false;
    }
    std::int64_t rows{1};
    for (std::int32_t dimension = 0; dimension < inputs[0].dims.nbDims - 1; ++dimension)
    {
        if (inputs[0].dims.d[dimension] <= 0
            || rows > std::numeric_limits<std::int64_t>::max() / inputs[0].dims.d[dimension])
        {
            return false;
        }
        rows *= inputs[0].dims.d[dimension];
    }
    mRows = rows;
    mHidden = static_cast<std::int32_t>(inputs[0].dims.d[inputs[0].dims.nbDims - 1]);
    return true;
}

void ResidualRmsNormPlugin::initializeSerializationFields()
{
    mSerializationFields.clear();
    mSerializationFields.emplace_back(
        "serialization_version", &mSerializationVersion, nvinfer1::PluginFieldType::kINT32, 1);
    mSerializationFields.emplace_back(
        "epsilon", &mEpsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
    mSerializationFields.emplace_back("extra_workspace_bytes", &mExtraWorkspaceBytes,
        nvinfer1::PluginFieldType::kINT64, 1);
    mSerializationCollection.nbFields = static_cast<std::int32_t>(mSerializationFields.size());
    mSerializationCollection.fields = mSerializationFields.data();
}

} // namespace upgrade_guard
