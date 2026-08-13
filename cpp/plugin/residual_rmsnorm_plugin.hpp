#pragma once

#include "kernels/residual_rmsnorm_launch.hpp"

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace upgrade_guard
{

inline constexpr char kPluginName[] = "ResidualRMSNorm";
inline constexpr char kPluginVersion[] = "1";
inline constexpr char kPluginNamespace[] = "com.udayarora.upgradeguard";
inline constexpr std::int32_t kSerializationVersion{1};

class ResidualRmsNormPlugin final : public nvinfer1::IPluginV3,
                                    public nvinfer1::IPluginV3OneCore,
                                    public nvinfer1::IPluginV3OneBuild,
                                    public nvinfer1::IPluginV3OneRuntime
{
public:
    ResidualRmsNormPlugin(float epsilon, std::int64_t extraWorkspaceBytes = 0);
    ResidualRmsNormPlugin(ResidualRmsNormPlugin const& other);

    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    std::int32_t getNbOutputs() const noexcept override;
    std::int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, std::int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, std::int32_t nbInputs) const noexcept override;
    std::int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, std::int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, std::int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, std::int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(std::int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
        std::int32_t nbInputs, std::int32_t nbOutputs) noexcept override;
    std::int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs, std::int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, std::int32_t nbOutputs) noexcept override;
    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, std::int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, std::int32_t nbOutputs) const noexcept override;
    std::int32_t getNbTactics() noexcept override;
    std::int32_t getValidTactics(std::int32_t* tactics, std::int32_t nbTactics) noexcept override;
    char const* getTimingCacheID() noexcept override;
    char const* getMetadataString() noexcept override;

    std::int32_t setTactic(std::int32_t tactic) noexcept override;
    std::int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs, std::int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, std::int32_t nbOutputs) noexcept override;
    std::int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    bool validateRuntimeShapes(nvinfer1::PluginTensorDesc const* inputs, std::int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, std::int32_t nbOutputs) noexcept;
    void initializeSerializationFields();

    float mEpsilon;
    std::int64_t mExtraWorkspaceBytes;
    std::int32_t mSerializationVersion{kSerializationVersion};
    std::int32_t mTactic{static_cast<std::int32_t>(ResidualRmsNormTactic::kScalarReference)};
    std::int64_t mRows{0};
    std::int32_t mHidden{0};
    std::string mTimingCacheId;
    std::string mMetadata;
    std::vector<nvinfer1::PluginField> mSerializationFields;
    nvinfer1::PluginFieldCollection mSerializationCollection{};
};

class ResidualRmsNormPluginCreator final : public nvinfer1::IPluginCreatorV3One
{
public:
    ResidualRmsNormPluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV3* createPlugin(char const* name,
        nvinfer1::PluginFieldCollection const* fields,
        nvinfer1::TensorRTPhase phase) noexcept override;

private:
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mCollection{};
};

} // namespace upgrade_guard
