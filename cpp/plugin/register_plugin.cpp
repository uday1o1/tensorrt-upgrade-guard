#include "plugin/residual_rmsnorm_plugin.hpp"

#include <NvInferRuntime.h>

namespace
{

class Registrar
{
public:
    Registrar()
    {
        auto* registry = getPluginRegistry();
        if (registry != nullptr)
        {
            bool const registered
                = registry->registerCreator(mCreator, upgrade_guard::kPluginNamespace);
            static_cast<void>(registered);
        }
    }

private:
    upgrade_guard::ResidualRmsNormPluginCreator mCreator;
};

Registrar gRegistrar{};

} // namespace
