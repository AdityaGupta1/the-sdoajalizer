#include "node_color.hpp"

#include "cuda_includes.hpp"

NodeColor::NodeColor()
    : Node("color")
{
    addPin(PinType::OUTPUT, "image");
}

bool NodeColor::drawPinExtras(const Pin* pin, int pinNumber)
{
    switch (pinNumber)
    {
    case 0: // image
        ImGui::SameLine();
        return NodeUI::ColorEdit4(constParams.color);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeColor::evaluate()
{
    Texture* outTex = nodeEvaluator->requestSingleColorTexture();
    outTex->setSingleColor(ColorUtils::srgbToLinear(constParams.color));
    outputPins[0].propagateTexture(outTex);
}

std::string NodeColor::debugGetSrcFileName() const
{
    return __FILE__;
}
