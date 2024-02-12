#include "node_color.hpp"

#include "cuda_includes.hpp"

NodeColor::NodeColor()
    : Node("color")
{
    addPin(PinType::OUTPUT);
}

bool NodeColor::drawPinExtras(const Pin* pin, int pinNumber)
{
    switch (pinNumber)
    {
    case 0: // output
        ImGui::SameLine();
        return NodeUI::ColorEdit4(backupCol);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeColor::evaluate()
{
    Texture* outTex = nodeEvaluator->requestSingleColorTexture();
    outTex->setColor(backupCol);
    outputPins[0].propagateTexture(outTex);
}