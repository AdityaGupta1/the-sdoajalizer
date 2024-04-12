#include "node_lut.hpp"

#include "cuda_includes.hpp"

NodeLUT::NodeLUT()
    : Node("LUT")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "image");
    addPin(PinType::INPUT, "LUT");
}

bool NodeLUT::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT)
    {
        return false;
    }

    ImGui::SameLine();

    bool didParameterChange;
    switch (pinNumber)
    {
    case 0: // image
        return false;
    case 1: // LUT
        didParameterChange = NodeUI::FilePicker(&filePath, { "Cube LUTs (.cube)", "*.cube" });
        break;
    default:
        throw std::runtime_error("invalid pin number");
    }

    if (didParameterChange)
    {
        needsReloadFile = true;
    }
    return didParameterChange;
}

void NodeLUT::evaluate()
{
    Texture* inTex = inputPins[0].getSingleTexture();

    outputPins[0].propagateTexture(inTex);
}
