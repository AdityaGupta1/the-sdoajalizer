#include "node_colorramp.hpp"

#include "cuda_includes.hpp"

NodeColorRamp::NodeColorRamp()
    : Node("color ramp")
{
    addPin(PinType::OUTPUT, "color");

    addPin(PinType::INPUT, "factor").setSingleChannel();
}

bool NodeColorRamp::drawPinBeforeExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge() || pinNumber != 0) // factor
    {
        return false;
    }

    return gradientWidget.widget("a");
}

bool NodeColorRamp::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // factor
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.factor, 0.01f, 0.f, 1.f);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeColorRamp::_evaluate()
{
    //Texture* inTex = getPinTextureOrUniformColor(inputPins[0], ColorUtils::srgbToLinear(constParams.color));

    //if (inTex->isUniform())
    //{
    //    Texture* outTex = nodeEvaluator->requestUniformTexture();
    //    outTex->setUniformColor(invertCol(inTex->getUniformColor<TextureType::MULTI>()));
    //    outputPins[0].propagateTexture(outTex);
    //    return;
    //}

    //Texture* outTex = nodeEvaluator->requestTexture<TextureType::MULTI>(inTex->resolution);

    //const dim3 blockSize(DEFAULT_BLOCK_SIZE_1D);
    //const dim3 blocksPerGrid = calculateNumBlocksPerGrid(inTex->getNumPixels(), blockSize);
    //kernInvert << <blocksPerGrid, blockSize >> > (*inTex, *outTex);

    //outputPins[0].propagateTexture(outTex);
}
