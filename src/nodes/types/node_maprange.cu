#include "node_maprange.hpp"

#include "cuda_includes.hpp"

NodeMapRange::NodeMapRange()
    : Node("map range")
{
    addPin(PinType::OUTPUT, "value").setSingleChannel();

    addPin(PinType::INPUT, "value").setSingleChannel();
    addPin(PinType::INPUT, "old min").setNoConnect();
    addPin(PinType::INPUT, "old max").setNoConnect();
    addPin(PinType::INPUT, "new min").setNoConnect();
    addPin(PinType::INPUT, "new max").setNoConnect();
}

__host__ __device__ float mapRange(float v, float oldMin, float oldMax, float newMin, float newMax)
{
    float denom = oldMax - oldMin;
    if (denom == 0.f)
    {
        return 0.f;
    }

    float t = (v - oldMin) / denom;
    return newMin + t * newMax;
}

__global__ void kernMapRange(Texture inTex, Texture outTex, float oldMin, float oldMax, float newMin, float newMax)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= inTex.getNumPixels())
    {
        return;
    }

    float outValue = mapRange(inTex.getColor<TextureType::SINGLE>(idx), oldMin, oldMax, newMin, newMax);
    outTex.setColor<TextureType::SINGLE>(idx, outValue);
}

bool NodeMapRange::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // value
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.value, 0.01f);
    case 1: // old min
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.oldMin, 0.01f);
    case 2: // old max
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.oldMax, 0.01f);
    case 3: // new min
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.newMin, 0.01f);
    case 4: // new max
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.newMax, 0.01f);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeMapRange::_evaluate()
{
    Texture* inTex = getPinTextureOrUniformColor(inputPins[0], constParams.value);

    if (inTex->isUniform())
    {
        Texture* outTex = nodeEvaluator->requestUniformTexture();

        const float inValue = constParams.value;
        float outValue = mapRange(inValue, constParams.oldMin, constParams.oldMax, constParams.newMin, constParams.newMax);
        outTex->setUniformColor(outValue);

        outputPins[0].propagateTexture(outTex);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture<TextureType::SINGLE>(inTex->resolution);

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_1D);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(inTex->getNumPixels(), blockSize);
    kernMapRange<<<blocksPerGrid, blockSize>>>(
        *inTex, *outTex, 
        constParams.oldMin, constParams.oldMax, 
        constParams.newMin, constParams.newMax
    );

    outputPins[0].propagateTexture(outTex);
}
