#include "node_invert.hpp"

#include "cuda_includes.hpp"

NodeInvert::NodeInvert()
    : Node("invert")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "image");
}

__host__ __device__ glm::vec4 invertCol(glm::vec4 col)
{
    return glm::vec4(1.f - glm::vec3(col), col.a);
}

__global__ void kernInvert(Texture inTex, Texture outTex)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= inTex.getNumPixels())
    {
        return;
    }

    outTex.setColor<TextureType::MULTI>(idx, invertCol(inTex.getColor<TextureType::MULTI>(idx)));
}

bool NodeInvert::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // in color
        ImGui::SameLine();
        return NodeUI::ColorEdit4(constParams.color);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeInvert::_evaluate()
{
    Texture* inTex = getPinTextureOrUniformColor(inputPins[0], ColorUtils::srgbToLinear(constParams.color));

    if (inTex->isUniform())
    {
        Texture* outTex = nodeEvaluator->requestUniformTexture();
        outTex->setUniformColor(invertCol(inTex->getUniformColor<TextureType::MULTI>()));
        outputPins[0].propagateTexture(outTex);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture<TextureType::MULTI>(inTex->resolution);

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_1D);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(inTex->getNumPixels(), blockSize);
    kernInvert<<<blocksPerGrid, blockSize>>>(*inTex, *outTex);

    outputPins[0].propagateTexture(outTex);
}
