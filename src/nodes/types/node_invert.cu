#include "node_invert.hpp"

#include "cuda_includes.hpp"

NodeInvert::NodeInvert()
    : Node("invert")
{
    addPin(PinType::INPUT);
    addPin(PinType::OUTPUT);
}

__host__ __device__ glm::vec4 invertCol(glm::vec4 col)
{
    return glm::vec4(1.f - glm::vec3(col), col.a);
}

__global__ void kernInvert(Texture inTex, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= inTex.resolution.x || y >= inTex.resolution.y)
    {
        return;
    }

    int idx = y * inTex.resolution.x + x;
    glm::vec4 col = inTex.dev_pixels[idx];
    outTex.dev_pixels[idx] = invertCol(col);
}

bool NodeInvert::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // input
        ImGui::SameLine();
        return NodeUI::ColorEdit4(backupCol);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeInvert::evaluate()
{
    Texture* inTex = getPinTextureOrSingleColor(inputPins[0], ColorUtils::srgbToLinear(backupCol));

    if (inTex->isSingleColor())
    {
        Texture* outTex = nodeEvaluator->requestSingleColorTexture();
        outTex->setColor(invertCol(inTex->singleColor));
        outputPins[0].propagateTexture(outTex);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture(inTex->resolution);

    const dim3 blockSize(16, 16);
    const dim3 blocksPerGrid(inTex->resolution.x / 16 + 1, inTex->resolution.y / 16 + 1);
    kernInvert<<<blocksPerGrid, blockSize>>>(*inTex, *outTex);

    outputPins[0].propagateTexture(outTex);
}
