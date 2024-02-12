#include "node_mix.hpp"

#include "cuda_includes.hpp"

#include <stdexcept>

NodeMix::NodeMix()
    : Node("mix")
{
    addPin(PinType::INPUT, "input 1");
    addPin(PinType::INPUT, "input 2");
    addPin(PinType::OUTPUT);
}

__host__ __device__ glm::vec4 mixCols(glm::vec4 col1, glm::vec4 col2, float factor)
{
    return glm::mix(col1, col2, factor);
}

__global__ void kernMix(Texture inTex1, Texture inTex2, float factor, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    glm::vec4 col1, col2;

    if (x >= inTex1.resolution.x || y >= inTex1.resolution.y)
    {
        return;
    }

    int idx1 = y * inTex1.resolution.x + x;
    col1 = inTex1.dev_pixels[idx1];

    if (inTex2.isSingleColor())
    {
        col2 = inTex2.singleColor;
    }
    else
    {
        if (x < inTex2.resolution.x && y < inTex2.resolution.y)
        {
            int idx2 = y * inTex2.resolution.x + x;
            col2 = inTex2.dev_pixels[idx2];
        }
        else
        {
            col2 = glm::vec4(0, 0, 0, 1);
        }
    }

    outTex.dev_pixels[idx1] = mixCols(col1, col2, factor);
}

bool NodeMix::drawInputPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // input 1
        ImGui::SameLine();
        return ImGui::ColorEdit4("", glm::value_ptr(backupCol1));
    case 1: // input 2
        ImGui::SameLine();
        return ImGui::ColorEdit4("", glm::value_ptr(backupCol2));
    case 2: // factor
        // TODO
        return false;
    default:
        throw std::runtime_error("invalid pin number");
    }
}

// should work for differing resolutions but that hasn't been tested yet
void NodeMix::evaluate()
{
    Texture* inTex1 = getPinTextureOrSingleColor(inputPins[0], backupCol1);
    Texture* inTex2 = getPinTextureOrSingleColor(inputPins[1], backupCol2);

    if (inTex1->isSingleColor() && inTex2->isSingleColor())
    {
        Texture* outTex = nodeEvaluator->requestSingleColorTexture();
        outTex->setColor(mixCols(inTex1->singleColor, inTex2->singleColor, factor));

        outputPins[0].propagateTexture(outTex);
        return;
    }

    float realFactor;
    if (inTex1->isSingleColor())
    {
        std::swap(inTex1, inTex2);
        realFactor = 1.f - factor;
    }
    else
    {
        realFactor = factor;
    }

    // inTex1 is not a single color, inTex2 may or may not be a single color
    // additionally, neither one is nullptr

    Texture* outTex = nodeEvaluator->requestTexture(inTex1->resolution);

    const dim3 blockSize(16, 16);
    const dim3 blocksPerGrid(inTex1->resolution.x / 16 + 1, inTex1->resolution.y / 16 + 1);
    kernMix<<<blocksPerGrid, blockSize>>>(*inTex1, *inTex2, realFactor, *outTex);

    outputPins[0].propagateTexture(outTex);
}