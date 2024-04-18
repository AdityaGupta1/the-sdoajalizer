#include "node_mix.hpp"

#include "cuda_includes.hpp"

NodeMix::NodeMix()
    : Node("mix")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "image 1");
    addPin(PinType::INPUT, "image 2");
    addPin(PinType::INPUT, "factor");
}

__host__ __device__ glm::vec4 mixCols(glm::vec4 col1, glm::vec4 col2, float factor)
{
    return glm::mix(col1, col2, factor);
}

__global__ void kernMix(Texture inTex1, Texture inTex2, Texture inTexFactor, glm::ivec2 outRes, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= outRes.x || y >= outRes.y)
    {
        return;
    }

    glm::vec4 col1 = inTex1.getColorClamp(x, y);
    glm::vec4 col2 = inTex2.getColorClamp(x, y);
    float factor = inTexFactor.getColorClamp(x, y).r;

    outTex.dev_pixels[y * outRes.x + x] = mixCols(col1, col2, factor);
}

bool NodeMix::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // in color 1
        ImGui::SameLine();
        return NodeUI::ColorEdit4(constParams.color1);
    case 1: // in color 2
        ImGui::SameLine();
        return NodeUI::ColorEdit4(constParams.color2);
    case 2: // factor
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.factor, 0.01f, 0.f, 1.f);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

// should work for differing resolutions but that hasn't been tested yet
void NodeMix::_evaluate()
{
    Texture* inTex1 = getPinTextureOrSingleColor(inputPins[0], ColorUtils::srgbToLinear(constParams.color1));
    Texture* inTex2 = getPinTextureOrSingleColor(inputPins[1], ColorUtils::srgbToLinear(constParams.color2));
    Texture* inTexFactor = getPinTextureOrSingleColor(inputPins[2], constParams.factor);

    if (inTex1->isSingleColor() && inTex2->isSingleColor() && inTexFactor->isSingleColor())
    {
        Texture* outTex = nodeEvaluator->requestSingleColorTexture();
        outTex->setSingleColor(mixCols(inTex1->singleColor, inTex2->singleColor, inTexFactor->singleColor.r));

        outputPins[0].propagateTexture(outTex);
        return;
    }

    glm::ivec2 outRes = Texture::getFirstResolution({ inTex1, inTex2, inTexFactor });

    Texture* outTex = nodeEvaluator->requestTexture(outRes);

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(outRes, blockSize);
    kernMix<<<blocksPerGrid, blockSize>>>(*inTex1, *inTex2, *inTexFactor, outRes, *outTex);

    outputPins[0].propagateTexture(outTex);
}
