#include "node_mix.hpp"

#include "cuda_includes.hpp"

NodeMix::NodeMix()
    : Node("mix")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "factor").setSingleChannel();
    addPin(PinType::INPUT, "image 1");
    addPin(PinType::INPUT, "image 2");
}

bool NodeMix::drawPinBeforeExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::INPUT && pinNumber == 0) // factor
    {
        return NodeUI::Checkbox(constParams.clamp, "clamp");
    }

    return false;
}

bool NodeMix::drawPinExtras(const Pin* pin, int pinNumber)
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
    case 1: // in color 1
        ImGui::SameLine();
        return NodeUI::ColorEdit4(constParams.color1);
    case 2: // in color 2
        ImGui::SameLine();
        return NodeUI::ColorEdit4(constParams.color2);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

__host__ __device__ glm::vec4 mixCols(glm::vec4 col1, glm::vec4 col2, float factor, bool clamp)
{
    if (clamp)
    {
        factor = glm::clamp(factor, 0.f, 1.f);
    }
    return glm::mix(col1, col2, factor);
}

__global__ void kernMix(Texture inTex1, Texture inTex2, Texture inTexFactor, bool clamp, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= outTex.resolution.x || y >= outTex.resolution.y)
    {
        return;
    }

    glm::vec4 col1 = inTex1.getColorClamp<TextureType::MULTI>(x, y);
    glm::vec4 col2 = inTex2.getColorClamp<TextureType::MULTI>(x, y);
    float factor = inTexFactor.getColorClamp<TextureType::SINGLE>(x, y);

    outTex.setColor<TextureType::MULTI>(x, y, mixCols(col1, col2, factor, clamp));
}

// should work for differing resolutions but that hasn't been tested yet
void NodeMix::_evaluate()
{
    Texture* inTexFactor = getPinTextureOrUniformColor(inputPins[0], constParams.factor);
    Texture* inTex1 = getPinTextureOrUniformColor(inputPins[1], ColorUtils::srgbToLinear(constParams.color1));
    Texture* inTex2 = getPinTextureOrUniformColor(inputPins[2], ColorUtils::srgbToLinear(constParams.color2));

    if (inTex1->isUniform() && inTex2->isUniform() && inTexFactor->isUniform())
    {
        Texture* outTex = nodeEvaluator->requestUniformTexture();
        outTex->setUniformColor(mixCols(
            inTex1->getUniformColor<TextureType::MULTI>(),
            inTex2->getUniformColor<TextureType::MULTI>(),
            inTexFactor->getUniformColor<TextureType::SINGLE>(),
            constParams.clamp
        ));

        outputPins[0].propagateTexture(outTex);
        return;
    }

    glm::ivec2 outRes = Texture::getFirstResolutionFromList({ inTex1, inTex2, inTexFactor });
    Texture* outTex = nodeEvaluator->requestTexture<TextureType::MULTI>(outRes);

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_2D_X, DEFAULT_BLOCK_SIZE_2D_Y);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(outRes, blockSize);
    kernMix<<<blocksPerGrid, blockSize>>>(*inTex1, *inTex2, *inTexFactor, constParams.clamp, *outTex);

    outputPins[0].propagateTexture(outTex);
}
