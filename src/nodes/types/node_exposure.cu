#include "node_exposure.hpp"

#include "cuda_includes.hpp"

NodeExposure::NodeExposure()
    : Node("exposure")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "image");
    addPin(PinType::INPUT, "exposure").setNoConnect();
}

__global__ void kernExposure(Texture inTex, Texture outTex, float multiplier)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= inTex.getNumPixels())
    {
        return;
    }

    glm::vec4 col = inTex.getColor<TextureType::MULTI>(idx);
    outTex.setColor<TextureType::MULTI>(idx, glm::vec4(glm::vec3(col) * multiplier, col.a));
}

bool NodeExposure::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // image
        ImGui::SameLine();
        return NodeUI::ColorEdit4(constParams.color);
    case 1: // exposure
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.exposure, 0.01f);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeExposure::_evaluate()
{
    Texture* inTex = getPinTextureOrUniformColor(inputPins[0], ColorUtils::srgbToLinear(constParams.color));

    if (inTex->isUniform()) {
        Texture* outTex = nodeEvaluator->requestUniformTexture();

        const glm::vec4 inCol = inTex->getUniformColor<TextureType::MULTI>();
        glm::vec4 outCol = glm::vec4(glm::vec3(inCol) * powf(2.f, constParams.exposure), inCol.a);
        outTex->setUniformColor(outCol);

        outputPins[0].propagateTexture(outTex);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture<TextureType::MULTI>(inTex->resolution);

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_1D);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(inTex->getNumPixels(), blockSize);
    kernExposure<<<blocksPerGrid, blockSize>>>(*inTex, *outTex, powf(2.f, constParams.exposure));

    outputPins[0].propagateTexture(outTex);
}
