#include "node_brightnesscontrast.hpp"

#include "cuda_includes.hpp"

NodeBrightnessContrast::NodeBrightnessContrast()
    : Node("brightness/contrast")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "image");
    addPin(PinType::INPUT, "brightness").setNoConnect();
    addPin(PinType::INPUT, "contrast").setNoConnect();
}

__host__ __device__ glm::vec4 applyBrightnessContrast(glm::vec4 col, float brightness, float contrast)
{
    return glm::vec4((contrast + 1.f) * (glm::vec3(col) - 0.5f) + 0.5f + brightness, col.a);
}

__global__ void kernBrightnessContrast(Texture inTex, Texture outTex, float brightness, float contrast)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= inTex.getNumPixels())
    {
        return;
    }

    glm::vec4 outCol = applyBrightnessContrast(inTex.getColor<TextureType::MULTI>(idx), brightness, contrast);
    outTex.setColor<TextureType::MULTI>(idx, outCol);
}

bool NodeBrightnessContrast::drawPinExtras(const Pin* pin, int pinNumber)
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
    case 1: // brightness
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.brightness, 0.01f);
    case 2: // contrast
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.contrast, 0.01f, -1.f, FLT_MAX);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeBrightnessContrast::_evaluate()
{
    Texture* inTex = getPinTextureOrUniformColor(inputPins[0], ColorUtils::srgbToLinear(constParams.color));

    if (inTex->isUniform()) {
        Texture* outTex = nodeEvaluator->requestUniformTexture();

        glm::vec4 inCol = inTex->getUniformColor<TextureType::MULTI>();
        glm::vec4 outCol = applyBrightnessContrast(inCol, constParams.brightness, constParams.contrast);
        outTex->setUniformColor(outCol);

        outputPins[0].propagateTexture(outTex);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture<TextureType::MULTI>(inTex->resolution);

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_1D);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(inTex->getNumPixels(), blockSize);
    kernBrightnessContrast<<<blocksPerGrid, blockSize>>>(*inTex, *outTex, constParams.brightness, constParams.contrast);

    outputPins[0].propagateTexture(outTex);
}
