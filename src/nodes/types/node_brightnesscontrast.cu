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

__global__ void kernBrightnessContrast(Texture inTex, float brightness, float contrast, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= inTex.resolution.x || y >= inTex.resolution.y)
    {
        return;
    }

    int idx = y * inTex.resolution.x + x;
    outTex.dev_pixels[idx] = applyBrightnessContrast(inTex.dev_pixels[idx], brightness, contrast);
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

void NodeBrightnessContrast::evaluate()
{
    Texture* inTex = getPinTextureOrSingleColor(inputPins[0], ColorUtils::srgbToLinear(constParams.color));

    if (inTex->isSingleColor()) {
        Texture* outTex = nodeEvaluator->requestSingleColorTexture();

        if (constParams.brightness == 0.f && constParams.contrast == 0.f) {
            outTex->setSingleColor(inTex->singleColor);
        }
        else
        {
            glm::vec4 outCol = applyBrightnessContrast(inTex->singleColor, constParams.brightness, constParams.contrast);
            outTex->setSingleColor(outCol);
        }

        outputPins[0].propagateTexture(outTex);
        return;
    }

    // inTex is not a single color
    if (constParams.brightness == 0.f && constParams.contrast == 0.f) {
        outputPins[0].propagateTexture(inTex);
        return;
    }

    // inTex is not a single color and backupExposure != 0.f
    Texture* outTex = nodeEvaluator->requestTexture(inTex->resolution);

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(inTex->resolution, blockSize);
    kernBrightnessContrast<<<blocksPerGrid, blockSize>>>(*inTex, constParams.brightness, constParams.contrast, *outTex);

    outputPins[0].propagateTexture(outTex);
}
