#include "node_exposure.hpp"

#include "cuda_includes.hpp"

NodeExposure::NodeExposure()
    : Node("exposure")
{
    addPin(PinType::INPUT, "exposure").setNoConnection();
    addPin(PinType::INPUT, "image");
    addPin(PinType::OUTPUT, "image");
}

__global__ void kernExposure(Texture inTexExposure, Texture inTexCol, Texture outTex)
{
    //const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    //const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    //if (x >= inTex.resolution.x || y >= inTex.resolution.y)
    //{
    //    return;
    //}

    //int idx = y * inTex.resolution.x + x;
    //glm::vec4 col = inTex.dev_pixels[idx];
    //outTex.dev_pixels[idx] = invertCol(col);
}

bool NodeExposure::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // exposure
        ImGui::SameLine();
        return NodeUI::FloatEdit(backupExposure, 0.1f);
    case 1: // image
        ImGui::SameLine();
        return NodeUI::ColorEdit4(backupCol);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeExposure::evaluate()
{
    Texture* inTexCol = getPinTextureOrSingleColor(inputPins[1], ColorUtils::srgbToLinear(backupCol));

    //if (inTex->isSingleColor())
    //{
    //    Texture* outTex = nodeEvaluator->requestSingleColorTexture();
    //    outTex->setColor(invertCol(inTex->singleColor));
    //    outputPins[0].propagateTexture(outTex);
    //    return;
    //}

    //Texture* outTex = nodeEvaluator->requestTexture(inTex->resolution);

    //const dim3 blockSize(16, 16);
    //const dim3 blocksPerGrid = calculateBlocksPerGrid(inTex->resolution, blockSize);
    //kernInvert<<<blocksPerGrid, blockSize>>>(*inTex, *outTex);

    //outputPins[0].propagateTexture(outTex);
}

std::string NodeExposure::debugGetSrcFileName() const
{
    return __FILE__;
}

