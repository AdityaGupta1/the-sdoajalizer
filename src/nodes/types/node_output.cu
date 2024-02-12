#include "node_output.hpp"

#include "cuda_includes.hpp"

NodeOutput::NodeOutput()
    : Node("output")
{
    addPin(PinType::INPUT);
}

unsigned int NodeOutput::getTitleBarColor() const
{
    return IM_COL32(255, 85, 0, 255);
}

unsigned int NodeOutput::getTitleBarSelectedColor() const
{
    return IM_COL32(255, 128, 0, 255);
}

__global__ void kernFillSingleColor(Texture outTex, glm::vec4 col)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= outTex.resolution.x || y >= outTex.resolution.y)
    {
        return;
    }

    outTex.dev_pixels[y * outTex.resolution.x + x] = col;
}

void NodeOutput::evaluate()
{
    Texture* inTex = inputPins[0].getSingleTexture();

    if (inTex == nullptr)
    {
        nodeEvaluator->setOutputTexture(nullptr);
        return;
    }

    if (!inTex->isSingleColor())
    {
        nodeEvaluator->setOutputTexture(inTex);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture();

    const dim3 blockSize(16, 16);
    const dim3 blocksPerGrid(outTex->resolution.x / 16 + 1, outTex->resolution.y / 16 + 1);
    kernFillSingleColor<<<blocksPerGrid, blockSize>>>(*outTex, inTex->singleColor);

    nodeEvaluator->setOutputTexture(outTex);
}
