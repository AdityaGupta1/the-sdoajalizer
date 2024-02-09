#include "node_invert.hpp"

#include "../../cuda_includes.hpp"

NodeInvert::NodeInvert()
    : Node("invert")
{
    addPins(1, 1);
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
    outTex.dev_pixels[idx] = glm::vec4(1.f - glm::vec3(col), col.a);
}

void NodeInvert::evaluate()
{
    Texture* inTex = inputPins[0].getSingleTexture();

    if (inTex == nullptr)
    {
        outputPins[0].propagateTexture(nullptr);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture(inTex->resolution);

    const dim3 blockSize(16, 16);
    const dim3 blocksPerGrid(inTex->resolution.x / 16 + 1, inTex->resolution.y / 16 + 1);
    kernInvert<<<blocksPerGrid, blockSize>>>(*inTex, *outTex);

    outputPins[0].propagateTexture(outTex);
}