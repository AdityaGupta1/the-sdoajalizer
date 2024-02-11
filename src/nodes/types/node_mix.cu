#include "node_mix.hpp"

#include "cuda_includes.hpp"

NodeMix::NodeMix()
    : Node("mix")
{
    addPin(PinType::INPUT, "input 1");
    addPin(PinType::INPUT, "input 2");
    addPin(PinType::OUTPUT);
}

__global__ void kernMix(Texture inTex1, Texture inTex2, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= inTex1.resolution.x || y >= inTex1.resolution.y)
    {
        return;
    }

    int idx1 = y * inTex1.resolution.x + x;
    glm::vec4 col1 = inTex1.dev_pixels[idx1];

    glm::vec4 col2;
    if (x < inTex2.resolution.x && y < inTex2.resolution.y)
    {
        int idx2 = y * inTex2.resolution.x + x;
        col2 = inTex2.dev_pixels[idx2];
    }
    else
    {
        col2 = glm::vec4(0, 0, 0, 1);
    }

    outTex.dev_pixels[idx1] = glm::mix(col1, col2, 0.5f);
}

// should work for differing resolutions but not tested yet
void NodeMix::evaluate()
{
    Texture* inTex1 = inputPins[0].getSingleTexture();
    Texture* inTex2 = inputPins[1].getSingleTexture();

    if (inTex1 == nullptr && inTex2 == nullptr)
    {
        outputPins[0].propagateTexture(nullptr);
        return;
    }

    if (inTex1 == nullptr)
    {
        std::swap(inTex1, inTex2);
    }

    Texture* outTex = nodeEvaluator->requestTexture(inTex1->resolution);

    const dim3 blockSize(16, 16);
    const dim3 blocksPerGrid(inTex1->resolution.x / 16 + 1, inTex1->resolution.y / 16 + 1);
    kernMix<<<blocksPerGrid, blockSize>>>(*inTex1, *inTex2, *outTex);

    outputPins[0].propagateTexture(outTex);
}