#include "node_noise.hpp"

#include "cuda_includes.hpp"

#include <glm/gtc/noise.hpp>

NodeNoise::NodeNoise()
    : Node("noise")
{
    addPins(0, 1);
}

__global__ void kernNoise(Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= outTex.resolution.x || y >= outTex.resolution.y)
    {
        return;
    }

    float noise = glm::simplex(glm::vec2(x, y) * 0.005f);
    outTex.dev_pixels[y * outTex.resolution.x + x] = glm::vec4(glm::vec3(noise), 1);
}

void NodeNoise::evaluate()
{
    Texture* outTex = nodeEvaluator->requestTexture();

    const dim3 blockSize(16, 16);
    const dim3 blocksPerGrid(outTex->resolution.x / 16 + 1, outTex->resolution.y / 16 + 1);
    kernNoise<<<blocksPerGrid, blockSize>>>(*outTex);

    outputPins[0].propagateTexture(outTex);

    if (getIsSelected())
    {
        nodeEvaluator->setSelectedTexture(outTex);
    }
}
