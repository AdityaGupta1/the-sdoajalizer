#include "node_noise.hpp"

#include "cuda_includes.hpp"

#include <glm/gtc/noise.hpp>

NodeNoise::NodeNoise()
    : Node("noise")
{
    addPin(PinType::OUTPUT, "image");
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

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(outTex->resolution, blockSize);
    kernNoise<<<blocksPerGrid, blockSize>>>(*outTex);

    outputPins[0].propagateTexture(outTex);
}
