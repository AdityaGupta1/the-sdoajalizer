#include "node_noise.hpp"

#include <cuda_runtime.h>

NodeNoise::NodeNoise()
    : Node("noise")
{
    addPins(0, 1);
}

__global__ void kernNoise(Texture tex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= tex.resolution.x || y >= tex.resolution.y)
    {
        return;
    }

    glm::vec2 uv = glm::vec2(x, y) / glm::vec2(tex.resolution);
}

void NodeNoise::evaluate()
{
    Texture* texture = nodeEvaluator->requestTexture();
    outputPins[0].propagateTexture(texture);
}