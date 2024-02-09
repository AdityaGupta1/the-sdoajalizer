#include "node_noise.hpp"

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

NodeNoise::NodeNoise()
    : Node("noise")
{
    addPins(0, 1);
}

__global__ void kernNoise(Texture texture)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= texture.resolution.x || y >= texture.resolution.y)
    {
        return;
    }

    glm::vec2 uv = glm::vec2(x, y) / glm::vec2(texture.resolution);
    texture.dev_pixels[y * texture.resolution.x + x] = glm::vec4(uv, 0, 1);
}

void NodeNoise::evaluate()
{
    Texture* texture = nodeEvaluator->requestTexture();

    const dim3 blockSize(16, 16);
    const dim3 blocksPerGrid(texture->resolution.x / 16 + 1, texture->resolution.y / 16 + 1);
    kernNoise<<<blocksPerGrid, blockSize>>>(*texture);

    outputPins[0].propagateTexture(texture);
}