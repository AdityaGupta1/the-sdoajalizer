#include "node_invert.hpp"

#include "../../cuda_includes.hpp"

NodeInvert::NodeInvert()
    : Node("invert")
{
    addPins(1, 1);
}

__global__ void kernInvert(Texture texture)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= texture.resolution.x || y >= texture.resolution.y)
    {
        return;
    }

    int idx = y * texture.resolution.x + x;
    glm::vec4 col = texture.dev_pixels[idx];
    texture.dev_pixels[idx] = glm::vec4(1.f - glm::vec3(col), col.a);
}

void NodeInvert::evaluate()
{
    Texture* texture = inputPins[0].getSingleTexture();

    const dim3 blockSize(16, 16);
    const dim3 blocksPerGrid(texture->resolution.x / 16 + 1, texture->resolution.y / 16 + 1);
    kernInvert<<<blocksPerGrid, blockSize>>>(*texture);

    outputPins[0].propagateTexture(texture);
}