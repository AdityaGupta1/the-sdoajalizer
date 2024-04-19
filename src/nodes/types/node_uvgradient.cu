#include "node_uvgradient.hpp"

#include "cuda_includes.hpp"

NodeUvGradient::NodeUvGradient()
    : Node("uv gradient")
{
    addPin(PinType::OUTPUT, "image");
}

__global__ void kernUvGradient(Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= outTex.resolution.x || y >= outTex.resolution.y)
    {
        return;
    }

    glm::vec2 uv = glm::vec2(x, y) / glm::vec2(outTex.resolution);
    outTex.setColor<TextureType::MULTI>(x, y, glm::vec4(uv, 0, 1));
}

void NodeUvGradient::_evaluate()
{
    Texture* outTex = nodeEvaluator->requestTexture<TextureType::MULTI>();

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_2D_X, DEFAULT_BLOCK_SIZE_2D_Y);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(outTex->resolution, blockSize);
    kernUvGradient<<<blocksPerGrid, blockSize>>>(*outTex);

    outputPins[0].propagateTexture(outTex);
}
