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
    outTex.dev_pixels[y * outTex.resolution.x + x] = glm::vec4(uv, 0, 1);
}

void NodeUvGradient::evaluate()
{
    Texture* outTex = nodeEvaluator->requestTexture();

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(outTex->resolution, blockSize);
    kernUvGradient<<<blocksPerGrid, blockSize>>>(*outTex);

    outputPins[0].propagateTexture(outTex);
}

std::string NodeUvGradient::debugGetSrcFileName() const
{
    return __FILE__;
}
