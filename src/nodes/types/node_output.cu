#include "node_output.hpp"

#include "cuda_includes.hpp"

NodeOutput::NodeOutput()
    : Node("output")
{
    addPin(PinType::INPUT, "image");
}

unsigned int NodeOutput::getTitleBarColor() const
{
    return IM_COL32(190, 85, 0, 255);
}

unsigned int NodeOutput::getTitleBarHoveredColor() const
{
    return IM_COL32(255, 129, 66, 255);
}

bool NodeOutput::drawPinExtras(const Pin* pin, int pinNumber)
{
    return false;
}

__host__ __device__ glm::vec4 hdrToLdr(glm::vec4 col)
{
    glm::vec3 rgb = glm::max(glm::vec3(col), 0.f);
    rgb = ColorUtils::linearToSrgb(rgb);
    return glm::vec4(rgb, col.a);
}

__global__ void kernFillUniformColor(Texture outTex, glm::vec4 col)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= outTex.getNumPixels())
    {
        return;
    }

    outTex.setColor<TextureType::MULTI>(idx, col);
}

__global__ void kernCopyToOutTex(Texture inTex, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= outTex.resolution.x || y >= outTex.resolution.y)
    {
        return;
    }

    glm::vec4 col;
    if (x >= inTex.resolution.x || y >= inTex.resolution.y)
    {
        col = glm::vec4(0, 0, 0, 1);
    }
    else
    {
        col = hdrToLdr(inTex.getColor<TextureType::MULTI>(x, y));
    }

    outTex.setColor<TextureType::MULTI>(x, y, col);
}

void NodeOutput::_evaluate()
{
    Texture* inTex = inputPins[0].getSingleTexture();

    if (inTex == nullptr)
    {
        nodeEvaluator->setOutputTexture(nullptr);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture<TextureType::MULTI>();

    if (inTex->isUniform())
    {
        glm::vec4 ldrCol = hdrToLdr(inTex->getUniformColor<TextureType::MULTI>());

        const dim3 blockSize(DEFAULT_BLOCK_SIZE_1D);
        const dim3 blocksPerGrid = calculateNumBlocksPerGrid(outTex->getNumPixels(), blockSize);
        kernFillUniformColor<<<blocksPerGrid, blockSize>>>(*outTex, ldrCol);
    }
    else
    {
        const dim3 blockSize(DEFAULT_BLOCK_SIZE_2D_X, DEFAULT_BLOCK_SIZE_2D_Y);
        const dim3 blocksPerGrid = calculateNumBlocksPerGrid(outTex->resolution, blockSize);
        kernCopyToOutTex<<<blocksPerGrid, blockSize>>>(*inTex, *outTex);
    }

    nodeEvaluator->setOutputTexture(outTex);
}
