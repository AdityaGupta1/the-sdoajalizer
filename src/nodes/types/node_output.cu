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

unsigned int NodeOutput::getTitleBarSelectedColor() const
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

__global__ void kernFillSingleColor(Texture outTex, glm::vec4 col)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= outTex.resolution.x || y >= outTex.resolution.y)
    {
        return;
    }

    outTex.setColor(x, y, col);
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
        col = hdrToLdr(inTex.getColor(x, y));
    }

    outTex.setColor(x, y, col);
}

void NodeOutput::_evaluate()
{
    Texture* inTex = inputPins[0].getSingleTexture();

    if (inTex == nullptr)
    {
        nodeEvaluator->setOutputTexture(nullptr);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture();

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(outTex->resolution, blockSize);
    if (inTex->isSingleColor())
    {
        auto ldrCol = hdrToLdr(inTex->singleColor);
        kernFillSingleColor<<<blocksPerGrid, blockSize>>>(*outTex, ldrCol);
    }
    else
    {
        kernCopyToOutTex<<<blocksPerGrid, blockSize>>>(*inTex, *outTex);
    }

    nodeEvaluator->setOutputTexture(outTex);
}
