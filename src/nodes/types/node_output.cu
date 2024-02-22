#include "node_output.hpp"

#include "cuda_includes.hpp"

NodeOutput::NodeOutput()
    : Node("output")
{
    addPin(PinType::INPUT, "image");
}

unsigned int NodeOutput::getTitleBarColor() const
{
    return IM_COL32(255, 85, 0, 255);
}

unsigned int NodeOutput::getTitleBarSelectedColor() const
{
    return IM_COL32(255, 128, 0, 255);
}

__host__ __device__ glm::vec4 hdrToLdr(glm::vec4 col)
{
    col = glm::vec4(glm::max(glm::vec3(col), 0.f), col.a);
    // TODO: tone mapping?
    return ColorUtils::linearToSrgb(col);
}

__global__ void kernFillSingleColor(Texture outTex, glm::vec4 col)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= outTex.resolution.x || y >= outTex.resolution.y)
    {
        return;
    }

    outTex.dev_pixels[y * outTex.resolution.x + x] = col;
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
        col = hdrToLdr(inTex.dev_pixels[y * inTex.resolution.x + x]);
    }

    outTex.dev_pixels[y * outTex.resolution.x + x] = col;
}

void NodeOutput::evaluate()
{
    Texture* inTex = inputPins[0].getSingleTexture();

    if (inTex == nullptr)
    {
        nodeEvaluator->setOutputTexture(nullptr);
        return;
    }

    Texture* outTex = nodeEvaluator->requestTexture();

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid = calculateBlocksPerGrid(outTex->resolution, blockSize);
    if (inTex->isSingleColor())
    {
        kernFillSingleColor<<<blocksPerGrid, blockSize>>>(*outTex, hdrToLdr(inTex->singleColor));
    }
    else
    {
        kernCopyToOutTex<<<blocksPerGrid, blockSize>>>(*inTex, *outTex);
    }

    nodeEvaluator->setOutputTexture(outTex);
}

std::string NodeOutput::debugGetSrcFileName() const
{
    return __FILE__;
}

