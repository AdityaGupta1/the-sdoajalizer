#include "node_bloom.hpp"

#include "cuda_includes.hpp"

NodeBloom::NodeBloom()
    : Node("bloom")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "image");
    addPin(PinType::INPUT, "threshold").setNoConnect();
    addPin(PinType::INPUT, "iterations").setNoConnect();
}

__global__ void kernCopyWithThreshold(Texture inTex, float threshold, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= inTex.resolution.x || y >= inTex.resolution.y)
    {
        return;
    }

    int idx = y * inTex.resolution.x + x;
    glm::vec4 inCol = inTex.dev_pixels[idx];

    glm::vec4 outCol;
    if (ColorUtils::luminance(glm::vec3(inCol)) >= threshold)
    {
        outCol = inCol;
    }
    else
    {
        outCol = glm::vec4(0, 0, 0, 1);
    }

    outTex.dev_pixels[idx] = outCol;
}

// http://demofox.org/gauss.html
// sigma = 2, support = 0.995
__constant__ float constant_gaussianKernel[] = {
    0.0000,	0.0000,	0.0000,	0.0000,	0.0000,	0.0001,	0.0001,	0.0001,	0.0001,	0.0001,	0.0000,	0.0000,	0.0000,	0.0000,	0.0000,
    0.0000,	0.0000,	0.0000,	0.0001,	0.0002,	0.0003,	0.0004,	0.0005,	0.0004,	0.0003,	0.0002,	0.0001,	0.0000,	0.0000,	0.0000,
    0.0000,	0.0000,	0.0001,	0.0003,	0.0006,	0.0011,	0.0016,	0.0018,	0.0016,	0.0011,	0.0006,	0.0003,	0.0001,	0.0000,	0.0000,
    0.0000,	0.0001,	0.0003,	0.0008,	0.0018,	0.0034,	0.0049,	0.0055,	0.0049,	0.0034,	0.0018,	0.0008,	0.0003,	0.0001,	0.0000,
    0.0000,	0.0002,	0.0006,	0.0018,	0.0043,	0.0079,	0.0115,	0.0130,	0.0115,	0.0079,	0.0043,	0.0018,	0.0006,	0.0002,	0.0000,
    0.0001,	0.0003,	0.0011,	0.0034,	0.0079,	0.0146,	0.0211,	0.0239,	0.0211,	0.0146,	0.0079,	0.0034,	0.0011,	0.0003,	0.0001,
    0.0001,	0.0004,	0.0016,	0.0049,	0.0115,	0.0211,	0.0305,	0.0345,	0.0305,	0.0211,	0.0115,	0.0049,	0.0016,	0.0004,	0.0001,
    0.0001,	0.0005,	0.0018,	0.0055,	0.0130,	0.0239,	0.0345,	0.0390,	0.0345,	0.0239,	0.0130,	0.0055,	0.0018,	0.0005,	0.0001,
    0.0001,	0.0004,	0.0016,	0.0049,	0.0115,	0.0211,	0.0305,	0.0345,	0.0305,	0.0211,	0.0115,	0.0049,	0.0016,	0.0004,	0.0001,
    0.0001,	0.0003,	0.0011,	0.0034,	0.0079,	0.0146,	0.0211,	0.0239,	0.0211,	0.0146,	0.0079,	0.0034,	0.0011,	0.0003,	0.0001,
    0.0000,	0.0002,	0.0006,	0.0018,	0.0043,	0.0079,	0.0115,	0.0130,	0.0115,	0.0079,	0.0043,	0.0018,	0.0006,	0.0002,	0.0000,
    0.0000,	0.0001,	0.0003,	0.0008,	0.0018,	0.0034,	0.0049,	0.0055,	0.0049,	0.0034,	0.0018,	0.0008,	0.0003,	0.0001,	0.0000,
    0.0000,	0.0000,	0.0001,	0.0003,	0.0006,	0.0011,	0.0016,	0.0018,	0.0016,	0.0011,	0.0006,	0.0003,	0.0001,	0.0000,	0.0000,
    0.0000,	0.0000,	0.0000,	0.0001,	0.0002,	0.0003,	0.0004,	0.0005,	0.0004,	0.0003,	0.0002,	0.0001,	0.0000,	0.0000,	0.0000,
    0.0000,	0.0000,	0.0000,	0.0000,	0.0000,	0.0001,	0.0001,	0.0001,	0.0001,	0.0001,	0.0000,	0.0000,	0.0000,	0.0000,	0.0000
};
#define gaussianKernelRadius 7

__global__ void kernBlur(Texture inTex, Texture outTex)
{
    // assuming block size of 32 x 32
    constexpr int sharedMemXSize = 32 + 2 * gaussianKernelRadius;
    constexpr int sharedMemYSize = 32 + 2 * gaussianKernelRadius;
    constexpr int sharedMemSize = sharedMemXSize * sharedMemYSize;
    __shared__ glm::vec4 shared_colors[sharedMemSize];

    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int localIdx = threadIdx.y * blockDim.x + threadIdx.x; // 0 to 1024
    const glm::ivec2 cornerPos = glm::ivec2(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y);
    glm::ivec2 loadPos, storePos;

    for (int storeIdx = localIdx; storeIdx < sharedMemSize; storeIdx += blockDim.x * blockDim.y)
    {
        int sharedMemX = storeIdx % sharedMemXSize;
        int sharedMemY = storeIdx / sharedMemXSize;

        loadPos.x = cornerPos.x + sharedMemX - gaussianKernelRadius;
        loadPos.y = cornerPos.y + sharedMemY - gaussianKernelRadius;

        if (loadPos.x < 0)
        {
            loadPos.x = -(loadPos.x + 1);
        }
        else if (loadPos.x >= inTex.resolution.x)
        {
            loadPos.x = 2 * inTex.resolution.x - loadPos.x - 1;
        }

        if (loadPos.y < 0)
        {
            loadPos.y = -(loadPos.y + 1);
        }
        else if (loadPos.y >= inTex.resolution.y)
        {
            loadPos.y = 2 * inTex.resolution.y - loadPos.y - 1;
        }

        shared_colors[storeIdx] = inTex.dev_pixels[loadPos.y * inTex.resolution.x + loadPos.x];
    }

    __syncthreads();

    if (x >= inTex.resolution.x || y >= inTex.resolution.y)
    {
        return;
    }

    glm::vec4 sum = glm::vec4(0.f);
    float weightSum = 0.f;

    for (int dy = 0; dy <= 2 * gaussianKernelRadius; ++dy)
    {
        for (int dx = 0; dx <= 2 * gaussianKernelRadius; ++dx)
        {
            int sampleX = threadIdx.x + dx;
            int sampleY = threadIdx.y + dy;

            float kernelValue = constant_gaussianKernel[dy * (2 * gaussianKernelRadius + 1) + dx];

            glm::vec4 sampleColor = shared_colors[sampleY * sharedMemXSize + sampleX];
            sum += sampleColor * kernelValue;
            weightSum += kernelValue;
        }
    }

    glm::vec4 outCol = sum / weightSum;

    int idx = y * inTex.resolution.x + x;
    outTex.dev_pixels[idx] = outCol;
}

__global__ void kernAdd(Texture inTex1, Texture inTex2, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= inTex1.resolution.x || y >= inTex1.resolution.y)
    {
        return;
    }

    int idx = y * inTex1.resolution.x + x;
    glm::vec4 inCol1 = inTex1.dev_pixels[idx];
    glm::vec4 inCol2 = inTex2.dev_pixels[idx];
    outTex.dev_pixels[idx] = glm::vec4(glm::vec3(inCol1) + glm::vec3(inCol2), inCol1.a);
}

bool NodeBloom::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pin->pinType == PinType::OUTPUT || pin->hasEdge())
    {
        return false;
    }

    switch (pinNumber)
    {
    case 0: // image
        return false;
    case 1: // threshold
        ImGui::SameLine();
        return NodeUI::FloatEdit(backupThreshold, 0.01f, 0.f, FLT_MAX);
    case 2: // iterations
        ImGui::SameLine();
        return NodeUI::IntEdit(backupIterations, 0.10f, 1, INT_MAX);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeBloom::evaluate()
{
    Texture* inTex = getPinTextureOrSingleColor(inputPins[0], glm::vec4(0, 0, 0, 1));

    if (inTex->isSingleColor())
    {
        // bloom has no real meaning for single color textures
        // could run bloom on a texture made up of that single color but that feels kind of useless
        outputPins[0].propagateTexture(inTex);
        return;
    }

    Texture* outTex1 = nodeEvaluator->requestTexture(inTex->resolution);
    Texture* outTex2 = nodeEvaluator->requestTexture(inTex->resolution);

    const dim3 blockSize(DEFAULT_BLOCK_SIZE_X, DEFAULT_BLOCK_SIZE_Y);
    const dim3 blocksPerGrid = calculateBlocksPerGrid(inTex->resolution, blockSize);

    kernCopyWithThreshold<<<blocksPerGrid, blockSize>>>(*inTex, backupThreshold, *outTex1);

    const dim3 blurBlockSize(32, 32);
    const dim3 blurBlocksPerGrid = calculateBlocksPerGrid(inTex->resolution, blurBlockSize);
    for (int i = 0; i < backupIterations; ++i)
    {
        kernBlur<<<blurBlocksPerGrid, blurBlockSize>>>(*outTex1, *outTex2);
        std::swap(outTex1, outTex2);
    }

    kernAdd<<<blocksPerGrid, blockSize>>>(*inTex, *outTex1, *outTex2);

    outputPins[0].propagateTexture(outTex2);
}

std::string NodeBloom::debugGetSrcFileName() const
{
    return __FILE__;
}

