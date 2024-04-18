#include "node_bloom.hpp"

#include "cuda_includes.hpp"

#include "npp_includes.hpp"

std::array<float*, NodeBloom::numBloomKernels> NodeBloom::dev_bloomKernels = {};

NodeBloom::NodeBloom()
    : Node("bloom")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "image");
    addPin(PinType::INPUT, "threshold").setNoConnect();
    addPin(PinType::INPUT, "size").setNoConnect();
    addPin(PinType::INPUT, "mix").setNoConnect();

    setExpensive();
}

void NodeBloom::freeDeviceMemory()
{
    for (auto& dev_kernel : dev_bloomKernels)
    {
        if (dev_kernel != nullptr)
        {
            cudaFree(dev_kernel);
            dev_kernel = nullptr;
        }
    }
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
    glm::vec3 inRgb = glm::vec3(inCol) * inCol.a;

    glm::vec4 outCol;
    if (ColorUtils::luminance(inRgb) >= threshold)
    {
        outCol = glm::vec4(inRgb, 1);
    }
    else
    {
        outCol = glm::vec4(0, 0, 0, 1);
    }

    outTex.dev_pixels[idx] = outCol;
}

__device__ float calculateKernelWeight(float u, float v, float scale)
{
    float r = (u * u + v * v) * scale;
    float d = -powf(r, 0.0625f) * 9.0f;
    float f = expf(d);
    float w = (0.5f + 0.5f * cosf(u * glm::pi<float>())) * (0.5f + 0.5f * cosf(v * glm::pi<float>()));
    return f * w;
}

// https://github.com/blender/blender/blob/7da72a938c05fe5662db3654f8dbd02a67c0150b/source/blender/compositor/operations/COM_GlareFogGlowOperation.cc
__global__ void kernFillBlurKernel(float* kernel, int kernelDiameter, float scale)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= kernelDiameter || y >= kernelDiameter)
    {
        return;
    }

    float u = 2.0f * (x / (float)kernelDiameter) - 1.0f;
    float v = 2.0f * (y / (float)kernelDiameter) - 1.0f;

    int idx = y * kernelDiameter + x;
    kernel[idx] = calculateKernelWeight(u, v, scale);
}

__global__ void kernAdd(Texture inTexBase, Texture inTexProcessed, float mix, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= inTexBase.resolution.x || y >= inTexBase.resolution.y)
    {
        return;
    }

    int idx = y * inTexBase.resolution.x + x;
    glm::vec3 baseRgb(inTexBase.dev_pixels[idx]);
    glm::vec3 processedCol(inTexProcessed.dev_pixels[idx]);

    glm::vec3 fullRgb(baseRgb + processedCol);
    glm::vec3 outRgb;
    if (mix < 0.f)
    {
        outRgb = glm::mix(fullRgb, baseRgb, -mix);
    }
    else
    {
        outRgb = glm::mix(fullRgb, processedCol, mix);
    }

    outTex.dev_pixels[idx] = glm::vec4(outRgb, 1);
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
        return NodeUI::FloatEdit(constParams.threshold, 0.01f, 0.f, FLT_MAX);
    case 2: // size
        ImGui::SameLine();
        return NodeUI::IntEdit(constParams.size, 0.02f, sizeMin, sizeMax);
    case 3: // mix
        ImGui::SameLine();
        return NodeUI::FloatEdit(constParams.mix, 0.01f, -1.f, 1.f);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeBloom::_evaluate()
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
    const dim3 blocksPerGrid = calculateNumBlocksPerGrid(inTex->resolution, blockSize);

    kernCopyWithThreshold<<<blocksPerGrid, blockSize>>>(*inTex, constParams.threshold, *outTex1);

    const int kernelRadius = 1 << constParams.size;
    const int kernelDiameter = 2 * kernelRadius + 1;
    NppiSize oKernelSize = { kernelDiameter, kernelDiameter };

    float*& dev_kernel = dev_bloomKernels[constParams.size - sizeMin];

    if (dev_kernel == nullptr)
    {
        cudaMalloc(&dev_kernel, kernelDiameter * kernelDiameter * sizeof(float));

        const float scale = (1.f / 256.f) * powf(kernelDiameter, 2.38f); // last parameter is manually adjusted to get good visual results
        kernFillBlurKernel<<<blocksPerGrid, blockSize>>>(dev_kernel, kernelDiameter, scale);
    }

    const int width = outTex1->resolution.x;
    const int height = outTex1->resolution.y;
    NppiSize oSrcSize = { width, height };
    NppiPoint oSrcOffset = { 0, 0 };

    NppiSize oSizeROI = { width, height };

    NppiPoint oAnchor = { kernelRadius, kernelRadius };

    NPP_CHECK(nppiFilterBorder_32f_C4R(
        (Npp32f*)outTex1->dev_pixels, width * 4 * sizeof(float),
        oSrcSize, oSrcOffset,
        (Npp32f*)outTex2->dev_pixels, width * 4 * sizeof(float),
        oSizeROI,
        (Npp32f*)dev_kernel, oKernelSize, oAnchor,
        NPP_BORDER_REPLICATE
    ));
    std::swap(outTex1, outTex2);

    kernAdd<<<blocksPerGrid, blockSize>>>(*inTex, *outTex1, constParams.mix, *outTex2);

    outputPins[0].propagateTexture(outTex2);
}
