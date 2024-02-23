#include "node_bloom.hpp"

#include "cuda_includes.hpp"

NodeBloom::NodeBloom()
    : Node("bloom")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "image");
    addPin(PinType::INPUT, "threshold").setNoConnect();
    addPin(PinType::INPUT, "size").setNoConnect();
    addPin(PinType::INPUT, "mix").setNoConnect();
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

// https://github.com/blender/blender/blob/7da72a938c05fe5662db3654f8dbd02a67c0150b/source/blender/compositor/operations/COM_GlareFogGlowOperation.cc
__global__ void kernBlur(Texture inTex, int blurKernelRadius, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= inTex.resolution.x || y >= inTex.resolution.y)
    {
        return;
    }

    int blurKernelDiameter = 2 * blurKernelRadius + 1;

    float scale = (1.f / 256.f) * powf(blurKernelDiameter, 2.1f);

    glm::vec3 sum = glm::vec3(0.f);

    float u, v, r, d, f, w, weight;
    for (int dy = 0; dy <= 2 * blurKernelRadius; ++dy)
    {
        v = 2.0f * (dy / (float)blurKernelDiameter) - 1.0f;

        for (int dx = 0; dx <= 2 * blurKernelRadius; ++dx)
        {
            u = 2.0f * (dx / (float)blurKernelDiameter) - 1.0f;
            r = (u * u + v * v) * scale;
            d = -powf(r, 0.0625f) * 9.0f;
            f = expf(d);
            w = (0.5f + 0.5f * cosf(u * glm::pi<float>())) * (0.5f + 0.5f * cosf(v * glm::pi<float>()));
            weight = f * w;

            glm::ivec2 loadPos(x + dx - blurKernelRadius, y + dy - blurKernelRadius);
            loadPos = glm::clamp(loadPos, glm::ivec2(0), inTex.resolution - 1);

            glm::vec3 sampleColor = glm::vec3(inTex.dev_pixels[loadPos.y * inTex.resolution.x + loadPos.x]);
            sum += sampleColor * weight;
        }
    }

    int idx = y * inTex.resolution.x + x;
    outTex.dev_pixels[idx] = glm::vec4(sum, 1.f);
}

__global__ void kernAdd(Texture inTex1, Texture inTex2, float mix, Texture outTex)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= inTex1.resolution.x || y >= inTex1.resolution.y)
    {
        return;
    }

    int idx = y * inTex1.resolution.x + x;
    glm::vec4 baseCol = inTex1.dev_pixels[idx];
    glm::vec4 processedCol = inTex2.dev_pixels[idx];

    glm::vec4 fullCol = glm::vec4(glm::vec3(baseCol) + glm::vec3(processedCol), baseCol.a);
    glm::vec4 outCol;
    if (mix < 0.f)
    {
        outCol = glm::mix(fullCol, baseCol, -mix);
    }
    else
    {
        outCol = glm::mix(fullCol, processedCol, mix);
    }

    outTex.dev_pixels[idx] = outCol;
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
    case 2: // size
        ImGui::SameLine();
        return NodeUI::IntEdit(backupSize, 0.02f, sizeMin, sizeMax);
    case 3: // mix
        ImGui::SameLine();
        return NodeUI::FloatEdit(backupMix, 0.01f, -1.f, 1.f);
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

    kernBlur<<<blocksPerGrid, blockSize>>>(*outTex1, 1 << backupSize, *outTex2);
    std::swap(outTex1, outTex2);

    kernAdd<<<blocksPerGrid, blockSize>>>(*inTex, *outTex1, backupMix, *outTex2);

    outputPins[0].propagateTexture(outTex2);
}

std::string NodeBloom::debugGetSrcFileName() const
{
    return __FILE__;
}

