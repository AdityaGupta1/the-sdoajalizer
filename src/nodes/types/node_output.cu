#include "node_output.hpp"

#include "cuda_includes.hpp"

std::vector<const char*> NodeOutput::toneMappingOptions = { "none", "reinhard", "ACES filmic", "AgX"};

NodeOutput::NodeOutput()
    : Node("output")
{
    addPin(PinType::INPUT, "image");
    addPin(PinType::INPUT, "tone mapping").setNoConnect();
}

unsigned int NodeOutput::getTitleBarColor() const
{
    return IM_COL32(255, 85, 0, 255);
}

unsigned int NodeOutput::getTitleBarSelectedColor() const
{
    return IM_COL32(255, 128, 0, 255);
}

__host__ __device__ glm::vec4 hdrToLdr(glm::vec4 col, int toneMapping)
{
    glm::vec3 rgb = glm::max(glm::vec3(col), 0.f);
    
    switch (toneMapping)
    {
    case 0:
        break;
    case 1:
        rgb = ColorUtils::reinhard(rgb);
        break;
    case 2:
        rgb = ColorUtils::ACESFilm(rgb);
        break;
    case 3:
        rgb = ColorUtils::AgX(rgb);
        break;
    }

    return glm::vec4(ColorUtils::linearToSrgb(rgb), col.a);
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

__global__ void kernCopyToOutTex(Texture inTex, int toneMapping, Texture outTex)
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
        col = hdrToLdr(inTex.dev_pixels[y * inTex.resolution.x + x], toneMapping);
    }

    outTex.dev_pixels[y * outTex.resolution.x + x] = col;
}

bool NodeOutput::drawPinExtras(const Pin* pin, int pinNumber)
{
    if (pinNumber == 1)
    {
        ImGui::SameLine();
        return NodeUI::Dropdown(selectedToneMapping, toneMappingOptions);
    }

    return false;
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
        auto ldrCol = hdrToLdr(inTex->singleColor, selectedToneMapping);
        kernFillSingleColor<<<blocksPerGrid, blockSize>>>(*outTex, ldrCol);
    }
    else
    {
        kernCopyToOutTex<<<blocksPerGrid, blockSize>>>(*inTex, selectedToneMapping, *outTex);
    }

    nodeEvaluator->setOutputTexture(outTex);
}

std::string NodeOutput::debugGetSrcFileName() const
{
    return __FILE__;
}

