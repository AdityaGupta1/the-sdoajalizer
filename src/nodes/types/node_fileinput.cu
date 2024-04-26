#include "node_fileinput.hpp"

#include "cuda_includes.hpp"

#include "stb_image.h"
#include "tinyexr.h"
#include <filesystem>

std::vector<const char*> NodeFileInput::colorSpaceOptions = { "linear", "sRGB" };

NodeFileInput::NodeFileInput()
    : Node("file input")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "color space").setNoConnect();

    setExpensive();
}

unsigned int NodeFileInput::getTitleBarColor() const
{
    return IM_COL32(7, 94, 11, 255);
}

unsigned int NodeFileInput::getTitleBarHoveredColor() const
{
    return IM_COL32(47, 153, 53, 255);
}

bool NodeFileInput::drawPinExtras(const Pin* pin, int pinNumber)
{
    ImGui::SameLine();

    if (pin->pinType == PinType::INPUT)
    {
        switch (pinNumber)
        {
        case 0: // color space
            return NodeUI::Dropdown(selectedColorSpace, colorSpaceOptions);
        default:
            throw std::runtime_error("invalid pin number");
        }
    }
    else
    {
        switch (pinNumber)
        {
        case 0: // file input
        {
            bool didParameterChange = NodeUI::FilePicker(&filePath, { "Image Files (.png, .jpg, .jpeg, .exr)", "*.png *.jpg *.jpeg *.exr" });

            if (didParameterChange)
            {
                selectedColorSpace = isFileExr() ? 0 : 1; // linear if EXR, sRGB otherwise
            }

            return didParameterChange;
        }
        default:
            throw std::runtime_error("invalid pin number");
        }
    }

    return false;
}

__global__ void kernSrgbToLinear(Texture tex)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= tex.getNumPixels())
    {
        return;
    }

    tex.setColor<TextureType::MULTI>(idx, ColorUtils::srgbToLinear(tex.getColor<TextureType::MULTI>(idx)));
}

bool NodeFileInput::isFileExr() const
{
    return std::filesystem::path(filePath).extension().string() == ".exr";
}

void NodeFileInput::_evaluate()
{
    bool isExr = isFileExr();

    float* host_pixels = nullptr;
    int width, height;
    if (isExr)
    {
        const char* err = nullptr;

        int ret = LoadEXR(&host_pixels, &width, &height, filePath.c_str(), &err);

        if (ret != TINYEXR_SUCCESS)
        {
            if (err)
            {
                fprintf(stderr, "ERR : %s\n", err);
                FreeEXRErrorMessage(err);
            }

            return;
        }
    }
    else
    {
        stbi_ldr_to_hdr_gamma(selectedColorSpace == 0 ? 1.0f : 2.2f); // 1.0f if linear, 2.2f if sRGB

        int channels;
        host_pixels = stbi_loadf(filePath.c_str(), &width, &height, &channels, 4);
    }

    if (host_pixels == nullptr)
    {
        return;
    }

    glm::ivec2 resolution = glm::ivec2(width, height);
    int numPixels = width * height;

    Texture* outTex = nodeEvaluator->requestTexture<TextureType::MULTI>(resolution);
    CUDA_CHECK(cudaMemcpy(outTex->getDevPixels<TextureType::MULTI>(), host_pixels, numPixels * 4 * sizeof(float), cudaMemcpyHostToDevice));

    if (isExr)
    {
        free(host_pixels);

        if (selectedColorSpace == 1) // sRGB
        {
            const dim3 blockSize(DEFAULT_BLOCK_SIZE_1D);
            const dim3 blocksPerGrid = calculateNumBlocksPerGrid(outTex->getNumPixels(), blockSize);
            kernSrgbToLinear<<<blocksPerGrid, blockSize>>>(*outTex);
        }
    }
    else
    {
        stbi_image_free(host_pixels);
    }

    outputPins[0].propagateTexture(outTex);
}
