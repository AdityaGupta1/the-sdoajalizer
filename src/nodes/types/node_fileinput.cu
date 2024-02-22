#include "node_fileinput.hpp"

#include "cuda_includes.hpp"

#include "stb_image.h"
#include <filesystem>

std::vector<const char*> NodeFileInput::colorSpaceOptions = { "linear", "sRGB" };

NodeFileInput::NodeFileInput()
    : Node("file input")
{
    addPin(PinType::OUTPUT, "image");

    addPin(PinType::INPUT, "color space").setNoConnect();
}

void NodeFileInput::reloadFile()
{
    if (texFile != nullptr) {
        --texFile->numReferences;
        texFile = nullptr;
    }

    if (selectedColorSpace == 0) // linear
    {
        stbi_ldr_to_hdr_gamma(1.0f);
    }
    else // sRGB
    {
        stbi_ldr_to_hdr_gamma(2.2f);
    }

    int width, height, channels;
    float* host_pixels = stbi_loadf(filePath.c_str(), &width, &height, &channels, 4);

    if (host_pixels == nullptr) {
        return;
    }

    texFile = nodeEvaluator->requestTexture(glm::ivec2(width, height));
    CUDA_CHECK(cudaMemcpy(texFile->dev_pixels, host_pixels, width * height * 4 * sizeof(float), cudaMemcpyHostToDevice));

    stbi_image_free(host_pixels);
}

bool NodeFileInput::drawPinExtras(const Pin* pin, int pinNumber)
{
    ImGui::SameLine();

    bool didParameterChange;

    if (pin->pinType == PinType::INPUT)
    {
        switch (pinNumber)
        {
        case 0: // color space
            didParameterChange = NodeUI::Dropdown(selectedColorSpace, colorSpaceOptions);
            break;
        default:
            throw std::runtime_error("invalid pin number");
        }
    }
    else
    {
        switch (pinNumber)
        {
        case 0: // file input
            didParameterChange = NodeUI::FilePicker(&filePath);

            if (didParameterChange)
            {
                std::string extension = std::filesystem::path(filePath).extension().string();
                if (extension == ".exr")
                {
                    selectedColorSpace = 0; // linear
                }
                else
                {
                    selectedColorSpace = 1; // sRGB
                }
            }

            break;
        default:
            throw std::runtime_error("invalid pin number");
        }
    }

    if (didParameterChange) {
        needsReloadFile = true;
    }
    return didParameterChange;
}

void NodeFileInput::evaluate()
{
    if (needsReloadFile) {
        needsReloadFile = false;
        reloadFile();
    }

    Texture* outTex;
    if (texFile == nullptr)
    {
        outTex = nodeEvaluator->requestSingleColorTexture();
        outTex->setSingleColor(glm::vec4(0, 0, 0, 1));
    }
    else
    {
        outTex = texFile;
    }

    ++outTex->numReferences; // cache this texture
    outputPins[0].propagateTexture(outTex);
}

std::string NodeFileInput::debugGetSrcFileName() const
{
    return __FILE__;
}
