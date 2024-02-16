#include "node_fileinput.hpp"

#include "cuda_includes.hpp"

#include "stb_image.h"

NodeFileInput::NodeFileInput()
    : Node("file input")
{
    addPin(PinType::OUTPUT);
}

void NodeFileInput::reloadFile()
{
    if (texFile != nullptr) {
        --texFile->numReferences;
        texFile = nullptr;
    }

    int width, height, channels;
    float* host_pixels = stbi_loadf(filePath.c_str(), &width, &height, &channels, 4);

    if (host_pixels == nullptr) {
        return;
    }

    texFile = nodeEvaluator->requestTexture(glm::ivec2(width, height));
    cudaMemcpy(texFile->dev_pixels, host_pixels, width * height * 4 * sizeof(float), cudaMemcpyHostToDevice);

    stbi_image_free(host_pixels);
}

bool NodeFileInput::drawPinExtras(const Pin* pin, int pinNumber)
{
    ImGui::SameLine();
    bool didParameterChange = NodeUI::FilePicker(&filePath);
    needsReloadFile = didParameterChange;
    return didParameterChange;
}

void NodeFileInput::evaluate()
{
    if (needsReloadFile) {
        needsReloadFile = false;
        reloadFile();
    }

    Texture* outTex = (texFile == nullptr) ? nodeEvaluator->requestSingleColorTexture() : texFile;
    outputPins[0].propagateTexture(outTex);
}
